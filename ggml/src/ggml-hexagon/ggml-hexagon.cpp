#ವ್ಯಾಖ್ಯಾನಿಸು USE_MATH_ನಿರೂಪಿಸುತ್ತದೆ

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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#    define WIN32_LEAN_AND_ಸರಾಸರಿ
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#ಇಲ್ಲದಿದ್ದರೆ
#    include <semaphore.h>
#    include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಸೂಚನೆ clang diagnostic ignored "-Wnested-anon-types"
#ಸೂಚನೆ clang diagnostic ignored "-Wlanguage-extension-token"
#ಸೂಚನೆ clang diagnostic ignored "-Wgnu-anonymous-struct"
#ಸೂಚನೆ clang diagnostic ignored "-Wmicrosoft-enum-value"

#ಅನ್ತರ್ಭಾವ <AEEStdErr.h>
#ಅನ್ತರ್ಭಾವ <dspqueue.h>
#ಅನ್ತರ್ಭಾವ <rpcmem.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
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

ಬಳಸಿ intvec  = std::vector<ಪೂರ್ಣಾಂಕ>;
ಬಳಸಿ uintvec = std::vector<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>;
ಬಳಸಿ u32vec  = std::vector<uint32_t>;

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_HEXAGON_MAX_SESSIONS          16

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ       128

ರಚನೆ ggml_hexagon_device_config {
    ಪೂರ್ಣಾಂಕ         physical_idx = 0;
    ಪೂರ್ಣಾಂಕ         virtual_idx  = 0;
    ಪೂರ್ಣಾಂಕ         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_ಗುಂಪು;
};

ಸ್ಥಿರ ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_arch    = 0; // autodetect
ಸ್ಥಿರ size_t opt_ndev    = 1;
ಸ್ಥಿರ size_t opt_nhvx    = 0; // use all
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ಸ್ಥಿರ size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_etm     = 0;
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_verbose = 0;
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ಸ್ಥಿರ ತಾರ್ಕಿಕ   opt_hostbuf = ಸುಳ್ಳು;
ಸ್ಥಿರ ತಾರ್ಕಿಕ   opt_dma64   = ಸುಳ್ಳು;
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_power_ಬಹುಲಕ = HTP_POWER_ಬಹುಲಕ_ಅವಧಿ; // DCVS/HMX power profile requested at session start

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::pair<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *, ಪೂರ್ಣಾಂಕ> htp_power_ಬಹುಲಕ_ಹೆಸರುಗಳು[] = {
    { "low_power_saver",            HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_POWER_SAVER },
    { "power_saver",                HTP_POWER_ಬಹುಲಕ_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_ಸಮತೋಲಿತ },
    { "balanced",                   HTP_POWER_ಬಹುಲಕ_ಸಮತೋಲಿತ },
    { "high_performance",           HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_ಬಹುಲಕ_SUSTAINED_ಮೇಲಿನ_PERFORMANCE },
    { "burst",                      HTP_POWER_ಬಹುಲಕ_ಅವಧಿ },
    { "default",                    HTP_POWER_ಬಹುಲಕ_ಅವಧಿ }, // flip this to change the default mode
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ htp_power_ಬಹುಲಕ_from_str(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * s) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & e : htp_power_ಬಹುಲಕ_ಹೆಸರುಗಳು) {
#ಆದರೆ defined(_MSC_VER)
        ಆದರೆ (_stricmp(s, e.ಮೊದಲ) == 0) {
#ಇಲ್ಲದಿದ್ದರೆ
        ಆದರೆ (strcasecmp(s, e.ಮೊದಲ) == 0) {
#ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ e.second;
        }
    }
    ಹಿಂತಿರುಗಿ -1;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * htp_power_ಬಹುಲಕ_ಗಮ್ಯ_str(ಪೂರ್ಣಾಂಕ mode) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & e : htp_power_ಬಹುಲಕ_ಹೆಸರುಗಳು) {
        ಆದರೆ (e.second == mode) {
            ಹಿಂತಿರುಗಿ e.ಮೊದಲ;
        }
    }
    ಹಿಂತಿರುಗಿ "unknown";
}

ಸ್ಥಿರ size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ಸ್ಥಿರ size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ಸ್ಥಿರ size_t opt_mbuf_ಒಟ್ಟು  = 0;                        // total buffer space limit (0 = unconstrained)

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_fa_ತಲೆ_ವಿಭಜಿಸು = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ಸ್ಥಿರ u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_opbatch  = 1280; // max number of ops in a batch
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_opqueue  = 8;   // max number of pending batches
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_oppoll   = 0;    // polling for batch completions
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_opfusion  = 1;    // enable/disable op fusion
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

ಎಣಿಕೆ ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ಕೂಡಿಸು = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ಕೂಡಿಸು   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_is_fusion_enabled(ಪೂರ್ಣಾಂಕ flag) {
    ಆದರೆ (opt_opfusion <= 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (opt_opfusion == 1) ಹಿಂತಿರುಗಿ ನಿಜ; // 1 enables all
    ಹಿಂತಿರುಗಿ (opt_opfusion & flag) != 0;
}

ಸ್ಥಿರ std::regex* opt_opfilter = ಖಾಲಿ; // regex of ops to not claim

#ವ್ಯಾಖ್ಯಾನಿಸು HEX_VERBOSE(...) \
    ಆದರೆ (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿತಿ_ಗಮ್ಯ_str(uint32_t ಸ್ಥಿತಿ_2) {
    ಆಯ್ಕೆ (ಸ್ಥಿತಿ_2) {
        ಸ್ಥಿತಿ HTP_ಸ್ಥಿತಿ_OK:
            ಹಿಂತಿರುಗಿ "OK";
        ಸ್ಥಿತಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT:
            ಹಿಂತಿರುಗಿ "NO-SUPPORT";
        ಸ್ಥಿತಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS:
            ಹಿಂತಿರುಗಿ "INVAL-PARAMS";
        ಸ್ಥಿತಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL:
            ಹಿಂತಿರುಗಿ "VTCM-TOO-SMALL";
        ಸ್ಥಿತಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR:
            ಹಿಂತಿರುಗಿ "INTERNAL-ERROR";
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ "UNKNOWN";
    }
}

// ** debug helpers

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_exec(ಸ್ಥಿರಾಂಕ std::string &sess_name, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು, ಸ್ಥಿರಾಂಕ uint32_t req_flags) {
    ಆದರೆ (!opt_verbose) ಹಿಂತಿರುಗಿ;

    htp_opformat fmt(ಗಂಟು);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                ಗಂಟು.op_name().c_str(), fmt.ಹೆಸರುಗಳು, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_supp(ಸ್ಥಿರಾಂಕ std::string &sess_name, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ತಾರ್ಕಿಕ supp) {
    ಆದರೆ (!opt_verbose) ಹಿಂತಿರುಗಿ;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.ಹೆಸರುಗಳು, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * htp_event_name(uint16_t id) {
    ಆಯ್ಕೆ (id) {
        ಸ್ಥಿತಿ HTP_TRACE_EVT_DMA:            ಹಿಂತಿರುಗಿ "DMA";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_COMP:       ಹಿಂತಿರುಗಿ "HVX_COMP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_A_QUANT:    ಹಿಂತಿರುಗಿ "HVX_A_QUANT";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_A_PREP:     ಹಿಂತಿರುಗಿ "HVX_A_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_W_DEQUANT:  ಹಿಂತಿರುಗಿ "HVX_W_DEQUANT";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_W_PREP:     ಹಿಂತಿರುಗಿ "HVX_W_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_O_PROC:     ಹಿಂತಿರುಗಿ "HVX_O_PROC";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_FA_QK:      ಹಿಂತಿರುಗಿ "HVX_QK_FA";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_FA_SFM:     ಹಿಂತಿರುಗಿ "HVX_SFM_FA";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_FA_Q_PREP:  ಹಿಂತಿರುಗಿ "HVX_Q_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_FA_K_PREP:  ಹಿಂತಿರುಗಿ "HVX_K_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_FA_V_PREP:  ಹಿಂತಿರುಗಿ "HVX_V_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_PREP:   ಹಿಂತಿರುಗಿ "HVX_GDN_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_SOLVE:  ಹಿಂತಿರುಗಿ "HVX_GDN_SOLVE";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_V_PREP: ಹಿಂತಿರುಗಿ "HVX_GDN_V_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_D_PREP: ಹಿಂತಿರುಗಿ "HVX_GDN_D_PREP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_ಹೊರಹೋಗು:    ಹಿಂತಿರುಗಿ "HVX_GDN_OUT";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_STATE:  ಹಿಂತಿರುಗಿ "HVX_GDN_STATE";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HVX_GDN_REM:    ಹಿಂತಿರುಗಿ "HVX_GDN_REM";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_HMX_COMP:       ಹಿಂತಿರುಗಿ "HMX_COMP";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_L2FLUSH:        ಹಿಂತಿರುಗಿ "L2FLUSH";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_INIT:           ಹಿಂತಿರುಗಿ "INIT";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_BUFF:           ಹಿಂತಿರುಗಿ "BUFF";
        ಸ್ಥಿತಿ HTP_TRACE_EVT_FENCE:          ಹಿಂತಿರುಗಿ "FENCE";
        ಪೂರ್ವನಿಯೋಜಿತ:                           ಹಿಂತಿರುಗಿ "UNKNOWN";
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_prof(ಸ್ಥಿರಾಂಕ std::string &sess_name, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು, ಸ್ಥಿರಾಂಕ htp_prof_desc & pd) {
    ಆದರೆ (!opt_profile) ಹಿಂತಿರುಗಿ;

    uint32_t op_usec = pd.usecs;
    uint32_t op_ಚಕ್ರಗಳು = pd.ಚಕ್ರಗಳು_stop - pd.ಚಕ್ರಗಳು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t * pmu = pd.pmu;

    ಅಕ್ಷರ pmu_str[256] = "";
    ಆದರೆ (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, ಗಾತ್ರ(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(ಗಂಟು);
    ದಶಮಾಂಶ mhz = op_usec > 0 ? (ದಶಮಾಂಶ) op_ಚಕ್ರಗಳು / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            ಗಂಟು.op_name().c_str(), fmt.ಹೆಸರುಗಳು, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_ಚಕ್ರಗಳು, pd.ಚಕ್ರಗಳು_ಆರಂಭ, mhz, pmu_str);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_batch_prof(ಸ್ಥಿರಾಂಕ std::string & sess_name, ಸ್ಥಿರಾಂಕ htp_opbatch_rsp & rsp) {
    uint64_t batch_ಚಕ್ರಗಳು = rsp.ಚಕ್ರಗಳು_stop - rsp.ಚಕ್ರಗಳು_ಆರಂಭ;
    ದಶಮಾಂಶ batch_mhz = rsp.usecs > 0 ? (ದಶಮಾಂಶ) batch_ಚಕ್ರಗಳು / rsp.usecs : 0.0f;

    ಅಕ್ಷರ evt_str[256] = "----";
    ಆದರೆ (opt_profile == 3) {
        snprintf(evt_str, ಗಾತ್ರ(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) batch_ಚಕ್ರಗಳು, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) rsp.ಚಕ್ರಗಳು_ಆರಂಭ, batch_mhz);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_trace_events(ಸ್ಥಿರಾಂಕ std::string & sess_name, ಸ್ಥಿರಾಂಕ htp_opbatch_rsp & rsp,
                                           ಸ್ಥಿರಾಂಕ htp_trace_desc * trace_events, uint32_t n_traces) {
    ಆದರೆ (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        ಗಾಗಿ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        ಗಾಗಿ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ಗಾಗಿ (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & e = trace_events[t * n_traces + idx];
                ತಾರ್ಕಿಕ is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.ಚಕ್ರಗಳು);
            }
        }
    }
}

ಎಣಿಕೆ ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_ತೂಕ    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ರಚನೆ ggml_hexagon_session;

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_matmul_ಕೂಡಿಸು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_unary_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    uint32_t op,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_unary_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    uint32_t op,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params * kparams,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = ಶೂನ್ಯಸೂಚಕ
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_get_ಅಡ್ಡಸಾಲುಗಳು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_softmax_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_softmax_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_rope_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_rope_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ssm_conv_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_ssm_conv_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_gated_delta_net_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_gdn_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ವಿಂಗಡಿಸು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ತಾರ್ಕಿಕ is_ತುದಿ_k,
    ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_pool_2d_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_pool_2d_kernel_params * kparams,
    ತಾರ್ಕಿಕ is_pool_1d
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_concat_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_concat_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_cpy_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_copy_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_im2col_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_3d_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_im2col_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmnx_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * act,
    int32_t n_weights,
    ರಚನೆ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmidnx_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * act,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    int32_t n_weights,
    ರಚನೆ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_allreduce_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ತಾರ್ಕಿಕ has_ಕೂಡಿಸು,
    ತಾರ್ಕಿಕ is_ಅಡ್ಡಸಾಲು_bcast,
    ತಾರ್ಕಿಕ is_shard_ok,
    ರಚನೆ htp_allreduce_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(ಸ್ಥಿರಾಂಕ int64_t * ne, ಸ್ಥಿರಾಂಕ size_t * nb, size_t * stride);
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_can_collapse(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst);
ಸ್ಥಿರ ggml_tensor ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t);

ಸ್ಥಿರ ತಾರ್ಕಿಕ mm_is_hmx_eligible(ಸ್ಥಿರಾಂಕ ggml_tensor * t);
ಸ್ಥಿರ htp_op_ಸಂಕೇತ op_remap_ಗಮ್ಯ_htp(ಸ್ಥಿರಾಂಕ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_nx_kernel(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_id_nx_kernel(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat(ಸ್ಥಿರಾಂಕ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_pair(ಸ್ಥಿರಾಂಕ ggml_tensor * n1, ಸ್ಥಿರಾಂಕ ggml_tensor * n2);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id(ಸ್ಥಿರಾಂಕ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id_pair(ಸ್ಥಿರಾಂಕ ggml_tensor * n1, ಸ್ಥಿರಾಂಕ ggml_tensor * n2);

// ** backend sessions

ರಚನೆ ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_ಗಾತ್ರ { 0 };
    uint32_t             flags { 0 };
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_tensor_is_fuseable(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (!t->extra) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಸ್ವಯಂಚಾಲಿತ extra = (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_tensor_extra *) t->extra;
    ಹಿಂತಿರುಗಿ (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_can_rt_repack(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (opt_rt_repack == 0 || !htp_is_tiled_ತೂಕ_tensor(t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (t->ಬಫರ್ && ggml_backend_buft_is_host(t->ಬಫರ್->buft)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_tensor_is_tiled(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra = (ಸ್ಥಿರಾಂಕ ggml_hexagon_tensor_extra *) t->extra;
    ಆದರೆ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ggml_hexagon_can_rt_repack(t);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_needs_repack(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra = (ಸ್ಥಿರಾಂಕ ggml_hexagon_tensor_extra *) t->extra;
    ಆದರೆ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ggml_hexagon_can_rt_repack(t);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_tensors_overlap(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * b) {
    ಸ್ಥಿರಾಂಕ uintptr_t a0 = (uintptr_t) a->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uintptr_t b0 = (uintptr_t) b->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uintptr_t a1 = a0 + ggml_nbytes(a);
    ಸ್ಥಿರಾಂಕ uintptr_t b1 = b0 + ggml_nbytes(b);

    ಹಿಂತಿರುಗಿ a0 < b1 && b0 < a1;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_hexagon_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (t->ne[1] > 1 && (t->nb[1] & 127) != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (t->ne[2] > 1 && (t->nb[2] & 127) != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (t->ne[3] > 1 && (t->nb[3] & 127) != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ರಚನೆ htp_opnode;

ರಚನೆ ggml_hexagon_opbatch;
ರಚನೆ ggml_hexagon_opqueue;
ರಚನೆ ggml_hexagon_shared_ಬಫರ್;
ರಚನೆ ggml_hexagon_fence_ಬಫರ್;
ರಚನೆ ggml_hexagon_session;
ರಚನೆ ggml_backend_hexagon_device_context;

ರಚನೆ ggml_hexagon_mdev_ಗುಂಪು {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ರಚನೆ ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    ಅಸ್ಥಿರ uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ರಚನೆ ggml_hexagon_event {
    ggml_hexagon_session * sess         = ಶೂನ್ಯಸೂಚಕ;
    ggml_hexagon_session * fence_sess   = ಶೂನ್ಯಸೂಚಕ;
    ಅಸ್ಥಿರ uint32_t *    fence_ಸ್ಥಾನ   = ಶೂನ್ಯಸೂಚಕ;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ರಚನೆ ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         ಸರತಿ_id;
    ಪೂರ್ಣಾಂಕ              phys_idx;
    ಪೂರ್ಣಾಂಕ              virt_idx;
    ತಾರ್ಕಿಕ             valid_session;
    ತಾರ್ಕಿಕ             valid_handle;
    ತಾರ್ಕಿಕ             valid_ಸರತಿ;
    ತಾರ್ಕಿಕ             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_ಸರತಿ;

    std::unordered_map<ಪೂರ್ಣಾಂಕ, std::unique_ptr<ggml_hexagon_shared_ಬಫರ್>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_ಎಳೆಗಳು   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_ಗಾತ್ರ   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> ಕೊನೆಯ_error{HTP_ಸ್ಥಿತಿ_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_ಗುಂಪು                mdev;
    ggml_backend_dev_t                     dev       = ಶೂನ್ಯಸೂಚಕ;
    ggml_backend_hexagon_device_context *  dev_ctx   = ಶೂನ್ಯಸೂಚಕ;
    ggml_hexagon_fence_ಬಫರ್ *            fence_buf = ಶೂನ್ಯಸೂಚಕ;

    ggml_hexagon_session(ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config, ggml_backend_dev_t dev = ಶೂನ್ಯಸೂಚಕ, uint32_t mdev_idx = 0, uint32_t mdev_ಎಣಿಕೆ = 0) noexcept(ಸುಳ್ಳು);
    ~ggml_hexagon_session() noexcept(ನಿಜ);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* c_name() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ name.c_str(); }

    ಶೂನ್ಯ allocate(ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config) noexcept(ಸುಳ್ಳು);
    ಶೂನ್ಯ release() noexcept(ನಿಜ);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    ಶೂನ್ಯ      free_fence(ಶೂನ್ಯ * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_ಸ್ಥಾನ = ಶೂನ್ಯಸೂಚಕ;
    std::unordered_map<uint64_t, ಅಸ್ಥಿರ uint32_t *> cpy_fence_slots;

    ಶೂನ್ಯ ಸರತಿಗೆ_ಸೇರಿಸು_mdev_ಗುಂಪು();
    ಶೂನ್ಯ ಸರತಿಗೆ_ಸೇರಿಸು_op(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು);
    ಶೂನ್ಯ ಸರತಿಗೆ_ಸೇರಿಸು_cpy(ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ggml_tensor * sync_tensor = ಶೂನ್ಯಸೂಚಕ, uint32_t fence_seq = 0);
    ಶೂನ್ಯ ಸರತಿಗೆ_ಸೇರಿಸು_fence(ಸ್ಥಿರಾಂಕ ggml_tensor * sync_tensor, uint32_t fence_seq = 0, ತಾರ್ಕಿಕ wait = ನಿಜ);
    ಶೂನ್ಯ ಸರತಿಗೆ_ಸೇರಿಸು_allreduce(ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> & src_tensors,
                           ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_ನಮೂದು = 0, uint32_t fence_seq_exit = 0);

    ಶೂನ್ಯ ಆರಂಭ_batch();
    ಶೂನ್ಯ flush_sync(ತಾರ್ಕಿಕ all = ನಿಜ);
    ಶೂನ್ಯ flush_async();
    ಶೂನ್ಯ flush_batch(size_t min_ops = 1);
    ಶೂನ್ಯ flush_peers();
    ಶೂನ್ಯ flush_pending(ತಾರ್ಕಿಕ all = ನಿಜ);

    ggml_hexagon_shared_ಬಫರ್ * mmap_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * t);
    ತಾರ್ಕಿಕ clone_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್*);
    ಶೂನ್ಯ release_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್*);
    ಶೂನ್ಯ unclone_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್*);

    ಶೂನ್ಯ ಕೂಡಿಸು_peer(ggml_hexagon_session * peer) {
        ಆದರೆ (ಇದು->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } ಇಲ್ಲದಿದ್ದರೆ {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ರಚನೆ ggml_backend_hexagon_device_context {
    ಪೂರ್ಣಾಂಕ                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = ಶೂನ್ಯಸೂಚಕ;

    ggml_backend_ಬಫರ್_type ಬಫರ್_type       = {};
    ggml_backend_ಬಫರ್_type host_ಬಫರ್_type  = {};
    ggml_backend_ಬಫರ್_type fence_ಬಫರ್_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(ಪೂರ್ಣಾಂಕ dev_id, ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * c_name() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ config.name.c_str(); }

    ggml_hexagon_session * session() {
        ಆದರೆ (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        ಹಿಂತಿರುಗಿ sess.get();
    }
};

ರಚನೆ ggml_backend_hexagon_ಬಫರ್_type_context {
    ggml_backend_hexagon_ಬಫರ್_type_context(ಸ್ಥಿರಾಂಕ std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ಇದು->dev_ctx = dev_ctx;
        ಇದು->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ರಚನೆ ggml_hexagon_rpcmem_ಖಂಡ {
    uint8_t * ತಳ = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_ಖಂಡ(size_t size) {
        ತಳ = (uint8_t *) rpcmem_alloc2(RPCMEM_ರಾಶಿ_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        ಆದರೆ (!ತಳ) {
            ಎಸೆ std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_ಗಮ್ಯ_fd(ತಳ);
        ಆದರೆ (fd < 0) {
            rpcmem_free(ತಳ);
            ಎಸೆ std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ಇದು->size = size;
    }

    ~ggml_hexagon_rpcmem_ಖಂಡ() {
        ಆದರೆ (ತಳ) {
            rpcmem_free(ತಳ);
        }
    }
};

ರಚನೆ ggml_hexagon_shared_ಬಫರ್ {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_ಖಂಡ> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    ತಾರ್ಕಿಕ     mapped;
    ತಾರ್ಕಿಕ     pinned;
    ತಾರ್ಕಿಕ     extended;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * c_name() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ sess->c_name(); }
    uint8_t *    ತಳ()   ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ mem ? mem->ತಳ : ಶೂನ್ಯಸೂಚಕ; }
    size_t       size()   ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ mem ? mem->size : 0;  }
    ಪೂರ್ಣಾಂಕ          fd()     ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ mem ? mem->fd   : -1; }

    ಶೂನ್ಯ mmap(ತಾರ್ಕಿಕ extended = ಸುಳ್ಳು) {
        ಆದರೆ (!ಇದು->mem)   ಹಿಂತಿರುಗಿ;
        ಆದರೆ (ಇದು->mapped) ಹಿಂತಿರುಗಿ;

        GGML_ASSERT(!ಇದು->pinned || !extended);

        ಇದು->extended = extended;
        fastrpc_map_flags flags = ಇದು->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        ಪೂರ್ಣಾಂಕ err = fastrpc_mmap(sess->domain_id, fd(), (ಶೂನ್ಯ *) ತಳ(), 0, size(), flags);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (ಚಿಹ್ನೆರಹಿತ) err);
            ಎಸೆ std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (ಶೂನ್ಯ *) ತಳ(), size(), fd(), pinned, extended);

        ಇದು->mapped = ನಿಜ;
    }

    ಶೂನ್ಯ unmap() {
        ಆದರೆ (!ಇದು->mapped) ಹಿಂತಿರುಗಿ;

        ಆದರೆ (!ಇದು->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        ಆದರೆ (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (ಶೂನ್ಯ *) ತಳ(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (ಶೂನ್ಯ *) ತಳ(), size(), fd());

        ಇದು->mapped = ಸುಳ್ಳು;
    }

    ಶೂನ್ಯ alloc(size_t size) {
        ಆದರೆ (ಇದು->mem) ಹಿಂತಿರುಗಿ;

        ಇದು->mem = std::make_shared<ggml_hexagon_rpcmem_ಖಂಡ>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (ಶೂನ್ಯ *) ತಳ(), ಇದು->size(), fd(), (ಪೂರ್ಣಾಂಕ) pinned);
        ಆದರೆ (ಇದು->pinned) {
            mmap();
        }
    }

    ಶೂನ್ಯ ಬಿಡುಗಡೆ() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (ಶೂನ್ಯ *) ತಳ(), size(), fd());
        ಇದು->mem  = ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_hexagon_shared_ಬಫರ್(ggml_hexagon_session * sess, size_t size, ತಾರ್ಕಿಕ pinned = ಸುಳ್ಳು) {
        ಇದು->sess     = sess;
        ಇದು->mapped   = ಸುಳ್ಳು;
        ಇದು->pinned   = pinned;
        ಇದು->extended = ಸುಳ್ಳು;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_ಸ್ಥಾನಾಂತರ = (size + 4095) & ~4095;
        size_t ಒಟ್ಟು_ಗಾತ್ರ   = guard_ಸ್ಥಾನಾಂತರ + 4096;
        ಆದರೆ (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            ಒಟ್ಟು_ಗಾತ್ರ = (ಒಟ್ಟು_ಗಾತ್ರ + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(ಒಟ್ಟು_ಗಾತ್ರ);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_ಬಫರ್(ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್ & other) {
        ಇದು->sess   = sess;
        ಇದು->mem    = other.mem;
        ಇದು->mapped = ಸುಳ್ಳು;
        ಇದು->pinned = other.pinned;
        ಇದು->extended = other.extended;
    }

    ~ggml_hexagon_shared_ಬಫರ್() {
        ಬಿಡುಗಡೆ();
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * extra : tensor_extra) {
            ಅಳಿಸು extra;
        }
    }
};

ರಚನೆ ggml_hexagon_fence_ಬಫರ್ : ಸಾರ್ವಜನಿಕ ggml_hexagon_shared_ಬಫರ್ {
    uint32_t              ಸ್ಥಾನ_ಎಣಿಕೆ = 0;
    uint32_t              ಸ್ಥಾನ_ತಲೆ  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_ಬಫರ್   backend_ಬಫರ್{};

    ggml_hexagon_fence_ಬಫರ್(ggml_hexagon_session * sess, ggml_backend_ಬಫರ್_type_t buft, size_t size)
        : ggml_hexagon_shared_ಬಫರ್(sess, size, ಸುಳ್ಳು /* pinned */),
          ಸ್ಥಾನ_ಎಣಿಕೆ(size / GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ),
          ಸ್ಥಾನ_ತಲೆ(0) {
        backend_ಬಫರ್.buft    = buft;
        backend_ಬಫರ್.context = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(ಇದು);
        backend_ಬಫರ್.size    = size;
        mmap(ಸುಳ್ಳು);
    }

    uint8_t * alloc_ಸ್ಥಾನ(uint32_t n_slots = 1) {
        uint8_t * ptr = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (n_slots == 1 && !free_slots.ಖಾಲಿಗಳು()) {
            uint32_t ಸ್ಥಾನ = free_slots.back();
            free_slots.pop_back();
            ptr = ತಳ() + (size_t) ಸ್ಥಾನ * GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಸ್ಥಾನ_ತಲೆ + n_slots <= ಸ್ಥಾನ_ಎಣಿಕೆ) {
            uint32_t ಸ್ಥಾನ = ಸ್ಥಾನ_ತಲೆ;
            ಸ್ಥಾನ_ತಲೆ += n_slots;
            ptr = ತಳ() + (size_t) ಸ್ಥಾನ * GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ;
        }
        ಆದರೆ (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ);
        }
        ಹಿಂತಿರುಗಿ ptr;
    }

    ಶೂನ್ಯ free_ಸ್ಥಾನ(ಶೂನ್ಯ * ptr, uint32_t n_slots = 1) {
        ಆದರೆ (!ptr) ಹಿಂತಿರುಗಿ;
        uint32_t ಸ್ಥಾನ = ((uint8_t *) ptr - ತಳ()) / GGML_HEXAGON_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ;
        ಗಾಗಿ (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(ಸ್ಥಾನ + i);
        }
    }
};

ಸಾಲೊಳಗೆ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_ಸ್ಥಾನ(n_slots);
    GGML_ASSERT(ptr);
    ಹಿಂತಿರುಗಿ ptr;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_hexagon_session::free_fence(ಶೂನ್ಯ * ptr, uint32_t n_slots) {
    ಆದರೆ (fence_buf) {
        fence_buf->free_ಸ್ಥಾನ(ptr, n_slots);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(ಬಫರ್->context);
    sbuf->sess->unclone_ಬಫರ್(sbuf);
    ಅಳಿಸು sbuf;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(ಬಫರ್->context);
    ಹಿಂತಿರುಗಿ sbuf->ತಳ();
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_hexagon_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(ಬಫರ್->context);
    ಸ್ವಯಂಚಾಲಿತ sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (ಶೂನ್ಯ *) sbuf->ತಳ(), tensor->ದತ್ತಾಂಶ, ggml_nbytes(tensor));

    ಸ್ವಯಂಚಾಲಿತ extra = ಹೊಸ ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                   ggml_tensor *         tensor,
                                                   ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                   size_t                ಸ್ಥಾನಾಂತರ,
                                                   size_t                size) {
    ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ಸ್ವಯಂಚಾಲಿತ sbuf  = (ggml_hexagon_shared_ಬಫರ್ *) ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ sess  = sbuf->sess;

    ಆದರೆ (ggml_backend_ಬಫರ್_get_usage(ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_ತೂಕ;
        ಆದರೆ (htp_is_tiled_ತೂಕ_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, (ಪೂರ್ಣಾಂಕ) ಬಫರ್->usage, extra->flags);

    ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಸ್ಥಾನಾಂತರ == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.ಖಾಲಿಗಳು()) {
        htp_repack_tensor_tiled(tensor, ದತ್ತಾಂಶ, size);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ಸ್ಮರಣೆ_ನಕಲು(extra->shadow_buf.ದತ್ತಾಂಶ() + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
    extra->shadow_ಗಾತ್ರ += size;

    ಆದರೆ (extra->shadow_ಗಾತ್ರ >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ದತ್ತಾಂಶ(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_ಗಾತ್ರ = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                   ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                   ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                   size_t                ಸ್ಥಾನಾಂತರ,
                                                   size_t                size) {
    ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ಸ್ವಯಂಚಾಲಿತ sbuf  = (ggml_hexagon_shared_ಬಫರ್ *) ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, (ಪೂರ್ಣಾಂಕ) ಬಫರ್->usage, extra->flags);

    ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);
        ಹಿಂತಿರುಗಿ;
    }

    htp_unpack_tensor_tiled(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t      ಬಫರ್,
                                                   ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src,
                                                   ರಚನೆ ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(ಬಫರ್);
    GGML_ಬಳಸದ(src);
    GGML_ಬಳಸದ(dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_ಗಣ_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                          ggml_tensor *         tensor,
                                                          ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                          size_t                ಸ್ಥಾನಾಂತರ,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_ದತ್ತಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ಸ್ವಯಂಚಾಲಿತ sbuf  = (ggml_hexagon_shared_ಬಫರ್ *) ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ sess  = sbuf->sess;

    ಆದರೆ (ggml_backend_ಬಫರ್_get_usage(ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_ತೂಕ;
        ಆದರೆ (htp_is_tiled_ತೂಕ_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ಬಫರ್->usage, extra->flags);

    ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ಸ್ಮರಣೆ_ನಕಲು((uint8_t *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ + i * stride_tensor, (ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, size);
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(extra->shadow_buf.ದತ್ತಾಂಶ() + ಸ್ಥಾನಾಂತರ + i * stride_tensor, (ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, size);
    }
    extra->shadow_ಗಾತ್ರ += n_copies * size;

    ಆದರೆ (extra->shadow_ಗಾತ್ರ >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ದತ್ತಾಂಶ(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_ಗಾತ್ರ = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_get_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                      ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                      ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                      size_t                ಸ್ಥಾನಾಂತರ,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_ದತ್ತಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ಸ್ವಯಂಚಾಲಿತ sbuf  = (ggml_hexagon_shared_ಬಫರ್ *) ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ಬಫರ್->usage);

    ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ಸ್ಮರಣೆ_ನಕಲು((uint8_t *)ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ uint8_t *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ + i * stride_tensor, size);
        }
        ಹಿಂತಿರುಗಿ;
    }

    size_t ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t ತುಂಡು_ಗಾತ್ರ     = tensor->ne[1] * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, tensor->ne[0]);
    size_t ತುಂಡು_ಸ್ಥಾನಾಂತರ   = ಸ್ಥಾನಾಂತರ % ತುಂಡು_ಗಾತ್ರ;
    size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, tensor->ne[0]);

    GGML_ASSERT((ತುಂಡು_ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ % ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((ಸ್ಥಾನಾಂತರ + ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ) <= ggml_nbytes(tensor));

    std::vector<uint8_t> ತಾತ್ಕಾಲಿಕ_buf(ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ);
    htp_unpack_tensor_tiled(ತಾತ್ಕಾಲಿಕ_buf.ದತ್ತಾಂಶ(), tensor, ಸ್ಥಾನಾಂತರ, ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ);

    ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
        ಸ್ಮರಣೆ_ನಕಲು((uint8_t *) ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, ತಾತ್ಕಾಲಿಕ_buf.ದತ್ತಾಂಶ() + i * stride_tensor, size);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ಸ್ವಯಂಚಾಲಿತ sbuf = (ggml_hexagon_shared_ಬಫರ್ *) ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (ಶೂನ್ಯ *) sbuf->ತಳ(), sbuf->size());
    memset(sbuf->ತಳ(), value, sbuf->size());
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_hexagon_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_hexagon_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_hexagon_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_hexagon_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_ಬಫರ್_ಗಣ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_ಬಫರ್_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

// ** backend buffer type

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_host_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                        ggml_tensor *         tensor,
                                                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                        size_t                ಸ್ಥಾನಾಂತರ,
                                                        size_t                size) {
    ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_host_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                        ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                        ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                        size_t                ಸ್ಥಾನಾಂತರ,
                                                        size_t                size) {
    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);
    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_hexagon_host_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_hexagon_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_hexagon_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

// ** backend buffer type

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_hexagon_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t ಬಫರ್_type) {
    ಹಿಂತಿರುಗಿ static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(ಬಫರ್_type->context)->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_hexagon_ಬಫರ್_type_alloc_ಬಫರ್(
            ggml_backend_ಬಫರ್_type_t ಬಫರ್_type, size_t size) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(ಬಫರ್_type->context)->dev_ctx;
    ಸ್ವಯಂಚಾಲಿತ sess    = dev_ctx->session();
    ಆದರೆ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಪ್ರಯತ್ನಿಸು {
        ggml_hexagon_shared_ಬಫರ್ * sbuf = ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, size, ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(ಬಫರ್_type, ggml_backend_hexagon_ಬಫರ್_interface, sbuf, size);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_hexagon_host_ಬಫರ್_type_alloc_ಬಫರ್(
            ggml_backend_ಬಫರ್_type_t ಬಫರ್_type, size_t size) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(ಬಫರ್_type->context)->dev_ctx;
    ಸ್ವಯಂಚಾಲಿತ sess    = dev_ctx->session();
    ಆದರೆ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಪ್ರಯತ್ನಿಸು {
        ggml_hexagon_shared_ಬಫರ್ * sbuf = ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, size, ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(ಬಫರ್_type, ggml_backend_hexagon_host_ಬಫರ್_interface, sbuf, size);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 128;  // HVX alignment
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (!ggml_backend_buft_is_host(buft) && htp_is_tiled_ತೂಕ_tensor(t)) {
        int64_t ne0 = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[0], 32);
        int64_t ne1 = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_ಗಾತ್ರ = htp_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0) * ne1 * ne2 * ne3;
        ಹಿಂತಿರುಗಿ std::max(tiled_ಗಾತ್ರ, ggml_nbytes(t));
    }
    ಹಿಂತಿರುಗಿ ggml_nbytes(t);
}

ಸ್ಥಿರ size_t parse_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str, size_t default_unit = 1024 * 1024) {
    ಆದರೆ (!str || str[0] == '\0') {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಅಕ್ಷರ * end = ಖಾಲಿ;
    ದ್ವಿಗುಣ val = strtod(str, &end);
    ಆದರೆ (val < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ (end && *end) {
        ಆಗ (*end == ' ') end++;
        ಆದರೆ (*end == 'k' || *end == 'K') {
            ಹಿಂತಿರುಗಿ (size_t) (val * 1024);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (*end == 'm' || *end == 'M') {
            ಹಿಂತಿರುಗಿ (size_t) (val * 1024 * 1024);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (*end == 'g' || *end == 'G') {
            ಹಿಂತಿರುಗಿ (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    ಹಿಂತಿರುಗಿ (size_t) (val * default_unit);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ opt_mbuf_dyn;
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_host_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ನಿಜ;
    GGML_ಬಳಸದ(buft);
}

ರಚನೆ ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_ಅಂಶ {
    size_t size;
    ಪೂರ್ಣಾಂಕ    ಮೊದಲ;
    ಪೂರ್ಣಾಂಕ    ಕೊನೆಯ;
};

ಬಳಸಿ ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_t = std::vector<ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_ಅಂಶ>;

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_hexagon_kv_layer_suffix(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(t->name, "cache_", 6) != 0) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * p = ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(t->name, "_l");
    ಆದರೆ (!p || !ಅಂಕಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)p[2])) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ p;
}

ರಚನೆ ggml_backend_hexagon_alloc_unit {
    size_t size;
    ಪೂರ್ಣಾಂಕ    ಮೊದಲ;
    ಪೂರ್ಣಾಂಕ    ಕೊನೆಯ;
};

ಸ್ಥಿರ ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_t ggml_backend_hexagon_alloc_ಬಫರ್_n_plan(
        ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_t plan;

    ಸ್ಥಿರಾಂಕ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ಸ್ಥಿರಾಂಕ size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    ಪೂರ್ಣಾಂಕ i = 0;
    ಆಗ (i < n_tensors) {
        ರಚನೆ ggml_tensor * t = tensors[i];
        size_t unit_ಗಾತ್ರ = 0;
        ಪೂರ್ಣಾಂಕ unit_ಮೊದಲ = i;
        ಪೂರ್ಣಾಂಕ unit_ಕೊನೆಯ = i + 1;

        ಆದರೆ (t->ದತ್ತಾಂಶ == ಖಾಲಿ && t->view_src == ಖಾಲಿ) {
            unit_ಗಾತ್ರ += GGML_PAD(ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, t), alignment);
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        ಆಗ (unit_ಕೊನೆಯ < n_tensors) {
            ರಚನೆ ggml_tensor * ಮುಂದಿನ = tensors[unit_ಕೊನೆಯ];

            ಆದರೆ (ಮುಂದಿನ->view_src != ಖಾಲಿ) {
                unit_ಕೊನೆಯ++;
                ಮುಂದುವರಿ;
            }

            ಆದರೆ (layer_suffix != ಖಾಲಿ) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮುಂದಿನ_suffix = ggml_hexagon_kv_layer_suffix(ಮುಂದಿನ);
                ಆದರೆ (ಮುಂದಿನ_suffix != ಖಾಲಿ && strcmp(layer_suffix, ಮುಂದಿನ_suffix) == 0) {
                    ಆದರೆ (ಮುಂದಿನ->ದತ್ತಾಂಶ == ಖಾಲಿ) {
                        unit_ಗಾತ್ರ += GGML_PAD(ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, ಮುಂದಿನ), alignment);
                    }
                    unit_ಕೊನೆಯ++;
                    ಮುಂದುವರಿ;
                }
            }

            ನಿಲ್ಲಿಸು;
        }

        units.push_back({ unit_ಗಾತ್ರ, unit_ಮೊದಲ, unit_ಕೊನೆಯ });
        i = unit_ಕೊನೆಯ;
    }

    size_t cur_buf_ಗಾತ್ರ  = 0;
    ಪೂರ್ಣಾಂಕ    cur_buf_ಮೊದಲ = 0;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & unit : units) {
        ಆದರೆ (unit.size == 0) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (cur_buf_ಗಾತ್ರ > 0 && (cur_buf_ಗಾತ್ರ + unit.size) > max_size) {
            plan.push_back({ cur_buf_ಗಾತ್ರ, cur_buf_ಮೊದಲ, unit.ಮೊದಲ });
            cur_buf_ಗಾತ್ರ  = 0;
            cur_buf_ಮೊದಲ = unit.ಮೊದಲ;
        }

        cur_buf_ಗಾತ್ರ += unit.size;
    }

    ಆದರೆ (cur_buf_ಗಾತ್ರ > 0) {
        plan.push_back({ cur_buf_ಗಾತ್ರ, cur_buf_ಮೊದಲ, n_tensors });
    }

    ಹಿಂತಿರುಗಿ plan;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_hexagon_ಬಫರ್_type_alloc_ಬಫರ್_n(
        ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ಸ್ಥಿರಾಂಕ ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_t plan = ggml_backend_hexagon_alloc_ಬಫರ್_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_ಬಫರ್_t> buffers;
    buffers.reserve(plan.size());

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಅಂಶ : plan) {
        ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_buft_alloc_ಬಫರ್(buft, ಅಂಶ.size);
        ಆದರೆ (ಬಫರ್ == ಖಾಲಿ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), ಅಂಶ.size);
            ಗಾಗಿ (ggml_backend_ಬಫರ್_t b : buffers) {
                ggml_backend_ಬಫರ್_free(b);
            }
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }

        ರಚನೆ ggml_tallocr tallocr = ggml_tallocr_new(ಬಫರ್);

        ರಚನೆ ggml_tensor * t_failed = ಖಾಲಿ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಅಂಶ.ಮೊದಲ; j < ಅಂಶ.ಕೊನೆಯ; j++) {
            ರಚನೆ ggml_tensor * t = tensors[j];
            ಆದರೆ (t->ದತ್ತಾಂಶ == ಖಾಲಿ) {
                ಆದರೆ (t->view_src == ಖಾಲಿ) {
                    ಆದರೆ (ggml_tallocr_alloc(&tallocr, t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (t->ಬಫರ್ == ಖಾಲಿ) {
                    ಆದರೆ (ggml_backend_view_init(t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (t->view_src != ಖಾಲಿ && t->ಬಫರ್ == ಖಾಲಿ) {
                    ಆದರೆ (ggml_backend_view_init(t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
        ಆದರೆ (t_failed != ಖಾಲಿ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ಗಾಗಿ (ggml_backend_ಬಫರ್_t b : buffers) {
                ggml_backend_ಬಫರ್_free(b);
            }
            ggml_backend_ಬಫರ್_free(ಬಫರ್);
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }

        buffers.push_back(ಬಫರ್);
    }

    ಆದರೆ (buffers.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (buffers.size() == 1) {
        ಹಿಂತಿರುಗಿ buffers[0];
    }

    ಹಿಂತಿರುಗಿ ggml_backend_multi_ಬಫರ್_alloc_ಬಫರ್(buffers.ದತ್ತಾಂಶ(), buffers.size());
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ_n(
        ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ಸ್ಥಿರಾಂಕ ggml_backend_hexagon_alloc_ಬಫರ್_n_plan_t plan = ggml_backend_hexagon_alloc_ಬಫರ್_n_plan(buft, tensors, n_tensors);

    size_t ಒಟ್ಟು = 0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಅಂಶ : plan) {
        ಒಟ್ಟು += ಅಂಶ.size;
    }
    ಹಿಂತಿರುಗಿ ಒಟ್ಟು;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_i ggml_backend_hexagon_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_ಬಫರ್_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_ಬಫರ್_type_alloc_ಬಫರ್_n,
    /* .get_alignment       = */ ggml_backend_hexagon_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ_n,
    /* .is_host             = */ ggml_backend_hexagon_ಬಫರ್_type_is_host,
};

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_i ggml_backend_hexagon_host_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_ಬಫರ್_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_ಬಫರ್_type_alloc_ಬಫರ್_n,
    /* .get_alignment       = */ ggml_backend_hexagon_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_ಬಫರ್_type_get_alloc_ಗಾತ್ರ_n,
    /* .is_host             = */ ggml_backend_hexagon_host_ಬಫರ್_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(ಪೂರ್ಣಾಂಕ dev_id, ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    ಬಫರ್_type.device  = dev;
    ಬಫರ್_type.iface   = ggml_backend_hexagon_ಬಫರ್_type_interface;
    ಬಫರ್_type.context = ಹೊಸ ggml_backend_hexagon_ಬಫರ್_type_context(config.name, ಇದು);

    host_ಬಫರ್_type.device  = dev;
    host_ಬಫರ್_type.iface   = ggml_backend_hexagon_host_ಬಫರ್_type_interface;
    host_ಬಫರ್_type.context = ಹೊಸ ggml_backend_hexagon_ಬಫರ್_type_context(config.name + "-HOST", ಇದು);

    fence_ಬಫರ್_type.device  = dev;
    fence_ಬಫರ್_type.iface   = ggml_backend_hexagon_ಬಫರ್_type_interface;
    fence_ಬಫರ್_type.context = ಹೊಸ ggml_backend_hexagon_ಬಫರ್_type_context(config.name + "-FENCE", ಇದು);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    ಅಳಿಸು static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(ಬಫರ್_type.context);
    ಅಳಿಸು static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(host_ಬಫರ್_type.context);
    ಅಳಿಸು static_cast<ggml_backend_hexagon_ಬಫರ್_type_context *>(fence_ಬಫರ್_type.context);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_hexagon(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್ * b) {
    ಹಿಂತಿರುಗಿ b->buft->iface.get_alignment == ggml_backend_hexagon_ಬಫರ್_type_get_alignment;
}

ರಚನೆ ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ>                b_map; // buffer fd   to index
    std::unordered_map<ಸ್ಥಿರಾಂಕ ggml_tensor*, ಪೂರ್ಣಾಂಕ> t_map; // tensor ptr  to index
    std::unordered_multimap<ಶೂನ್ಯ*, ಪೂರ್ಣಾಂಕ>         d_map; // tensor data to index

    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_bufs;     // num buffers in the batch
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_tens;     // num tensors ...
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_bufs_max;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_tens_max;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_ops_max;
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

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_ಗಾತ್ರ, size_t max_vmem) {
        ಇದು->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_ಗಾತ್ರ;
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

    ತಾರ್ಕಿಕ ಖಾಲಿಗಳು() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ n_ops == 0; }

    // add buffer and return its index
    ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_ಬಫರ್(ggml_hexagon_shared_ಬಫರ್ * sbuf) {
        // Lookup by fd
        ಸ್ವಯಂಚಾಲಿತ it = b_map.find(sbuf->fd());
        ಆದರೆ (it != b_map.end()) { ಹಿಂತಿರುಗಿ it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        ಪೂರ್ಣಾಂಕ bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.ತಳ  = (uint64_t) sbuf->ತಳ();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        ಆದರೆ (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (ಶೂನ್ಯ*) sbuf->ತಳ(), (size_t) b.size, b_vmem);

        ಹಿಂತಿರುಗಿ bi;
    }

    ತಾರ್ಕಿಕ same_shape(ಸ್ಥಿರಾಂಕ htp_tensor * h, ಸ್ಥಿರಾಂಕ ggml_tensor * t) ಸ್ಥಿರಾಂಕ {
        ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *) t->extra;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ಸ್ಥಿರಾಂಕ int64_t ne0_padded = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[0], 32);
        ಸ್ಥಿರಾಂಕ int64_t ne1_padded = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? ಗಾತ್ರ(ggml_fp16_t) : t->nb[0];
        ಎಣಿಕೆ ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        ಹಿಂತಿರುಗಿ (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಸ್ವಯಂಚಾಲಿತ extra = (ggml_hexagon_tensor_extra *) t->extra;
        ಸ್ವಯಂಚಾಲಿತ sbuf  = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(t->ಬಫರ್->context);

        // First lookup by tensor data
        ಸ್ವಯಂಚಾಲಿತ range = d_map.equal_range(t->ದತ್ತಾಂಶ);
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ it = range.ಮೊದಲ; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            ಆದರೆ (same_shape(h, t)) { ಹಿಂತಿರುಗಿ it->second; }
        }

        // Lookup by tensor ptr
        ಸ್ವಯಂಚಾಲಿತ it = t_map.find(t);
        ಆದರೆ (it != t_map.end()) { ಹಿಂತಿರುಗಿ it->second; }

        // Add new tensor to the batch
        ಪೂರ್ಣಾಂಕ ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->ದತ್ತಾಂಶ, ti});

        uint64_t t_ಸ್ಥಾನಾಂತರ = (uint8_t *) t->ದತ್ತಾಂಶ - sbuf->ತಳ();
        size_t   t_ಗಾತ್ರ   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = ಕೂಡಿಸು_ಬಫರ್(sbuf);
        h.ti    = ti;
        h.ದತ್ತಾಂಶ  = t_ಸ್ಥಾನಾಂತರ;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ಆದರೆ (is_tiled) {
            ಆದರೆ (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = ಗಾತ್ರ(ggml_fp16_t);
            } ಇಲ್ಲದಿದ್ದರೆ {
                h.nb[0] = t->nb[0];
            }
            ಸ್ಥಿರಾಂಕ int64_t ne0_padded = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[0], 32);
            ಸ್ಥಿರಾಂಕ int64_t ne1_padded = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[1], 32);
            h.nb[1] = htp_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_ಗಾತ್ರ  = h.size;
        } ಇಲ್ಲದಿದ್ದರೆ {
            h.size  = t_ಗಾತ್ರ;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_ತೂಕ) != 0) {
            h.flags |= HTP_TENSOR_ತೂಕ;
        }
        ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        ಆದರೆ ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (ಶೂನ್ಯ*) t->ದತ್ತಾಂಶ, (size_t) t_ಸ್ಥಾನಾಂತರ, t_ಗಾತ್ರ, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        ಹಿಂತಿರುಗಿ ti;
    }

    ತಾರ್ಕಿಕ fit_op(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (n_ops >= n_ops_max) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        ಪೂರ್ಣಾಂಕ ನೋಡಿದ_bufs[HTP_OP_MAX_BUFS];
        ಪೂರ್ಣಾಂಕ n_ನೋಡಿದ_bufs = 0;

        ಸ್ವಯಂಚಾಲಿತ fit_tensor = [&](ಸ್ಥಿರಾಂಕ ggml_tensor *t) {
            ಆದರೆ (!t) ಹಿಂತಿರುಗಿ;
            ಆದರೆ (!t_map.count(t)) {
                extra_tens++;

                ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(t->ಬಫರ್->context);
                ಪೂರ್ಣಾಂಕ fd = sbuf->fd();
                ಆದರೆ (!b_map.count(fd)) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ನೋಡಿದ_bufs; i++) {
                        ಆದರೆ (ನೋಡಿದ_bufs[i] == fd) ಹಿಂತಿರುಗಿ;
                    }
                    ಆದರೆ (n_ನೋಡಿದ_bufs < HTP_OP_MAX_BUFS) {
                        ನೋಡಿದ_bufs[n_ನೋಡಿದ_bufs++] = fd;
                    }
                    ಆದರೆ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src : ಗಂಟು.get_inputs()) {
            fit_tensor(src);
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * output : ಗಂಟು.get_outputs()) {
            fit_tensor(output);
        }

        ಆದರೆ ((extra_bufs + n_bufs) > n_bufs_max) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ ((extra_tens + n_tens) > n_tens_max) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ ((extra_vmem + b_vmem) > b_vmem_max) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // assumes that fit_op() was called first and returned true
    ಶೂನ್ಯ ಕೂಡಿಸು_op(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        // Add new op

        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = ಗಂಟು;

        htp_op_desc &o = h_ops[n];
        ಸ್ಮರಣೆ_ನಕಲು(o.params,        ಗಂಟು.ಗಂಟು->op_params, ಗಾತ್ರ(ಗಂಟು.ಗಂಟು->op_params));
        ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, ಗಂಟು.kernel_params,   ಗಾತ್ರ(o.kernel_params));
        o.opcode = ಗಂಟು.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? ಕೂಡಿಸು_tensor(inputs[i]) : 0xffff;
        }

        ಸ್ವಯಂಚಾಲಿತ outputs = ಗಂಟು.get_outputs();
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? ಕೂಡಿಸು_tensor(outputs[i]) : 0xffff;
        }
    }

    ಶೂನ್ಯ ವಿಂಗಡಿಸು_buffers() {
        ಆದರೆ (n_bufs <= 1) ಹಿಂತಿರುಗಿ;

        std::vector<ಪೂರ್ಣಾಂಕ> ಕ್ರಮ(n_bufs);
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_bufs; i++) { ಕ್ರಮ[i] = (ಪೂರ್ಣಾಂಕ) i; }

        std::stable_sort(ಕ್ರಮ.begin(), ಕ್ರಮ.end(), [&](ಪೂರ್ಣಾಂಕ a, ಪೂರ್ಣಾಂಕ b) {
            ಹಿಂತಿರುಗಿ h_bufs[a].size > h_bufs[b].size;
        });

        ತಾರ್ಕಿಕ already_sorted = ನಿಜ;
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_bufs; i++) {
            ಆದರೆ (ಕ್ರಮ[i] != (ಪೂರ್ಣಾಂಕ) i) {
                already_sorted = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (already_sorted) ಹಿಂತಿರುಗಿ;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ new_bi = 0; new_bi < n_bufs; new_bi++) {
            ಪೂರ್ಣಾಂಕ old_bi = ಕ್ರಮ[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    ಶೂನ್ಯ update_mdev_ಗುಂಪು(uint32_t mdev_idx) {
        ಆದರೆ (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_ಗುಂಪು) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    ತಾರ್ಕಿಕ try_fuse_common(std::initializer_list<ಸ್ಥಿರಾಂಕ ggml_tensor *> tensors) ಸ್ಥಿರಾಂಕ {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        ಪೂರ್ಣಾಂಕ ನೋಡಿದ_bufs[HTP_OP_MAX_BUFS];
        ಪೂರ್ಣಾಂಕ n_ನೋಡಿದ_bufs = 0;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * t : tensors) {
            ಆದರೆ (!t || t_map.count(t)) {
                ಮುಂದುವರಿ;
            }
            extra_tens++;
            ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(t->ಬಫರ್->context);
            ಪೂರ್ಣಾಂಕ fd = sbuf->fd();
            ಆದರೆ (!b_map.count(fd)) {
                ತಾರ್ಕಿಕ ಸಿಕ್ಕಿತು = ಸುಳ್ಳು;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ನೋಡಿದ_bufs; i++) {
                    ಆದರೆ (ನೋಡಿದ_bufs[i] == fd) {
                        ಸಿಕ್ಕಿತು = ನಿಜ;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (!ಸಿಕ್ಕಿತು) {
                    ಆದರೆ (n_ನೋಡಿದ_bufs < HTP_OP_MAX_BUFS) {
                        ನೋಡಿದ_bufs[n_ನೋಡಿದ_bufs++] = fd;
                    }
                    ಆದರೆ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        ಆದರೆ ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_fuse_common(ಸ್ಥಿರಾಂಕ ggml_tensor * t1, ಸ್ಥಿರಾಂಕ ggml_tensor * t2) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ try_fuse_common({t1, t2});
    }

    ತಾರ್ಕಿಕ try_fuse_allreduce_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0 || opt_ar_select != 2) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (ಗಂಟು.opcode != HTP_OP_ಕೂಡಿಸು) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode != HTP_OP_ALLREDUCE) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ವಯಂಚಾಲಿತ * ar_kparams = (ರಚನೆ htp_allreduce_kernel_params *) ಕೊನೆಯ_ಗಂಟು.kernel_params;
        ಸ್ಥಿರಾಂಕ uint32_t rank    = (uint32_t) ar_kparams->rank;
        ಸ್ಥಿರಾಂಕ uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ಸ್ಥಿರಾಂಕ ggml_tensor * ar_local = ಕೊನೆಯ_ಗಂಟು.inputs[rank];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_src0 = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_src1 = ಗಂಟು.src1();
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_dst  = ಗಂಟು.dst();

        ಆದರೆ (!ggml_hexagon_tensor_is_fuseable(ar_local)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * res_tensor;
        ಆದರೆ (ಕೂಡಿಸು_src0 == ar_local || ಕೂಡಿಸು_src0->ದತ್ತಾಂಶ == ar_local->ದತ್ತಾಂಶ) {
            res_tensor = ಕೂಡಿಸು_src1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೂಡಿಸು_src1 == ar_local || ಕೂಡಿಸು_src1->ದತ್ತಾಂಶ == ar_local->ದತ್ತಾಂಶ) {
            res_tensor = ಕೂಡಿಸು_src0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (ar_local->type != res_tensor->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಅಡ್ಡಸಾಲು_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        ಆದರೆ (!is_same_shape && !is_ಅಡ್ಡಸಾಲು_bcast) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಆದರೆ (is_same_shape) {
            ಆದರೆ (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಆದರೆ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(ಕೂಡಿಸು_dst)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // scatter is only valid for in-place add within max outputs
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && ಕೂಡಿಸು_dst->ದತ್ತಾಂಶ == ar_local->ದತ್ತಾಂಶ;

        ರಚನೆ htp_allreduce_kernel_params new_kparams;
        ಆದರೆ (!ggml_hexagon_precompute_allreduce_params(
            sess, ಕೂಡಿಸು_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, ನಿಜ, is_ಅಡ್ಡಸಾಲು_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ತುಂಬಿದ;

        ಗಾಗಿ (uint32_t r = 0; r < n_ranks; r++) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * ar_src = ಕೊನೆಯ_ಗಂಟು.inputs[r];
            ಆದರೆ (!ggml_hexagon_tensors_overlap(ಕೂಡಿಸು_dst, ar_src)) {
                ಮುಂದುವರಿ;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            ಆದರೆ (scatter_ok && r == rank) {
                ಮುಂದುವರಿ;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!try_fuse_common(res_tensor, ಕೂಡಿಸು_dst)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಕೊನೆಯ_ಗಂಟು.opcode = HTP_OP_ALLREDUCE_ಕೂಡಿಸು;
        ಕೊನೆಯ_ಗಂಟು.name   = "ALLREDUCE+ADD";
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(res_tensor);
        ಕೊನೆಯ_ಗಂಟು.outputs.clear();
        ಕೊನೆಯ_ಗಂಟು.outputs.push_back(ಕೂಡಿಸು_dst);
        ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &new_kparams, ಗಾತ್ರ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ಕೂಡಿಸು;
        ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &new_kparams, ಗಾತ್ರ(new_kparams));

        o.src[2 * n_ranks] = ಕೂಡಿಸು_tensor(res_tensor);
        ಆದರೆ (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            ಗಾಗಿ (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            ಗಾಗಿ (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            o.dst[0] = ಕೂಡಿಸು_tensor(ಕೂಡಿಸು_dst);
            ಗಾಗಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (ಪೂರ್ಣಾಂಕ) new_kparams.mode, (ಪೂರ್ಣಾಂಕ) new_kparams.n_dsts);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_fuse_rms_norm_mul(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (ಗಂಟು.opcode != HTP_OP_MUL) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode != HTP_OP_RMS_NORM) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * mul_src0 = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * mul_src1 = ಗಂಟು.src1();
        ಸ್ಥಿರಾಂಕ ggml_tensor * rms_ಹೊರಹೋಗು  = ಕೊನೆಯ_ಗಂಟು.dst();

        ಆದರೆ (!ggml_hexagon_tensor_is_fuseable(rms_ಹೊರಹೋಗು)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * ತೂಕ;
        ಆದರೆ (mul_src0 == rms_ಹೊರಹೋಗು || mul_src0->ದತ್ತಾಂಶ == rms_ಹೊರಹೋಗು->ದತ್ತಾಂಶ) {
            ತೂಕ = mul_src1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mul_src1 == rms_ಹೊರಹೋಗು || mul_src1->ದತ್ತಾಂಶ == rms_ಹೊರಹೋಗು->ದತ್ತಾಂಶ) {
            ತೂಕ = mul_src0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = ಕೊನೆಯ_ಗಂಟು.src0();

        ಆದರೆ (src0->ne[0] != ತೂಕ->ne[0] || src0->ne[0] != ಗಂಟು.dst()->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಅಡ್ಡಸಾಲು_bcast = (ತೂಕ->ne[1] == 1 && ತೂಕ->ne[2] == 1 && ತೂಕ->ne[3] == 1);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_same_shape = (src0->ne[0] == ತೂಕ->ne[0] && src0->ne[1] == ತೂಕ->ne[1] &&
                                    src0->ne[2] == ತೂಕ->ne[2] && src0->ne[3] == ತೂಕ->ne[3]);
        ಆದರೆ (!is_ಅಡ್ಡಸಾಲು_bcast && !is_same_shape) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಆದರೆ (!ggml_are_same_shape(src0, ಗಂಟು.dst())) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ggml_is_contiguous(src0) != ggml_is_contiguous(ಗಂಟು.dst())) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ರಚನೆ htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, ತೂಕ, ಗಂಟು.dst(), &new_kparams
        );

        ಆದರೆ ((size_t) new_kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!try_fuse_common(ತೂಕ, ಗಂಟು.dst())) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಕೊನೆಯ_ಗಂಟು.opcode = HTP_OP_RMS_NORM_MUL;
        ಕೊನೆಯ_ಗಂಟು.name   = "RMS_NORM+MUL";
        ಕೊನೆಯ_ಗಂಟು.inputs.clear();
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(src0);
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(ತೂಕ);
        ಕೊನೆಯ_ಗಂಟು.outputs.clear();
        ಕೊನೆಯ_ಗಂಟು.outputs.push_back(ಗಂಟು.dst());
        ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &new_kparams, ಗಾತ್ರ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &new_kparams, ಗಾತ್ರ(new_kparams));

        o.src[0] = ಕೂಡಿಸು_tensor(src0);
        o.src[1] = ಕೂಡಿಸು_tensor(ತೂಕ);
        ಗಾಗಿ (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = ಕೂಡಿಸು_tensor(ಗಂಟು.dst());
        ಗಾಗಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (ಗಂಟು.opcode != HTP_OP_ಕೂಡಿಸು) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode != HTP_OP_MUL_MAT) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_src0 = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_src1 = ಗಂಟು.src1();
        ಸ್ಥಿರಾಂಕ ggml_tensor * mm_ಹೊರಹೋಗು   = ಕೊನೆಯ_ಗಂಟು.dst();

        ಆದರೆ (!ggml_hexagon_tensor_is_fuseable(mm_ಹೊರಹೋಗು)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * src2;
        ಆದರೆ (ಕೂಡಿಸು_src0 == mm_ಹೊರಹೋಗು || ಕೂಡಿಸು_src0->ದತ್ತಾಂಶ == mm_ಹೊರಹೋಗು->ದತ್ತಾಂಶ) {
            src2 = ಕೂಡಿಸು_src1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೂಡಿಸು_src1 == mm_ಹೊರಹೋಗು || ಕೂಡಿಸು_src1->ದತ್ತಾಂಶ == mm_ಹೊರಹೋಗು->ದತ್ತಾಂಶ) {
            src2 = ಕೂಡಿಸು_src0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = ಕೊನೆಯ_ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = ಕೊನೆಯ_ಗಂಟು.src1();

        ಆದರೆ (src2->type != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        ಆದರೆ (!ggml_are_same_shape(mm_ಹೊರಹೋಗು, ಗಂಟು.dst())) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * orig_kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) ಕೊನೆಯ_ಗಂಟು.kernel_params;
        ರಚನೆ htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_ಕೂಡಿಸು_params(sess, src0, src1, src2, ಗಂಟು.dst(), &kparams);
        ಆದರೆ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (orig_kparams->collapse != kparams.collapse) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        ಆದರೆ (!can_fuse) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            ಆದರೆ (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_ಎಳೆಗಳು < orig_kparams->n_act_ಎಳೆಗಳು) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_ಎಳೆಗಳು, kparams.n_act_ಎಳೆಗಳು);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಆದರೆ (!try_fuse_common(src2, ಗಂಟು.dst())) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಕೊನೆಯ_ಗಂಟು.opcode = HTP_OP_MUL_MAT_ಕೂಡಿಸು;
        ಕೊನೆಯ_ಗಂಟು.name   = "MUL_MAT+ADD";
        ಕೊನೆಯ_ಗಂಟು.inputs.clear();
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(src0);
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(src1);
        ಕೊನೆಯ_ಗಂಟು.inputs.push_back(src2);
        ಕೊನೆಯ_ಗಂಟು.outputs.clear();
        ಕೊನೆಯ_ಗಂಟು.outputs.push_back(ಗಂಟು.dst());
        ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &kparams, ಗಾತ್ರ(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ಕೂಡಿಸು;
        ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &kparams, ಗಾತ್ರ(kparams));

        o.src[0] = ಕೂಡಿಸು_tensor(src0);
        o.src[1] = ಕೂಡಿಸು_tensor(src1);
        o.src[2] = ಕೂಡಿಸು_tensor(src2);
        ಗಾಗಿ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = ಕೂಡಿಸು_tensor(ಗಂಟು.dst());
        ಗಾಗಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_nx(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_MUL_MAT) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (!is_mergeable_mul_mat(ಗಂಟು.ಗಂಟು)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * w_ಒಳ = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * x_ಒಳ = ಗಂಟು.src1();
        ಸ್ಥಿರಾಂಕ ggml_tensor * d_ಒಳ = ಗಂಟು.dst();

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode == HTP_OP_MUL_MAT_NX) {
            ಸ್ಥಿರಾಂಕ uint32_t curr_n = (uint32_t) ಕೊನೆಯ_ಗಂಟು.outputs.size();
            ಆದರೆ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor * w0 = ಕೊನೆಯ_ಗಂಟು.inputs[0];
            ಸ್ಥಿರಾಂಕ ggml_tensor * x  = ಕೊನೆಯ_ಗಂಟು.inputs[curr_n];

            ಆದರೆ (x_ಒಳ != x || w_ಒಳ->type != w0->type || w_ಒಳ->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_ಒಳ) != ggml_hexagon_tensor_is_tiled(w0)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (!ಕೊನೆಯ_ಗಂಟು.fused.ಖಾಲಿಗಳು() && (mm_is_hmx_eligible(ಕೊನೆಯ_ಗಂಟು.fused[0]) != mm_is_hmx_eligible(ಗಂಟು.ಗಂಟು))) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * orig_kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) ಕೊನೆಯ_ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_ಒಳ, x, d_ಒಳ);
            ಆದರೆ (orig_kparams->collapse && !collapse) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ರಚನೆ htp_mm_kernel_params kparams;
            ಆದರೆ (collapse) {
                ಸ್ಥಿರಾಂಕ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            ಆದರೆ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (!try_fuse_common(w_ಒಳ, d_ಒಳ)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಕೊನೆಯ_ಗಂಟು.inputs[curr_n] = w_ಒಳ;
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(x);
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(d_ಒಳ);
            ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            ಗಾಗಿ (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = ಕೂಡಿಸು_tensor(ಕೊನೆಯ_ಗಂಟು.inputs[s]);
            }
            ಗಾಗಿ (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ಗಾಗಿ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = ಕೂಡಿಸು_tensor(ಕೊನೆಯ_ಗಂಟು.outputs[d]);
            }
            ಗಾಗಿ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        // Case 2: last_node is single MUL_MAT
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode == HTP_OP_MUL_MAT) {
            ಆದರೆ (!is_mergeable_mul_mat_pair(ಕೊನೆಯ_ಗಂಟು.ಗಂಟು, ಗಂಟು.ಗಂಟು)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor * w0 = ಕೊನೆಯ_ಗಂಟು.src0();
            ಸ್ಥಿರಾಂಕ ggml_tensor * x  = ಕೊನೆಯ_ಗಂಟು.src1();
            ಸ್ಥಿರಾಂಕ ggml_tensor * w1 = ಗಂಟು.src0();
            ಸ್ಥಿರಾಂಕ ggml_tensor * dst_0 = ಕೊನೆಯ_ಗಂಟು.dst();
            ಸ್ಥಿರಾಂಕ ggml_tensor * dst_1 = ಗಂಟು.dst();

            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * orig_kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) ಕೊನೆಯ_ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            ಆದರೆ (orig_kparams->collapse && !collapse) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ರಚನೆ htp_mm_kernel_params kparams;
            ಆದರೆ (collapse) {
                ಸ್ಥಿರಾಂಕ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            ಆದರೆ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (!try_fuse_common(w1, ಗಂಟು.dst())) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಕೊನೆಯ_ಗಂಟು.opcode = HTP_OP_MUL_MAT_NX;
            ಕೊನೆಯ_ಗಂಟು.name   = "MUL_MAT_NX";
            ಕೊನೆಯ_ಗಂಟು.inputs.clear();
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(w0);
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(w1);
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(x);
            ಕೊನೆಯ_ಗಂಟು.outputs.clear();
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(dst_0);
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(dst_1);
            ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            o.src[0] = ಕೂಡಿಸು_tensor(w0);
            o.src[1] = ಕೂಡಿಸು_tensor(w1);
            o.src[2] = ಕೂಡಿಸು_tensor(x);
            ಗಾಗಿ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = ಕೂಡಿಸು_tensor(dst_0);
            o.dst[1] = ಕೂಡಿಸು_tensor(dst_1);
            ಗಾಗಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_id_nx(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_MUL_MAT_ID) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (!is_mergeable_mul_mat_id(ಗಂಟು.ಗಂಟು)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * w_ಒಳ   = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * x_ಒಳ   = ಗಂಟು.src1();
        ಸ್ಥಿರಾಂಕ ggml_tensor * ids_ಒಳ = ಗಂಟು.ಗಂಟು->src[2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * d_ಒಳ   = ಗಂಟು.dst();

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ಸ್ಥಿರಾಂಕ uint32_t curr_n = (uint32_t) ಕೊನೆಯ_ಗಂಟು.outputs.size();
            ಆದರೆ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor * w0  = ಕೊನೆಯ_ಗಂಟು.inputs[0];
            ಸ್ಥಿರಾಂಕ ggml_tensor * x   = ಕೊನೆಯ_ಗಂಟು.inputs[curr_n];
            ಸ್ಥಿರಾಂಕ ggml_tensor * ids = ಕೊನೆಯ_ಗಂಟು.inputs[curr_n + 1];

            ಆದರೆ (x_ಒಳ != x || ids_ಒಳ != ids || w_ಒಳ->type != w0->type || w_ಒಳ->ne[0] != w0->ne[0] || w_ಒಳ->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_ಒಳ) != ggml_hexagon_tensor_is_tiled(w0)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (!ಕೊನೆಯ_ಗಂಟು.fused.ಖಾಲಿಗಳು() && (mm_is_hmx_eligible(ಕೊನೆಯ_ಗಂಟು.fused[0]) != mm_is_hmx_eligible(ಗಂಟು.ಗಂಟು))) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ರಚನೆ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_ಒಳ, curr_n + 1, &kparams);
            ಆದರೆ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (!try_fuse_common(w_ಒಳ, d_ಒಳ)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಕೊನೆಯ_ಗಂಟು.inputs[curr_n] = w_ಒಳ;
            ಕೊನೆಯ_ಗಂಟು.inputs[curr_n + 1] = x;
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(ids);
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(d_ಒಳ);
            ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            ಗಾಗಿ (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = ಕೂಡಿಸು_tensor(ಕೊನೆಯ_ಗಂಟು.inputs[s]);
            }
            ಗಾಗಿ (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ಗಾಗಿ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = ಕೂಡಿಸು_tensor(ಕೊನೆಯ_ಗಂಟು.outputs[d]);
            }
            ಗಾಗಿ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        // Case 2: last_node is single MUL_MAT_ID
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID) {
            ಆದರೆ (!is_mergeable_mul_mat_id_pair(ಕೊನೆಯ_ಗಂಟು.ಗಂಟು, ಗಂಟು.ಗಂಟು)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor * w0  = ಕೊನೆಯ_ಗಂಟು.src0();
            ಸ್ಥಿರಾಂಕ ggml_tensor * x   = ಕೊನೆಯ_ಗಂಟು.src1();
            ಸ್ಥಿರಾಂಕ ggml_tensor * ids = ಕೊನೆಯ_ಗಂಟು.ಗಂಟು->src[2];
            ಸ್ಥಿರಾಂಕ ggml_tensor * w1  = ಗಂಟು.src0();

            ರಚನೆ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, ಗಂಟು.dst(), 2, &kparams);
            ಆದರೆ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (!try_fuse_common(w1, ಗಂಟು.dst())) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor * dst_0 = ಕೊನೆಯ_ಗಂಟು.dst();
            ಸ್ಥಿರಾಂಕ ggml_tensor * dst_1 = ಗಂಟು.dst();

            ಕೊನೆಯ_ಗಂಟು.opcode = HTP_OP_MUL_MAT_ID_NX;
            ಕೊನೆಯ_ಗಂಟು.name   = "MUL_MAT_ID_NX";
            ಕೊನೆಯ_ಗಂಟು.inputs.clear();
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(w0);
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(w1);
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(x);
            ಕೊನೆಯ_ಗಂಟು.inputs.push_back(ids);
            ಕೊನೆಯ_ಗಂಟು.outputs.clear();
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(dst_0);
            ಕೊನೆಯ_ಗಂಟು.outputs.push_back(dst_1);
            ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            ಸ್ಮರಣೆ_ನಕಲು(ಕೊನೆಯ_ಗಂಟು.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            ಸ್ಮರಣೆ_ನಕಲು(o.kernel_params, &kparams, ಗಾತ್ರ(kparams));

            o.src[0] = ಕೂಡಿಸು_tensor(w0);
            o.src[1] = ಕೂಡಿಸು_tensor(w1);
            o.src[2] = ಕೂಡಿಸು_tensor(x);
            o.src[3] = ಕೂಡಿಸು_tensor(ids);
            ಗಾಗಿ (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = ಕೂಡಿಸು_tensor(dst_0);
            o.dst[1] = ಕೂಡಿಸು_tensor(dst_1);
            ಗಾಗಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ try_fuse_gdn_cpy(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_CPY) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        htp_opnode & ಕೊನೆಯ_ಗಂಟು = ops[n_ops - 1];
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.opcode != HTP_OP_GATED_DELTA_NET) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು.outputs.size() != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * gdn_ಹೊರಹೋಗು  = ಕೊನೆಯ_ಗಂಟು.dst();
        ಸ್ಥಿರಾಂಕ ggml_tensor * cpy_ಗಂಟು = ಗಂಟು.ಗಂಟು;
        ಸ್ಥಿರಾಂಕ ggml_tensor * cpy_src  = ಗಂಟು.src0();
        ಸ್ಥಿರಾಂಕ ggml_tensor * cpy_dst  = ಗಂಟು.dst();

        ಆದರೆ (!cpy_src || !cpy_dst || !cpy_dst->ದತ್ತಾಂಶ) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (gdn_ಹೊರಹೋಗು->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ ((gdn_ಹೊರಹೋಗು->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ ggml_tensor * v = ಕೊನೆಯ_ಗಂಟು.ಗಂಟು->src[2];
        ಆದರೆ (!v) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಸ್ಥಿರಾಂಕ int64_t S_v      = v->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t H        = v->ne[1];
        ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = v->ne[2];
        ಸ್ಥಿರಾಂಕ int64_t n_seqs   = v->ne[3];
        ಸ್ಥಿರಾಂಕ int64_t K        = ggml_get_op_params_i32(ಕೊನೆಯ_ಗಂಟು.ಗಂಟು, 0);
        ಸ್ಥಿರಾಂಕ size_t  ಬಾಲ_off = (size_t) S_v * H * n_ಟೋಕನುಗಳು * n_seqs * ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಸ್ಥಿರಾಂಕ int64_t D         = S_v * S_v * H;
        ಸ್ಥಿರಾಂಕ int64_t n_written = std::min<int64_t>(n_ಟೋಕನುಗಳು, K);

        ಆದರೆ (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_ಹೊರಹೋಗು && cpy_src->view_src->ದತ್ತಾಂಶ != gdn_ಹೊರಹೋಗು->ದತ್ತಾಂಶ) ||
            cpy_src->view_offs != ಬಾಲ_off || !ggml_is_contiguous(cpy_src)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * ಗಾತ್ರ(ದಶಮಾಂಶ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (n_written > 1) {
            ಆದರೆ (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * ಗಾತ್ರ(ದಶಮಾಂಶ)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಆದರೆ (!try_fuse_common({cpy_dst})) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಕೊನೆಯ_ಗಂಟು.name += "+CPY";
        ಕೊನೆಯ_ಗಂಟು.outputs.push_back(cpy_dst);
        ಕೊನೆಯ_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = ಕೂಡಿಸು_tensor(cpy_dst);
        ಗಾಗಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_fuse(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (!opt_opfusion) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ಕೂಡಿಸು) && try_fuse_allreduce_ಕೂಡಿಸು(ಗಂಟು)) ಹಿಂತಿರುಗಿ ನಿಜ;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(ಗಂಟು))  ಹಿಂತಿರುಗಿ ನಿಜ;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ಕೂಡಿಸು)   && try_fuse_mul_mat_ಕೂಡಿಸು(ಗಂಟು))   ಹಿಂತಿರುಗಿ ನಿಜ;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(ಗಂಟು))    ಹಿಂತಿರುಗಿ ನಿಜ;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(ಗಂಟು)) ಹಿಂತಿರುಗಿ ನಿಜ;
        ಆದರೆ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(ಗಂಟು))       ಹಿಂತಿರುಗಿ ನಿಜ;
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
};

ರಚನೆ ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ರಚನೆ ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_ಬಫರ್ *shm_buf;
    size_t                      shm_blk_ಗಾತ್ರ;
    size_t                      ಆಳ;

    ಬಳಸಿ opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       ಆರಂಭ_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_ಗಾತ್ರ, size_t ಆಳ) : ಆಳ(ಆಳ) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_ಗಾತ್ರ;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_ಗಾತ್ರ = 0;
        ಆದರೆ (opt_profile == 3) {
            tr_ಗಾತ್ರ = (HTP_MAX_NTHREADS + 1) * opt_optrace * ಗಾತ್ರ(htp_trace_desc);
        }

        shm_blk_ಗಾತ್ರ = ಗಾತ್ರ(htp_buf_desc)  * n_bufs    +
                       ಗಾತ್ರ(htp_tensor)    * n_tensors +
                       ಗಾತ್ರ(htp_op_desc)   * n_ops     +
                       ಗಾತ್ರ(htp_prof_desc) * n_ops     +
                       tr_ಗಾತ್ರ;

        shm_buf = ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, shm_blk_ಗಾತ್ರ * ಆಳ, ನಿಜ /* pinned */);

        op_cache.resize(ಆಳ);
        ಆರಂಭ_usec.resize(ಆಳ, 0);

        ಆದರೆ (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_ಗಾತ್ರ, ಆಳ, shm_buf->size(), shm_blk_ಗಾತ್ರ);
        }
    }

    ~ggml_hexagon_opqueue() {
        ಅಳಿಸು shm_buf;
    }

    size_t shm_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ shm_buf ? shm_buf->size() : 0; }

    // push new batch
    ತಾರ್ಕಿಕ ತಳ್ಳು(htp_opbatch_req& req, dspqueue_ಬಫರ್& dbuf, ಸ್ಥಿರಾಂಕ ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(ಗಾತ್ರ(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(ಗಾತ್ರ(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(ಗಾತ್ರ(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(ಗಾತ್ರ(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(ಗಾತ್ರ(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(ಗಾತ್ರ(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        ಆದರೆ (seq - shm_buf->sess->batch_rsp_seq > ಆಳ) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

        ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನ = (uint32_t) ((seq - 1) % ಆಳ);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[ಸ್ಥಾನ]   = op_batch->ops;
        ಆರಂಭ_usec[ಸ್ಥಾನ] = ggml_ಸಮಯ_us();

        ಸ್ಥಿರಾಂಕ size_t b_ಗಾತ್ರ = ಗಾತ್ರ(htp_buf_desc)  * req.n_bufs;
        ಸ್ಥಿರಾಂಕ size_t t_ಗಾತ್ರ = ಗಾತ್ರ(htp_tensor)    * req.n_tensors;
        ಸ್ಥಿರಾಂಕ size_t o_ಗಾತ್ರ = ಗಾತ್ರ(htp_op_desc)   * req.n_ops;
        ಸ್ಥಿರಾಂಕ size_t p_ಗಾತ್ರ = ಗಾತ್ರ(htp_prof_desc) * req.n_ops;

        size_t tr_ಗಾತ್ರ = 0;
        ಆದರೆ (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_ಗಾತ್ರ = (HTP_MAX_NTHREADS + 1) * req.n_traces * ಗಾತ್ರ(htp_trace_desc);
        } ಇಲ್ಲದಿದ್ದರೆ {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->ತಳ() + ((size_t) ಸ್ಥಾನ * shm_blk_ಗಾತ್ರ);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_ಬಫರ್_FLAG_FLUSH_SENDER | DSPQUEUE_ಬಫರ್_FLAG_INVALIDATE_RECIPIENT;
        dbuf.ಸ್ಥಾನಾಂತರ   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->ತಳ();
        dbuf.size     = b_ಗಾತ್ರ + t_ಗಾತ್ರ + o_ಗಾತ್ರ + p_ಗಾತ್ರ + tr_ಗಾತ್ರ;

        GGML_ASSERT(dbuf.size <= shm_blk_ಗಾತ್ರ);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_ಗಾತ್ರ;
        uint8_t * t_ptr = m_ptr; m_ptr += t_ಗಾತ್ರ;
        uint8_t * o_ptr = m_ptr;

        ಸ್ಮರಣೆ_ನಕಲು(b_ptr, (ಶೂನ್ಯ *) op_batch->h_bufs.ದತ್ತಾಂಶ(), b_ಗಾತ್ರ);
        ಸ್ಮರಣೆ_ನಕಲು(t_ptr, (ಶೂನ್ಯ *) op_batch->h_tens.ದತ್ತಾಂಶ(), t_ಗಾತ್ರ);
        ಸ್ಮರಣೆ_ನಕಲು(o_ptr, (ಶೂನ್ಯ *) op_batch->h_ops.ದತ್ತಾಂಶ(),  o_ಗಾತ್ರ);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_ಗಾತ್ರ, t_ಗಾತ್ರ, o_ಗಾತ್ರ, (size_t) dbuf.size);

        ಆದರೆ (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (ಶೂನ್ಯ *) b[i].ತಳ, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) t[i].ದತ್ತಾಂಶ, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ ತೆಗೆ(htp_opbatch_rsp rsp, dspqueue_ಬಫರ್ dbuf) {
        ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನ = (uint32_t) ((rsp.seq - 1) % ಆಳ);

        ಸ್ಥಿರಾಂಕ size_t b_ಗಾತ್ರ = ಗಾತ್ರ(htp_buf_desc)  * rsp.n_bufs;
        ಸ್ಥಿರಾಂಕ size_t t_ಗಾತ್ರ = ಗಾತ್ರ(htp_tensor)    * rsp.n_tensors;
        ಸ್ಥಿರಾಂಕ size_t o_ಗಾತ್ರ = ಗಾತ್ರ(htp_op_desc)   * rsp.n_ops;
        ಸ್ಥಿರಾಂಕ size_t p_ಗಾತ್ರ = ಗಾತ್ರ(htp_prof_desc) * rsp.n_ops;

        size_t tr_ಗಾತ್ರ = 0;
        uint32_t n_traces = 0;
        ಆದರೆ (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_ಗಾತ್ರ = (HTP_MAX_NTHREADS + 1) * n_traces * ಗಾತ್ರ(htp_trace_desc);
        }

        ಸ್ಥಿರಾಂಕ size_t m_ಗಾತ್ರ = b_ಗಾತ್ರ + t_ಗಾತ್ರ + o_ಗಾತ್ರ + p_ಗಾತ್ರ + tr_ಗಾತ್ರ;
        GGML_ASSERT(m_ಗಾತ್ರ <= shm_blk_ಗಾತ್ರ);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_ಗಾತ್ರ, t_ಗಾತ್ರ, o_ಗಾತ್ರ);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_ಗಾತ್ರ + t_ಗಾತ್ರ + o_ಗಾತ್ರ);

        ಆದರೆ (rsp.n_ops > 0) {
            ಸ್ವಯಂಚಾಲಿತ & ops = op_cache[ಸ್ಥಾನ];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ಸ್ಥಿರಾಂಕ htp_prof_desc * pd = (ಸ್ಥಿರಾಂಕ htp_prof_desc *) p_ptr;
            ಸ್ಥಿರಾಂಕ htp_trace_desc * trace_events = ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (opt_profile == 3) {
                trace_events = (ಸ್ಥಿರಾಂಕ htp_trace_desc *) (p_ptr + p_ಗಾತ್ರ);
            }

            ಆದರೆ (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            ಗಾಗಿ (uint32_t i = 0; i < rsp.n_ops; i++) {
                ಆದರೆ (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            ಆದರೆ (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

ಶೂನ್ಯ ggml_hexagon_session::flush_peers() {
    ಸ್ವಯಂಚಾಲಿತ vpeers = std::move(virt_peers);
    virt_peers.clear();
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * peer : vpeers) {
        peer->flush_sync();
    }

    ಸ್ವಯಂಚಾಲಿತ ppeers = std::move(phys_peers);
    phys_peers.clear();
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * peer : ppeers) {
        peer->flush_async();
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
        sub->flush_peers();
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

ಶೂನ್ಯ ggml_hexagon_session::flush_pending(ತಾರ್ಕಿಕ all) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
        sub->flush_pending(all);
        ಆದರೆ (sub->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
            ಇದು->ಕೊನೆಯ_error = sub->ಕೊನೆಯ_error.load();
        }
    }

    ಆಗ (ಇದು->batch_rsp_seq < ಇದು->batch_req_seq) {
        ರಚನೆ htp_opbatch_rsp rsp;
        uint32_t               rsp_ಗಾತ್ರ;
        uint32_t               flags;

        ರಚನೆ dspqueue_ಬಫರ್ dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ಸ್ಥಿರಾಂಕ uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        ಪೂರ್ಣಾಂಕ err = dspqueue_read(ಇದು->queue, &flags, 1, &n_dbufs, &dbuf, ಗಾತ್ರ(rsp), &rsp_ಗಾತ್ರ, (uint8_t *) &rsp, timeo);
        ಆದರೆ (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }

        // Basic sanity checks
        ಆದರೆ (rsp_ಗಾತ್ರ != ಗಾತ್ರ(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ಇದು->c_name(), rsp_ಗಾತ್ರ, n_dbufs);
        }

        ಆದರೆ (rsp.ಸ್ಥಿತಿ_2 > HTP_ಸ್ಥಿತಿ_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ಇದು->c_name(), ಸ್ಥಿತಿ_ಗಮ್ಯ_str(rsp.ಸ್ಥಿತಿ_2));
            ಇದು->ಕೊನೆಯ_error = rsp.ಸ್ಥಿತಿ_2;
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
                sub->ಕೊನೆಯ_error = rsp.ಸ್ಥಿತಿ_2;
            }
        }

        op_ಸರತಿ->ತೆಗೆ(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ಇದು->batch_rsp_seq + 1);
        ಇದು->batch_rsp_seq = rsp.seq;

        ಆದರೆ (!all) ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_sync(ತಾರ್ಕಿಕ all) {
    flush_async();
    flush_pending(all);
}

ಶೂನ್ಯ ggml_hexagon_session::ಆರಂಭ_batch() {
    ಆದರೆ (ಇದು->mdev.count > 1) {
        ಸರತಿಗೆ_ಸೇರಿಸು_mdev_ಗುಂಪು();
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_batch(size_t min_ops) {
    ಆದರೆ (op_batch->n_ops < min_ops) { ಹಿಂತಿರುಗಿ; }

    op_batch->ವಿಂಗಡಿಸು_buffers();

    htp_opbatch_req req {};
    dspqueue_ಬಫರ್ dbuf{};

    ಸ್ಥಿರಾಂಕ uint64_t seq = ++ಇದು->batch_req_seq;

    op_batch->update_mdev_ಗುಂಪು(ಇದು->mdev.idx);

    ಆದರೆ (!op_ಸರತಿ->ತಳ್ಳು(req, dbuf, op_batch, seq)) {
        flush_pending(ಸುಳ್ಳು);
        op_ಸರತಿ->ತಳ್ಳು(req, dbuf, op_batch, seq);
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_ಬಫರ್ sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_ಗುಂಪು(sub->mdev.idx);

        ಆದರೆ (!sub->op_ಸರತಿ->ತಳ್ಳು(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(ಸುಳ್ಳು);
            sub->op_ಸರತಿ->ತಳ್ಳು(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        ಪೂರ್ಣಾಂಕ err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, ಗಾತ್ರ(sub_req), (ಸ್ಥಿರಾಂಕ uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        ಆದರೆ (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ಇದು->c_name(), dbuf.ptr, dbuf.size);

    ಪೂರ್ಣಾಂಕ err = dspqueue_write(ಇದು->queue, 0, 1, &dbuf, ಗಾತ್ರ(req), (ಸ್ಥಿರಾಂಕ uint8_t*) &req, DSPQUEUE_TIMEOUT);
    ಆದರೆ (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ಇದು->c_name(), (ಚಿಹ್ನೆರಹಿತ) err);
    }

    op_batch->reset();
}

ಶೂನ್ಯ ggml_hexagon_session::ಸರತಿಗೆ_ಸೇರಿಸು_op(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
    ಸ್ವಯಂಚಾಲಿತ clone_tensor_ಬಫರ್ = [ಇದು](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಸ್ವಯಂಚಾಲಿತ sbuf = ಇದು->mmap_tensor(t);
        ಆದರೆ (!sbuf) ಹಿಂತಿರುಗಿ;
        ಆದರೆ (sbuf->sess != ಇದು) {
            ಇದು->clone_ಬಫರ್(sbuf);
        }
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
            sub->clone_ಬಫರ್(sbuf);
        }
    };

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ t : ಗಂಟು.get_inputs()) {
        clone_tensor_ಬಫರ್(t);
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ t : ಗಂಟು.get_outputs()) {
        clone_tensor_ಬಫರ್(t);
    }

    ಆದರೆ (opt_opfusion && op_batch->try_fuse(ಗಂಟು)) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (!op_batch->fit_op(ಗಂಟು)) {
        flush_async();
    }

    ಆದರೆ (op_batch->ಖಾಲಿಗಳು()) {
        ಆರಂಭ_batch();
    }

    ಆದರೆ (!op_batch->fit_op(ಗಂಟು)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->ಕೂಡಿಸು_op(ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::ಸರತಿಗೆ_ಸೇರಿಸು_mdev_ಗುಂಪು() {
    htp_opnode ಗುಂಪು_ಗಂಟು(HTP_OP_MDEV_ಗುಂಪು);

    uint8_t * fence_ಸ್ಥಾನ = ಇದು->mdev_fence_ಸ್ಥಾನ;

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.ಬಫರ್       = &ಇದು->fence_buf->backend_ಬಫರ್;
    dummy_t.extra        = &fence_extra;
    dummy_t.ದತ್ತಾಂಶ         = (ಶೂನ್ಯ *) fence_ಸ್ಥಾನ;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ;
    dummy_t.ne[1]        = (int64_t) ಇದು->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ಇದು->mdev.idx;

    ggml_tensor * ಗಂಟು = ಗುಂಪು_ಗಂಟು.ಕೂಡಿಸು_dummy(dummy_t);
    ಗಂಟು->src[0] = ಗಂಟು;
    ಗುಂಪು_ಗಂಟು.init(ಗಂಟು);
    ಗುಂಪು_ಗಂಟು.outputs.clear();
    ಗುಂಪು_ಗಂಟು.name = "MDEV_GROUP";

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub : ಇದು->mdev.sessions) {
        sub->clone_ಬಫರ್(ಇದು->fence_buf);
    }

    op_batch->ಕೂಡಿಸು_op(ಗುಂಪು_ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::ಸರತಿಗೆ_ಸೇರಿಸು_cpy(ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ with_fence = sync_tensor != ಶೂನ್ಯಸೂಚಕ;
    htp_opnode cpy_ಗಂಟು(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* ಗಂಟು = cpy_ಗಂಟು.ಕೂಡಿಸು_dummy(*dst);
    ಗಂಟು->op     = GGML_OP_CPY;
    ಗಂಟು->src[0] = const_cast<ggml_tensor *>(src);
    ಗಂಟು->src[1] = with_fence ? cpy_ಗಂಟು.ಕೂಡಿಸು_dummy(*sync_tensor) : ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (with_fence) {
        ಗಂಟು->op_params[0] = (int32_t) fence_seq;
    }

    cpy_ಗಂಟು.init(ಗಂಟು);
    ಆದರೆ (with_fence) {
        cpy_ಗಂಟು.name = "CPY+FENCE";
    }
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = ggml_hexagon_precompute_cpy_params(ಇದು, ಗಂಟು, (ರಚನೆ htp_copy_kernel_params *) cpy_ಗಂಟು.kernel_params);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params *) cpy_ಗಂಟು.kernel_params;
    ಆದರೆ (ok && !with_fence && kparams->ಒಟ್ಟು_elems == 0) {
        ಹಿಂತಿರುಗಿ;
    }
    ಇದು->ಸರತಿಗೆ_ಸೇರಿಸು_op(cpy_ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::ಸರತಿಗೆ_ಸೇರಿಸು_fence(ಸ್ಥಿರಾಂಕ ggml_tensor * sync_tensor, uint32_t fence_seq, ತಾರ್ಕಿಕ wait) {
    htp_opnode sync_ಗಂಟು(HTP_OP_FENCE);

    ggml_tensor* ಗಂಟು = sync_ಗಂಟು.ಕೂಡಿಸು_dummy(*sync_tensor);
    ಗಂಟು->op           = GGML_OP_NONE;
    ಗಂಟು->src[0]       = ಗಂಟು;
    ಗಂಟು->op_params[0] = (int32_t) fence_seq;
    ಗಂಟು->op_params[1] = wait ? 0 : 1;

    sync_ಗಂಟು.init(ಗಂಟು);
    sync_ಗಂಟು.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ಇದು->ಸರತಿಗೆ_ಸೇರಿಸು_op(sync_ಗಂಟು);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_allreduce_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ತಾರ್ಕಿಕ has_ಕೂಡಿಸು,
    ತಾರ್ಕಿಕ is_ಅಡ್ಡಸಾಲು_bcast,
    ತಾರ್ಕಿಕ is_shard_ok,
    ರಚನೆ htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_ಅಡ್ಡಸಾಲು_bcast = (has_ಕೂಡಿಸು && is_ಅಡ್ಡಸಾಲು_bcast) ? 1 : 0;

    ಸ್ಥಿರಾಂಕ uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (dst->type == GGML_TYPE_F16) ? ಗಾತ್ರ(ggml_fp16_t) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_contiguous = ggml_is_contiguous(dst);

    ಸ್ಥಿರಾಂಕ uint32_t ne0 = (uint32_t) dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_1d = is_contiguous && !(has_ಕೂಡಿಸು && is_ಅಡ್ಡಸಾಲು_bcast && ne1 > 1);

    ಆದರೆ (has_ಕೂಡಿಸು) {
        // sharded reduce-scatter for contiguous in-place add
        ಆದರೆ (opt_ar_scatter && is_shard_ok && use_1d && !is_ಅಡ್ಡಸಾಲು_bcast) {
            ಸ್ಥಿರಾಂಕ uint32_t rank_chunk_elems = hex_ಸುತ್ತು_ಮೇಲೆ((nelem + n_ranks - 1) / n_ranks, 128);
            ಸ್ಥಿರಾಂಕ uint32_t rank_elem_ಆರಂಭ  = (std::min)(rank * rank_chunk_elems, nelem);
            ಸ್ಥಿರಾಂಕ uint32_t rank_elem_ಕೊನೆ    = (std::min)(rank_elem_ಆರಂಭ + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_ಆರಂಭ = (int32_t) rank_elem_ಆರಂಭ;
            kparams->rank_nelem      = (int32_t) (rank_elem_ಕೊನೆ - rank_elem_ಆರಂಭ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ಆರಂಭ = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } ಇಲ್ಲದಿದ್ದರೆ {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ಆರಂಭ = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->n_dsts = (int32_t) n_ranks;
        ಆದರೆ (use_1d) {
            ಸ್ಥಿರಾಂಕ uint32_t rank_chunk_elems = hex_ಸುತ್ತು_ಮೇಲೆ((nelem + n_ranks - 1) / n_ranks, 128);
            ಸ್ಥಿರಾಂಕ uint32_t rank_elem_ಆರಂಭ  = (std::min)(rank * rank_chunk_elems, nelem);
            ಸ್ಥಿರಾಂಕ uint32_t rank_elem_ಕೊನೆ    = (std::min)(rank_elem_ಆರಂಭ + rank_chunk_elems, nelem);
            ಸ್ಥಿರಾಂಕ uint32_t rank_nelem       = rank_elem_ಕೊನೆ - rank_elem_ಆರಂಭ;
            kparams->rank_elem_ಆರಂಭ        = (int32_t) rank_elem_ಆರಂಭ;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ uint32_t rank_chunk_ಅಡ್ಡಸಾಲುಗಳು = (ne1 + n_ranks - 1) / n_ranks;
            ಸ್ಥಿರಾಂಕ uint32_t rank_r0         = (std::min)(rank * rank_chunk_ಅಡ್ಡಸಾಲುಗಳು, ne1);
            ಸ್ಥಿರಾಂಕ uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_ಅಡ್ಡಸಾಲುಗಳು, ne1);
            ಸ್ಥಿರಾಂಕ uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_ಆರಂಭ       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ತುಂಬಿದ;

    ಆದರೆ (use_1d) {
        ಸ್ಥಿರಾಂಕ uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, (std::max)(1u, rank_nelem / 128));
        kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ size_t n_vtcm_buffers = htp_allreduce_vtcm_ಬಫರ್_ಎಣಿಕೆ(n_ranks, n_ಎಳೆಗಳು, has_ಕೂಡಿಸು, is_ಅಡ್ಡಸಾಲು_bcast);

        uint32_t ಖಂಡ_elems = 65536;
        ಆದರೆ (ಖಂಡ_elems > rank_nelem / n_ಎಳೆಗಳು && rank_nelem / n_ಎಳೆಗಳು > 128) {
            ಖಂಡ_elems = hex_ಸುತ್ತು_ಮೇಲೆ(rank_nelem / (n_ಎಳೆಗಳು * 2), 128);
        }
        ಖಂಡ_elems = (std::max)(128u, ಖಂಡ_elems);

        kparams->ಖಂಡ_elems          = ಖಂಡ_elems;
        kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * ಖಂಡ_elems * elem_ಗಾತ್ರ;
        kparams->vtcm_ಗಾತ್ರ            = n_vtcm_buffers * kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;

        ಆಗ ((size_t) kparams->vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ && ಖಂಡ_elems > 128) {
            ಸ್ಥಿರಾಂಕ size_t max_ಬೈಟುಗಳು_ಪ್ರತಿ_buf = sess->vtcm_ಗಾತ್ರ / (n_vtcm_buffers * 2);
            ಖಂಡ_elems = (uint32_t) hex_align_ಕೆಳಕ್ಕೆ((size_t) (max_ಬೈಟುಗಳು_ಪ್ರತಿ_buf / elem_ಗಾತ್ರ), 128);
            ಆದರೆ (ಖಂಡ_elems < 128) ನಿಲ್ಲಿಸು;
            kparams->ಖಂಡ_elems          = ಖಂಡ_elems;
            kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * ಖಂಡ_elems * elem_ಗಾತ್ರ;
            kparams->vtcm_ಗಾತ್ರ            = n_vtcm_buffers * kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        }

        ಆದರೆ (sess->vtcm_ಗಾತ್ರ < (size_t) kparams->vtcm_ಗಾತ್ರ || ಖಂಡ_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        kparams->elems_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ((rank_nelem + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು, ಖಂಡ_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, (std::max)(1u, rank_nrows));
        kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ size_t n_vtcm_buffers = htp_allreduce_vtcm_ಬಫರ್_ಎಣಿಕೆ(n_ranks, n_ಎಳೆಗಳು, has_ಕೂಡಿಸು, is_ಅಡ್ಡಸಾಲು_bcast);

        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ne0 * elem_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = (uint32_t) hex_align_ಮೇಲೆ(ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 128);
        kparams->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

        ಸ್ಥಿರಾಂಕ uint32_t nrows_ಪ್ರತಿ_thread = (rank_nrows + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
        uint32_t ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = (std::min)(128u, nrows_ಪ್ರತಿ_thread);
        ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = (std::max)(1u, ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು);
        kparams->ಖಂಡ_elems = ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು;

        kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * (ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        kparams->vtcm_ಗಾತ್ರ            = n_vtcm_buffers * kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;

        ಆಗ ((size_t) kparams->vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ && ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು > 1) {
            ಸ್ಥಿರಾಂಕ size_t max_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_buf = sess->vtcm_ಗಾತ್ರ / (n_vtcm_buffers * 2 * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
            ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = (std::max)(1u, (uint32_t) max_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_buf);
            kparams->ಖಂಡ_elems          = ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು;
            kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * (ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
            kparams->vtcm_ಗಾತ್ರ            = n_vtcm_buffers * kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
            ಆದರೆ (max_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_buf == 0) ನಿಲ್ಲಿಸು;
        }

        ಆದರೆ (sess->vtcm_ಗಾತ್ರ < (size_t) kparams->vtcm_ಗಾತ್ರ || ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        kparams->elems_ಪ್ರತಿ_thread = nrows_ಪ್ರತಿ_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
}

ಶೂನ್ಯ ggml_hexagon_session::ಸರತಿಗೆ_ಸೇರಿಸು_allreduce(
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst,
    ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> & src_tensors,
    ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_ನಮೂದು,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_ಗಂಟು(HTP_OP_ALLREDUCE);

    ggml_tensor* ಗಂಟು = ar_ಗಂಟು.ಕೂಡಿಸು_dummy(*dst);
    ಗಂಟು->op           = GGML_OP_NONE;
    ಗಂಟು->op_params[0] = (int32_t) fence_seq_ನಮೂದು;
    ಗಂಟು->op_params[1] = (int32_t) fence_seq_exit;

    ar_ಗಂಟು.init(ಗಂಟು);

    ar_ಗಂಟು.inputs.clear();
    ಗಾಗಿ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ಗಂಟು.inputs.push_back(src_tensors[i]);
    }
    ಗಾಗಿ (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_ಗಂಟು.inputs.push_back(ar_ಗಂಟು.ಕೂಡಿಸು_dummy(*sync_tensors[i]));
    }

    ar_ಗಂಟು.outputs.clear();
    ಗಾಗಿ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ಗಂಟು.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ಇದು, dst, rank, n_ranks, ಸುಳ್ಳು, ಸುಳ್ಳು, /*is_shard_ok=*/ ಸುಳ್ಳು,
        (ರಚನೆ htp_allreduce_kernel_params *) ar_ಗಂಟು.kernel_params);

    ar_ಗಂಟು.name = "ALLREDUCE";
    ಇದು->ಸರತಿಗೆ_ಸೇರಿಸು_op(ar_ಗಂಟು);
}

ggml_hexagon_shared_ಬಫರ್ * ggml_hexagon_session::mmap_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಆದರೆ (!t) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    ಸ್ವಯಂಚಾಲಿತ sbuf = static_cast<ggml_hexagon_shared_ಬಫರ್ *>(t->ಬಫರ್->context);
    ಆದರೆ (!sbuf->mapped) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ತೂಕ = ggml_backend_ಬಫರ್_get_usage(t->ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ extended  = opt_dma64 && is_ತೂಕ;
        sbuf->mmap(extended);
    }
    ಹಿಂತಿರುಗಿ sbuf;
}

ತಾರ್ಕಿಕ ggml_hexagon_session::clone_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್ * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    ಆದರೆ (sbuf->sess == ಇದು) ಹಿಂತಿರುಗಿ ನಿಜ;

    ಸ್ವಯಂಚಾಲಿತ mem = sbuf->mem;
    ಪೂರ್ಣಾಂಕ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ಆದರೆ (ಇದು->cloned_buffers.find(fd) != ಇದು->cloned_buffers.end()) ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ಇದು->name.c_str(),
                sbuf->c_name(), sbuf->ತಳ(), sbuf->size(), fd);

    ಸ್ವಯಂಚಾಲಿತ clone = std::make_unique<ggml_hexagon_shared_ಬಫರ್>(ಇದು, *sbuf);
    ಪ್ರಯತ್ನಿಸು {
        clone->mmap(sbuf->extended);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ಇದು->c_name(), exc.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಇದು->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ಇದು);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ ggml_hexagon_session::release_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್ * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    ಸ್ವಯಂಚಾಲಿತ mem = sbuf->mem;
    ಪೂರ್ಣಾಂಕ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ಸ್ವಯಂಚಾಲಿತ it = ಇದು->cloned_buffers.find(fd);
    ಆದರೆ (it != ಇದು->cloned_buffers.end()) {
        ಸ್ವಯಂಚಾಲಿತ clone = std::move(it->second);
        ಇದು->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ಇದು);
}

ಶೂನ್ಯ ggml_hexagon_session::unclone_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_hexagon_shared_ಬಫರ್ * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    ಸ್ವಯಂಚಾಲಿತ mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * sess : sessions) {
        sess->release_ಬಫರ್(sbuf);
    }
}

ಸ್ಥಿರ size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_ಬಫರ್ *> sbufs;

    ಸ್ಥಿರಾಂಕ size_t MiB = 1024 * 1024;
    ಸ್ಥಿರಾಂಕ size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t ಹೆಜ್ಜೆ = 256u * MiB;

    ಪ್ರಯತ್ನಿಸು {
        sbufs.push_back(ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, GiB, ನಿಜ)); vmem += GiB;
        sbufs.push_back(ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, GiB, ನಿಜ)); vmem += GiB;
        sbufs.push_back(ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, GiB, ನಿಜ)); vmem += GiB;

        ಆಗ (1) {
            sbufs.push_back(ಹೊಸ ggml_hexagon_shared_ಬಫರ್(sess, ಹೆಜ್ಜೆ, ನಿಜ));
            vmem += ಹೆಜ್ಜೆ;
        }
    } ಹಿಡಿ (...) { }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ b : sbufs) { ಅಳಿಸು b; }

    ಹಿಂತಿರುಗಿ vmem - ಹೆಜ್ಜೆ; // backoff to account for overhead from internal mappings
}

ಶೂನ್ಯ ggml_hexagon_session::allocate(ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config) noexcept(ಸುಳ್ಳು) {
    ಪೂರ್ಣಾಂಕ phys_idx = config.physical_idx;
    ಪೂರ್ಣಾಂಕ virt_idx = config.virtual_idx;

    ಇದು->valid_session = ಸುಳ್ಳು;
    ಇದು->valid_handle  = ಸುಳ್ಳು;
    ಇದು->valid_ಸರತಿ   = ಸುಳ್ಳು;
    ಇದು->valid_iface   = ಸುಳ್ಳು;

    ಇದು->name          = config.name;
    ಇದು->phys_idx      = phys_idx;
    ಇದು->virt_idx      = virt_idx;
    ಇದು->domain_id     = config.domain_id;
    ಇದು->session_id    = 0;
    ಇದು->batch_req_seq = 0;
    ಇದು->batch_rsp_seq = 0;
    ಇದು->ಕೊನೆಯ_error    = HTP_ಸ್ಥಿತಿ_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ಇದು->name.c_str(), ಇದು->domain_id, phys_idx, virt_idx);

    ಆದರೆ (config.domain_id < 0 || config.domain_name.ಖಾಲಿಗಳು()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ಎಸೆ std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ಸ್ಥಿರಾಂಕ std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    ಆದರೆ (virt_idx > 0) {
        ರಚನೆ remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<ಅಕ್ಷರ *>(dom_name.c_str());
        n.session_name     = const_cast<ಅಕ್ಷರ *>(ಇದು->name.c_str());
        n.session_name_len = ಇದು->name.size();
        n.session_id       = virt_idx;

        ಪೂರ್ಣಾಂಕ err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (ಶೂನ್ಯ *) &n, ಗಾತ್ರ(n));
        ಆದರೆ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ಇದು->c_name(), phys_idx, virt_idx, err);
            ಎಸೆ std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ಇದು->session_id    = n.session_id;
        ಇದು->domain_id     = n.effective_domain_id;
        ಇದು->valid_session = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<ಅಕ್ಷರ *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        ಪೂರ್ಣಾಂಕ err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (ಶೂನ್ಯ *) &eff, ಗಾತ್ರ(eff));
        ಆದರೆ (err == AEE_SUCCESS) {
            ಇದು->domain_id = eff.effective_domain_id;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ಇದು->name.c_str(), err, ಇದು->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ರಚನೆ remote_rpc_control_unsigned_module u;
        u.domain = ಇದು->domain_id;
        u.enable = 1;
        ಪೂರ್ಣಾಂಕ err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (ಶೂನ್ಯ *) &u, ಗಾತ್ರ(u));
        ಆದರೆ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ಇದು->c_name(), err);
            ಎಸೆ std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    ಅಕ್ಷರ session_uri[256];
    {
        ಅಕ್ಷರ htp_uri[256];
        snprintf(htp_uri, ಗಾತ್ರ(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ರಚನೆ remote_rpc_get_uri u = {};
        u.session_id      = ಇದು->session_id;
        u.domain_name     = const_cast<ಅಕ್ಷರ *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<ಅಕ್ಷರ *>(htp_uri);
        u.module_uri_len  = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = ಗಾತ್ರ(session_uri);

        ಪೂರ್ಣಾಂಕ err = remote_session_control(FASTRPC_GET_URI, (ಶೂನ್ಯ *) &u, ಗಾತ್ರ(u));
        ಆದರೆ (err != AEE_SUCCESS) {
            snprintf(session_uri, ಗಾತ್ರ(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ಇದು->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ಇದು->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    ಪೂರ್ಣಾಂಕ err = htp_iface_open(session_uri, &ಇದು->handle);
    ಆದರೆ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ಇದು->c_name(), session_uri, err);
        ಎಸೆ std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ಇದು->valid_handle = ನಿಜ;

    // Query HW info and resolve session options
    {
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ hw_n_ಎಳೆಗಳು = 0;
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ hw_n_hvx     = 0;
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ hw_n_hmx     = 0;
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ hw_vtcm_ಗಾತ್ರ = 0;
        ಪೂರ್ಣಾಂಕ hw_err = htp_iface_hwinfo(ಇದು->handle, &hw_n_ಎಳೆಗಳು, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_ಗಾತ್ರ);
        ಆದರೆ (hw_err == 0) {
            ಸ್ಥಿರಾಂಕ uint32_t max_n_ಎಳೆಗಳು = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_ಎಳೆಗಳು);
            ಇದು->n_ಎಳೆಗಳು = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_ಎಳೆಗಳು) : max_n_ಎಳೆಗಳು;
            ಇದು->n_hvx     = ಇದು->n_ಎಳೆಗಳು;
            ಇದು->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ಇದು->vtcm_ಗಾತ್ರ = (uint64_t)hw_vtcm_ಗಾತ್ರ;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ಇದು->c_name(), ಇದು->n_ಎಳೆಗಳು, ಇದು->n_hvx, ಇದು->n_hmx,
                          (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)(ಇದು->vtcm_ಗಾತ್ರ / (1024 * 1024)));
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ಇದು->c_name(), hw_err);
            ಸ್ಥಿರಾಂಕ uint32_t default_n_ಎಳೆಗಳು = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ಇದು->n_ಎಳೆಗಳು = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_ಎಳೆಗಳು;
            ಇದು->n_hvx     = ಇದು->n_ಎಳೆಗಳು;
            ಇದು->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ಇದು->vtcm_ಗಾತ್ರ = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ರಚನೆ remote_rpc_control_latency l;
        l.enable = 1;

        ಪೂರ್ಣಾಂಕ err = remote_handle64_control(ಇದು->handle, DSPRPC_CONTROL_LATENCY, (ಶೂನ್ಯ *) &l, ಗಾತ್ರ(l));
        ಆದರೆ (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ಇದು->c_name(),
                  ಇದು->session_id, ಇದು->domain_id, session_uri, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) ಇದು->handle);

    ಸ್ಥಿರಾಂಕ size_t req_q_ಗಾತ್ರ = (ಗಾತ್ರ(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ಸ್ಥಿರಾಂಕ size_t rsp_q_ಗಾತ್ರ = (ಗಾತ್ರ(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ಇದು->domain_id,
                          0,              // Flags
                          req_q_ಗಾತ್ರ,     // Request  queue size (in bytes)
                          rsp_q_ಗಾತ್ರ,     // Response queue size (in bytes)
                          ಶೂನ್ಯಸೂಚಕ,        // Read packet callback (we handle reads explicitly)
                          ಶೂನ್ಯಸೂಚಕ,        // Error callback (we handle errors during reads)
                          (ಶೂನ್ಯ *) ಇದು,  // Callback context
                          &queue);
    ಆದರೆ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ಇದು->name.c_str(), (ಚಿಹ್ನೆರಹಿತ) err);
        ಎಸೆ std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ಇದು->valid_ಸರತಿ = ನಿಜ;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ಇದು->ಸರತಿ_id);
    ಆದರೆ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        ಎಸೆ std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    ಆದರೆ (opt_etm) {
        err = htp_iface_etm(ಇದು->handle, 1);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    // Allocate buffers and state for op batching
    ಇದು->op_ಸರತಿ = ಹೊಸ ggml_hexagon_opqueue(ಇದು, opt_opbatch, opt_opqueue);

    ಇದು->fence_buf = ಹೊಸ ggml_hexagon_fence_ಬಫರ್(ಇದು, &dev_ctx->fence_ಬಫರ್_type, 64 * 1024);
    ಆದರೆ (ಇದು->mdev.count > 1) {
        ಇದು->mdev_fence_ಸ್ಥಾನ = ಇದು->alloc_fence(ಇದು->mdev.count);
    }

    ಆದರೆ (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ಇದು);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ಇದು->c_name(), opt_vmem);
    }
    ಸ್ಥಿರಾಂಕ size_t shm_ಗಾತ್ರ = ಇದು->op_ಸರತಿ->shm_ಗಾತ್ರ();
    ಇದು->max_vmem = (opt_vmem > shm_ಗಾತ್ರ) ? (opt_vmem - shm_ಗಾತ್ರ) : opt_vmem;

    ಇದು->op_batch = ಹೊಸ ggml_hexagon_opbatch(ಇದು, opt_opbatch, ಇದು->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_ಆರಂಭ(ಇದು->handle, ಇದು->session_id, ಇದು->ಸರತಿ_id, ಇದು->n_ಎಳೆಗಳು, opt_nhmx, ಇದು->max_vmem, (uint32_t) opt_power_ಬಹುಲಕ);
    ಆದರೆ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ಇದು->c_name(), (ಚಿಹ್ನೆರಹಿತ) err);
        ಎಸೆ std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ಇದು->valid_iface = ನಿಜ;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ಇದು->c_name(), htp_power_ಬಹುಲಕ_ಗಮ್ಯ_str(opt_power_ಬಹುಲಕ));

    ಆದರೆ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ಇದು->handle, opt_profile, &pmu_conf);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }
}

ಶೂನ್ಯ ggml_hexagon_session::release() noexcept(ನಿಜ) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ಇದು->name.c_str());

    ಇದು->mdev.sessions.clear();

    ಪೂರ್ಣಾಂಕ err;

    ಆದರೆ (ಇದು->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ಇದು->handle);
        ಆದರೆ (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    ಅಳಿಸು ಇದು->op_batch;
    ಅಳಿಸು ಇದು->op_ಸರತಿ;
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & it : ಇದು->cpy_fence_slots) {
        free_fence((ಶೂನ್ಯ *) it.second, 1);
    }
    ಇದು->cpy_fence_slots.clear();

    ಆದರೆ (ಇದು->fence_buf) {
        unclone_ಬಫರ್(ಇದು->fence_buf);
        ಅಳಿಸು ಇದು->fence_buf;
        ಇದು->fence_buf = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆಗ (!ಇದು->cloned_buffers.ಖಾಲಿಗಳು()) {
        release_ಬಫರ್(ಇದು->cloned_buffers.begin()->second.get());
    }

    ಆದರೆ (opt_etm) {
        err = htp_iface_etm(ಇದು->handle, 0);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    ಆದರೆ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ಇದು->handle, 0, &pmu_conf);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    ಆದರೆ (ಇದು->valid_ಸರತಿ) {
        err = dspqueue_close(queue);
        ಆದರೆ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    }

    ಆದರೆ (ಇದು->valid_handle) {
        htp_iface_close(ಇದು->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ಸ್ಥಿರಾಂಕ ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_ಎಣಿಕೆ) noexcept(ಸುಳ್ಳು) {
    ಇದು->dev        = dev;
    ಇದು->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಇದು->mdev.idx   = mdev_idx;
    ಇದು->mdev.count = mdev_ಎಣಿಕೆ > 0 ? mdev_ಎಣಿಕೆ : (uint32_t) (1 + config.mdev_ಗುಂಪು.size());
    op_batch         = ಶೂನ್ಯಸೂಚಕ;
    op_ಸರತಿ         = ಶೂನ್ಯಸೂಚಕ;
    fence_buf        = ಶೂನ್ಯಸೂಚಕ;
    fence_seq        = ((uintptr_t)ಇದು) & 0xFFFF;

    ಪ್ರಯತ್ನಿಸು {
        allocate(config);
        ಆದರೆ (ಇದು->mdev.idx == 0 && !config.mdev_ಗುಂಪು.ಖಾಲಿಗಳು()) {
            ಗಾಗಿ (size_t i = 0; i < config.mdev_ಗುಂಪು.size(); i++) {
                ಇದು->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_ಗುಂಪು[i], ಇದು->dev, (uint32_t) (i + 1), ಇದು->mdev.count));
            }
        }
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & exc) {
        release();
        ಎಸೆ;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(ನಿಜ) {
    release();
}

// ** backend interface

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_flash_attn_is_hmx_eligible(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * k,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * v,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * sinks
) {
    ಆದರೆ (sess->n_hmx == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (opt_fa_select < 2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t DK = q->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    ಆದರೆ (DK % 8 != 0 || DV % 8 != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    GGML_ಬಳಸದ(sinks);

    // Explicit force mode
    ಆದರೆ (opt_fa_select > 2) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t M = q->ne[1];

    // Prefill or batched decode
    ಆದರೆ (M > 1) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = q->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_kv_heads = k->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t G = n_kv_heads > 0 ? n_ತಲೆ / n_kv_heads : 1;
    ಸ್ಥಿರಾಂಕ uint32_t S = k->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    ಆದರೆ (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < ಒಟ್ಟು_ಎಳೆಗಳು))) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // Tile alignment for 32-row HMX tiles
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tile_aligned = (G > 0 && (32 % G == 0));
    ಆದರೆ (!is_tile_aligned) {
        ಆದರೆ (sess->n_ಎಳೆಗಳು >= 6) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    ಆದರೆ (DK == 256) {
        s_cross = 4096;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (DK <= 64) {
        s_cross = (sess->n_ಎಳೆಗಳು >= 8) ? 2048 : ((sess->n_ಎಳೆಗಳು >= 6) ? 768 : 512);
    } ಇಲ್ಲದಿದ್ದರೆ {
        s_cross = (sess->n_ಎಳೆಗಳು >= 8) ? 1024 : 512;
    }

    ಹಿಂತಿರುಗಿ S >= s_cross;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_flash_attn_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_fa_kernel_params * kparams
) {
    ಆದರೆ (opt_fa_select < 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * q    = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * k    = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * v    = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮುಖವಾಡ = op->src[3];

    ಸ್ಥಿರಾಂಕ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ಸ್ಥಿರಾಂಕ uint32_t neq1 = q->ne[1];  // n_tokens
    ಸ್ಥಿರಾಂಕ uint32_t neq2 = q->ne[2];  // n_heads

    ಸ್ಥಿರಾಂಕ uint32_t nek1 = k->ne[1];  // kv_len

    ಸ್ಥಿರಾಂಕ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ಸ್ಥಿರಾಂಕ uint32_t DK = neq0;
    ಸ್ಥಿರಾಂಕ uint32_t DV = nev0;

    ಸ್ಥಿರಾಂಕ uint32_t n_kv_heads = k->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t G          = neq2 / n_kv_heads;

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale,         &op->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      &op->op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, &op->op_params[2], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->ತಲೆ_ವಿಭಜಿಸು = (opt_fa_ತಲೆ_ವಿಭಜಿಸು != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = q->ne[2];
    kparams->n_ತಲೆ_log2 = 1u << (uint32_t) std::floor(std::log2(n_ತಲೆ));
    kparams->m0 = std::ಘಾತ(2.0f, -(max_bias) / kparams->n_ತಲೆ_log2);
    kparams->m1 = std::ಘಾತ(2.0f, -(max_bias / 2.0f) / kparams->n_ತಲೆ_log2);

    // Check HMX eligibility
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * sinks = op->src[4];
    ಆದರೆ (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ಸ್ಥಿರಾಂಕ uint32_t DK_pad = hex_ಸುತ್ತು_ಮೇಲೆ(DK, 64);
        ಸ್ಥಿರಾಂಕ uint32_t DV_pad = hex_ಸುತ್ತು_ಮೇಲೆ(DV, 64);
        size_t Br = 0, Bc = 0;
        ಪೂರ್ಣಾಂಕ ret = hmx_fa_ಕಂಡುಹಿಡಿ_chunk_ಗಾತ್ರ(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_ಗಾತ್ರ, sess->n_ಎಳೆಗಳು, (q->type == GGML_TYPE_F32), sinks != ಶೂನ್ಯಸೂಚಕ, n_ತಲೆ);
        ಆದರೆ (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            ಆದರೆ (logit_softcap == 0.0f) {
                kparams->scale = scale * (ದಶಮಾಂಶ) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } ಇಲ್ಲದಿದ್ದರೆ {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ದಶಮಾಂಶ) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_ಖಂಡಗಳು = (nek1 + Bc - 1) / Bc;
            kparams->n_ಎಳೆಗಳು = (kparams->n_kv_ಖಂಡಗಳು >= 3 && sess->n_ಎಳೆಗಳು >= 2) ? sess->n_ಎಳೆಗಳು : 1;

            kparams->u.hmx.g_br = hex_align_ಮೇಲೆ(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_ಖಂಡಗಳು >= 3 && sess->n_ಎಳೆಗಳು >= 2) ? 1 : 0;
            kparams->vtcm_ಗಾತ್ರ = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_ಎಳೆಗಳು, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != ಶೂನ್ಯಸೂಚಕ, n_ತಲೆ);

            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_vec_ಬೈಟುಗಳು = hex_align_ಮೇಲೆ(Bc * ಗಾತ್ರ(uint16_t), 256);
            kparams->u.hmx.ಅಡ್ಡಸಾಲು_buf_stride = ಅಡ್ಡಸಾಲು_vec_ಬೈಟುಗಳು / 128; // HVX vector is 128 bytes

            ಸ್ಥಿರಾಂಕ size_t m_ಸಾಲು_ಬೈಟುಗಳು = hex_align_ಮೇಲೆ(Bc * ಗಾತ್ರ(uint16_t), 128);
            kparams->u.hmx.ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride = m_ಸಾಲು_ಬೈಟುಗಳು / ಗಾತ್ರ(uint16_t);
            kparams->u.hmx.ಮುಖವಾಡ_ಪ್ರಸಾರ = (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ && ಮುಖವಾಡ->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_ಮೌಲ್ಯಗಳು(G);
            ಆದರೆ (ಮುಖವಾಡ) {
                kparams->src3_div2 = init_fastdiv_ಮೌಲ್ಯಗಳು(ಮುಖವಾಡ->ne[2]);
                kparams->src3_div3 = init_fastdiv_ಮೌಲ್ಯಗಳು(ಮುಖವಾಡ->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_ಪ್ರತಿ_thread = 0;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_ಖಂಡಗಳು = (k->ne[1] + 64 - 1) / 64;
    kparams->n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(k->ne[0] * 2, 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(v->ne[0] * 2, 128);

    kparams->vtcm_ಗಾತ್ರ = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ, sinks != ಶೂನ್ಯಸೂಚಕ, n_ತಲೆ, sess->n_ಎಳೆಗಳು);

    kparams->u.hvx.ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded = ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded;
    kparams->u.hvx.ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded = ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded;
    kparams->u.hvx.ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded = ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[1]);
    kparams->ಪ್ರಸಾರ_rk2 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2]/k->ne[2]);
    kparams->ಪ್ರಸಾರ_rk3 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[3]/k->ne[3]);
    kparams->ಪ್ರಸಾರ_rv2 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2]/v->ne[2]);
    kparams->ಪ್ರಸಾರ_rv3 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[3]/v->ne[3]);
    ಆದರೆ (ಮುಖವಾಡ) {
        kparams->src3_div2 = init_fastdiv_ಮೌಲ್ಯಗಳು(ಮುಖವಾಡ->ne[2]);
        kparams->src3_div3 = init_fastdiv_ಮೌಲ್ಯಗಳು(ಮುಖವಾಡ->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_ಪ್ರತಿ_thread = (kparams->qrows + sess->n_ಎಳೆಗಳು - 1) / sess->n_ಎಳೆಗಳು;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_flash_attn_ext(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src3 = op->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src4 = op->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    ಆದರೆ ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src3 && src3->type != GGML_TYPE_F16) {  //ಮುಖವಾಡ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_fa_kernel_params kparams;
    ಆದರೆ (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_gated_delta_net(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಆದರೆ (opt_gdn_select < 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * q     = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * k     = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * v     = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * g     = op->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * beta  = op->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * state = op->src[5];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst   = op;

    ಆದರೆ (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(q) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(k) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t S_v      = v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H        = v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = v->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t K        = ggml_get_op_params_i32(op, 0);

    ಆದರೆ (S_v <= 0 || S_v > 128 || H <= 0 || n_ಟೋಕನುಗಳು <= 0 || n_seqs <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_ಟೋಕನುಗಳು || k->ne[2] != n_ಟೋಕನುಗಳು || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    ಆದರೆ (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->ne[0] != S_v * H || dst->ne[1] != n_ಟೋಕನುಗಳು * n_seqs + S_v * n_seqs * K) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = (uint32_t) (H * n_seqs);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_ಟೋಕನುಗಳು >= HTP_GDN_MIN_ಟೋಕನುಗಳು) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ಆದರೆ (can_use_hmx) {
        ರಚನೆ htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        ಆದರೆ (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, sess->vtcm_ಗಾತ್ರ, n_ಎಳೆಗಳು, ನಿಜ, &n_heads_batch)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_ಎಳೆಗಳು);
        ಆದರೆ (layout.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_is_hmx_eligible(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ತಾರ್ಕಿಕ is_matmul_id,
    ತಾರ್ಕಿಕ is_batched
) {
    GGML_ಬಳಸದ(dst);

    ಆದರೆ (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00  = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01  = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11  = src1->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12  = src1->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    ಆದರೆ (ne01 % HTP_TENSOR_TILE_N_ಅಡ್ಡಸಾಲುಗಳು != 0 && is_matmul_id) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // HMX supports F16, F32, and repack quantized types.
    ಆದರೆ (!htp_is_hmx_ತೂಕ_type((ggml_type) wtype)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    ಆದರೆ (ne00 % HTP_TENSOR_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಆದರೆ (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!is_matmul_id && is_batched && is_tiled) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // HMX assumes contiguous row-major layout.
    ಆದರೆ (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m = is_matmul_id ? ne12 : ne11;
    ಆದರೆ (m <= HTP_MM_HMX_MIN_NROWS) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(dst);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_hmx_mm_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * act,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ಪೂರ್ಣಾಂಕ wtype,
    ಪೂರ್ಣಾಂಕ ne00_padded,
    ಪೂರ್ಣಾಂಕ ne01_padded,
    ಪೂರ್ಣಾಂಕ ne02,
    ಪೂರ್ಣಾಂಕ ne11,
    ಪೂರ್ಣಾಂಕ ne12,
    ಪೂರ್ಣಾಂಕ ne11_padded,
    ತಾರ್ಕಿಕ is_matmul_id,
    ತಾರ್ಕಿಕ is_batched,
    size_t src2_ಗಾತ್ರ,
    size_t vtcm_budget,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ aligned_tile_ಗಾತ್ರ = is_tiled ? htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype) : 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = (ಪೂರ್ಣಾಂಕ)sess->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10 = act->ne[0];

    ಪೂರ್ಣಾಂಕ m_for_solver = ne11;
    ಪೂರ್ಣಾಂಕ m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    ಆದರೆ (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dst_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು = dst ? ggml_hexagon_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst) : ನಿಜ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ act_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು = ggml_hexagon_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(act);
        ಆದರೆ (dst_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು && act_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು) {
            m_for_solver = (ne11 + (ಪೂರ್ಣಾಂಕ) sess->mdev.count - 1) / (ಪೂರ್ಣಾಂಕ) sess->mdev.count;
            m_for_solver_padded = hex_ಸುತ್ತು_ಮೇಲೆ(std::max(m_for_solver, 32), 32);
        }
    }
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ pipeline = is_matmul_id ? ಸುಳ್ಳು : htp_mm_hmx_pipeline(m_for_solver);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_batched_val = is_matmul_id ? ಸುಳ್ಳು : is_batched;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುಂಪು_ಗಾತ್ರ = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_ಗಾತ್ರ = 0;
    ತಾರ್ಕಿಕ use_grouped = ಸುಳ್ಳು;
    ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು_selected = 0;

    ಆದರೆ (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && ಗುಂಪು_ಗಾತ್ರ > 1) {
        // Try grouped path first
        ಆದರೆ (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, ಗುಂಪು_ಗಾತ್ರ, n_ಎಳೆಗಳು, pipeline, src2_ಗಾತ್ರ, vtcm_budget, &m_chunk, &n_chunk, &act_ಎಳೆಗಳು_selected, &vtcm_ಗಾತ್ರ)) {
            use_grouped = ನಿಜ;
        }
    }

    ಆದರೆ (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        ಪೂರ್ಣಾಂಕ m_id_ಅಡ್ಡಸಾಲುಗಳು = 0;
        ಆದರೆ (dst && is_matmul_id) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts = ne02 > 0 ? ne02 : 1;
            ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_expert_ಅಡ್ಡಸಾಲುಗಳು = (size_t) dst->ne[1] * dst->ne[2];
            ಪೂರ್ಣಾಂಕ m_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) ((ಒಟ್ಟು_expert_ಅಡ್ಡಸಾಲುಗಳು + n_experts - 1) / n_experts);
            ಆದರೆ (sess->mdev.count > 1 && ggml_hexagon_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst)) {
                m_ಪ್ರತಿ_expert = (m_ಪ್ರತಿ_expert + (ಪೂರ್ಣಾಂಕ) sess->mdev.count - 1) / (ಪೂರ್ಣಾಂಕ) sess->mdev.count;
            }
            m_id_ಅಡ್ಡಸಾಲುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(std::max(m_ಪ್ರತಿ_expert, 32), 32);
        }
        ಸ್ಥಿರಾಂಕ uint32_t ವೆಚ್ಚ_m = is_matmul_id ? (uint32_t) m_id_ಅಡ್ಡಸಾಲುಗಳು : (uint32_t) m_for_solver;
        ಆದರೆ (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_ಅಡ್ಡಸಾಲುಗಳು, ne01_padded, m_for_solver_padded, ವೆಚ್ಚ_m, n_ಎಳೆಗಳು, pipeline, is_matmul_id, aligned_tile_ಗಾತ್ರ, src2_ಗಾತ್ರ, vtcm_budget, &m_chunk, &n_chunk, &act_ಎಳೆಗಳು_selected, &vtcm_ಗಾತ್ರ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    kparams->n_act_ಎಳೆಗಳು = act_ಎಳೆಗಳು_selected;
    kparams->tile_ಗಾತ್ರ = is_tiled ? htp_tensor_get_tile_ಗಾತ್ರ(wtype) : 0;
    kparams->aligned_tile_ಗಾತ್ರ = aligned_tile_ಗಾತ್ರ;
    kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
    kparams->vtcm_ಗಾತ್ರ = vtcm_ಗಾತ್ರ;
    kparams->vtcm_src0_ಗಾತ್ರ = 0;
    kparams->div_n_act_ಎಳೆಗಳು = init_fastdiv_ಮೌಲ್ಯಗಳು(act_ಎಳೆಗಳು_selected);
    kparams->div_ne00_padded   = init_fastdiv_ಮೌಲ್ಯಗಳು(ne00_padded);
    kparams->vtcm_act_ಗಾತ್ರ     = 0;
    kparams->vtcm_bias_ಗಾತ್ರ    = (int32_t) src2_ಗಾತ್ರ;
    kparams->vtcm_dst_ಗಾತ್ರ     = 0;

    ಆದರೆ (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_hvx_mm_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ಪೂರ್ಣಾಂಕ wtype,
    ತಾರ್ಕಿಕ is_tiled,
    ಪೂರ್ಣಾಂಕ ne02,
    ಪೂರ್ಣಾಂಕ ne03,
    ಪೂರ್ಣಾಂಕ ne10,
    ಪೂರ್ಣಾಂಕ ne11,
    ಪೂರ್ಣಾಂಕ ne12,
    ಪೂರ್ಣಾಂಕ ne13,
    ತಾರ್ಕಿಕ is_matmul_id,
    ಸ್ಥಿರಾಂಕ size_t src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t vtcm_budget,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    ಆದರೆ (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    }

    kparams->n_hmx = 0;
    kparams->n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;

    GGML_ಬಳಸದ(ne02);
    GGML_ಬಳಸದ(ne03);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nrows = ne11 * ne12 * ne13;

    ಆದರೆ (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(wtype);
        kparams->aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        ಆದರೆ (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (ಪೂರ್ಣಾಂಕ) sess->n_ಎಳೆಗಳು)
                                  ? HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ : HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು;
            kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);

            ರಚನೆ htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t ಉತ್ತಮ_n_prefetch = 2;
            ಗಾಗಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                    0, src0->nb[1], kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, d, ನಿಜ, ಸುಳ್ಳು
                );
                ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_budget) {
                    ಉತ್ತಮ_n_prefetch = d;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                ಹಿಂತಿರುಗಿ;
            }
            kparams->n_prefetch     = ಉತ್ತಮ_n_prefetch;
            kparams->vtcm_ಗಾತ್ರ      = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು;
            kparams->vtcm_act_ಗಾತ್ರ  = L.act_ಬೈಟುಗಳು;
            kparams->vtcm_bias_ಗಾತ್ರ = 0;
            kparams->vtcm_dst_ಗಾತ್ರ  = L.dst_ಬೈಟುಗಳು;
            ಹಿಂತಿರುಗಿ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ತಾರ್ಕಿಕ try_tiled = (k_align && opt_mm_select >= 1);
            ಆದರೆ (try_tiled) {
                kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
                ಆದರೆ (src1_nrows < (ಪೂರ್ಣಾಂಕ) sess->n_ಎಳೆಗಳು) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು;
                }

                ರಚನೆ htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t ಉತ್ತಮ_n_prefetch = 2;
                ಗಾಗಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, d, ಸುಳ್ಳು, ಸುಳ್ಳು
                    );
                    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_budget) {
                        ಉತ್ತಮ_n_prefetch = d;
                        ನಿಲ್ಲಿಸು;
                    }
                }

                uint32_t m_chunk = 0;
                ಆದರೆ (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಉತ್ತಮ_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = ಉತ್ತಮ_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_ಗಾತ್ರ      = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
                    kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು;
                    kparams->vtcm_act_ಗಾತ್ರ  = L.act_ಬೈಟುಗಳು;
                    kparams->vtcm_bias_ಗಾತ್ರ = L.bias_ಬೈಟುಗಳು;
                    kparams->vtcm_dst_ಗಾತ್ರ  = L.dst_ಬೈಟುಗಳು;
                    ಹಿಂತಿರುಗಿ;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ಹಿಂತಿರುಗಿ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ರಚನೆ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಆದರೆ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * 2, 128);
            kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು;
            kparams->vtcm_act_ಗಾತ್ರ  = L.act_ಬೈಟುಗಳು;
            kparams->vtcm_bias_ಗಾತ್ರ = L.bias_ಬೈಟುಗಳು;
            kparams->vtcm_dst_ಗಾತ್ರ = L.dst_ಬೈಟುಗಳು;
            kparams->n_prefetch = 16;
            ಹಿಂತಿರುಗಿ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ರಚನೆ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಆದರೆ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * 4, 128);
            kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು;
            kparams->vtcm_act_ಗಾತ್ರ  = L.act_ಬೈಟುಗಳು;
            kparams->vtcm_bias_ಗಾತ್ರ = L.bias_ಬೈಟುಗಳು;
            kparams->vtcm_dst_ಗಾತ್ರ = L.dst_ಬೈಟುಗಳು;
            kparams->n_prefetch = 16;
            ಹಿಂತಿರುಗಿ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ರಚನೆ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಆದರೆ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_ಎಳೆಗಳು,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * 4, 128);
            kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು;
            kparams->vtcm_act_ಗಾತ್ರ  = L.act_ಬೈಟುಗಳು;
            kparams->vtcm_bias_ಗಾತ್ರ = L.bias_ಬೈಟುಗಳು;
            kparams->vtcm_dst_ಗಾತ್ರ = L.dst_ಬೈಟುಗಳು;
            kparams->n_prefetch = 16;
            ಹಿಂತಿರುಗಿ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params_impl(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ಸ್ಥಿರಾಂಕ size_t src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ size_t src2_ಗಾತ್ರ,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_padded = is_tiled ? hex_ಸುತ್ತು_ಮೇಲೆ(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11_padded = hex_ಸುತ್ತು_ಮೇಲೆ(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01_tiled  = hex_ಸುತ್ತು_ಮೇಲೆ(ne01, 32);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ಸ್ಥಿರಾಂಕ size_t vtcm_budget = sess->vtcm_ಗಾತ್ರ;

    // Check HMX eligibility and try precomputing HMX parameters
    ತಾರ್ಕಿಕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ಆದರೆ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        ಆದರೆ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_ಗಾತ್ರ, vtcm_budget, kparams)) {
            ಹೋಗು finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ಮೌಲ್ಯಗಳು(ne11);
    kparams->div_r2       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(ಸ್ಥಿರಾಂಕ int64_t * ne, ಸ್ಥಿರಾಂಕ size_t * nb, size_t * stride) {
    size_t s = 0, ಮುಂದಿನ = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (ne[i] == 1) ಮುಂದುವರಿ;
        ಆದರೆ (s == 0) { s = nb[i]; ಮುಂದಿನ = s * ne[i]; ಮುಂದುವರಿ; }
        ಆದರೆ (nb[i] != ಮುಂದಿನ) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಮುಂದಿನ *= ne[i];
    }
    *stride = s ? s : nb[1];
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_can_collapse(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst) {
    size_t s1, sd;
    ಹಿಂತಿರುಗಿ (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ಕೂಡಿಸು) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_ಗಾತ್ರ(src1->type) && ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_ಗಾತ್ರ(dst->type) && ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(dst->ne, dst->nb, &sd);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_ಕೂಡಿಸು_can_collapse(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst
) {
    ಆದರೆ (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!src2) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಸ್ಥಿರಾಂಕ int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    ಆದರೆ (src2_nrows == 1) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    size_t s2;
    ಹಿಂತಿರುಗಿ src2->nb[0] == ggml_type_ಗಾತ್ರ(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(src2->ne, src2->nb, &s2);
}

ಸ್ಥಿರ ggml_tensor ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_ಅಡ್ಡಸಾಲುಗಳು_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    ಹಿಂತಿರುಗಿ c;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    ಆದರೆ (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ಸ್ಥಿರಾಂಕ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(src1);
        ಸ್ಥಿರಾಂಕ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        ಹಿಂತಿರುಗಿ;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_matmul_ಕೂಡಿಸು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    ಆದರೆ (ggml_hexagon_matmul_ಕೂಡಿಸು_can_collapse(src0, src1, src2, dst)) {
        ಸ್ಥಿರಾಂಕ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(src1);
        ಸ್ಥಿರಾಂಕ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(dst);
        ಸ್ಥಿರಾಂಕ ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * p_src2 = src2 ? &src2_collapsed : ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಿರಾಂಕ size_t src2_ಗಾತ್ರ = p_src2 ? hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_ಗಾತ್ರ, kparams);
        kparams->collapse = 1;
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ size_t src2_ಗಾತ್ರ = src2 ? hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_ಗಾತ್ರ, kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    uint32_t op,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params * kparams,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1->ne[0] * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst->ne[0]  * elem_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,  128);

    ಆದರೆ (!htp_ದ್ವಿಮಾನ_strides_ಸುರಕ್ಷಿತ(src0, src1, dst, elem_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಕೂಡಿಸು_id = op == HTP_OP_ಕೂಡಿಸು_ID;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_scalar = !is_ಕೂಡಿಸು_id && src1->ne[0] == 1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಅಡ್ಡಸಾಲು_bcast = !is_ಕೂಡಿಸು_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_same_shape = !is_ಕೂಡಿಸು_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_complex   = !is_ಕೂಡಿಸು_id && !is_scalar && !is_same_shape && !is_ಅಡ್ಡಸಾಲು_bcast && (src1->ne[0] == src0->ne[0]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_scalar_ಪ್ರಸಾರ = !is_ಕೂಡಿಸು_id && (ggml_nelements(src1) == 1);

    ಆದರೆ (!is_ಕೂಡಿಸು_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ಪ್ರಸಾರ)) {
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_elems = (uint32_t) ggml_nelements(src0);
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ uint32_t max_chunk_elems = 32768 / elem_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t min_chunk_elems = 256;
        ಸ್ಥಿರಾಂಕ uint32_t ಗುರಿ_chunk_elems = hex_ಸುತ್ತು_ಮೇಲೆ((ಒಟ್ಟು_elems + (2 * n_ಎಳೆಗಳು) - 1) / (2 * n_ಎಳೆಗಳು), 32);
        ಸ್ಥಿರಾಂಕ uint32_t chunk_ಗಾತ್ರ = (std::min)(max_chunk_elems, (std::max)(ಗುರಿ_chunk_elems, min_chunk_elems));
        ಸ್ಥಿರಾಂಕ uint32_t chunk_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * elem_ಗಾತ್ರ, 128);

        kparams->kernel_type           = HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED;
        kparams->n_ಎಳೆಗಳು             = n_ಎಳೆಗಳು;
        kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್       = 1;
        kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = is_scalar_ಪ್ರಸಾರ ? 0 : src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        kparams->src1_ಗಾತ್ರ             = 0;
        kparams->chunk_ಗಾತ್ರ            = chunk_ಗಾತ್ರ;
        kparams->chunk_ಬೈಟುಗಳು           = chunk_ಬೈಟುಗಳು;
        kparams->is_scalar             = is_scalar_ಪ್ರಸಾರ ? 1 : 0;

        ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout L;
        htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_ಗಾತ್ರ);
        ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು == 0 || L.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಎಣಿಕೆ htp_ದ್ವಿಮಾನ_kernel_type kernel_type;
    size_t src1_ಗಾತ್ರ = 0;
    size_t src2_ಗಾತ್ರ = 0;

    ಆದರೆ (is_ಕೂಡಿಸು_id) {
        GGML_ASSERT(src2 != ಶೂನ್ಯಸೂಚಕ);
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_ಕೂಡಿಸು_ID;
        src1_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(src1->ne[1] * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, 128);
        src2_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nelements(src2) * ಗಾತ್ರ(int32_t), 128);
        ಆದರೆ (src1_ಗಾತ್ರ + src2_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_ಅಡ್ಡಸಾಲು_bcast) {
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST;
        src1_ಗಾತ್ರ = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_scalar) {
        src1_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nelements(src1) * elem_ಗಾತ್ರ, 128);
        ಆದರೆ (src1_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_same_shape) {
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_complex) {
        src1_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nelements(src1) * elem_ಗಾತ್ರ + 128, 128);
        ಆದರೆ (src1_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        src1_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ggml_nelements(src1) * elem_ಗಾತ್ರ + 128, 128);
        ಆದರೆ (src1_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;
    kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->src1_ಗಾತ್ರ             = src1_ಗಾತ್ರ;
    kparams->src2_ಗಾತ್ರ             = src2_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    ತಾರ್ಕಿಕ ವಿಭಜಿಸು_at_ne01 = ಸುಳ್ಳು;
    ಆದರೆ (src0->ne[2] > 1) {
        ಆದರೆ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE) {
            ವಿಭಜಿಸು_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST) {
            ವಿಭಜಿಸು_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ವಿಭಜಿಸು_at_ne01 = ನಿಜ;
        }
    }
    kparams->ವಿಭಜಿಸು_at_ne01 = ವಿಭಜಿಸು_at_ne01 ? 1 : 0;

    ತಾರ್ಕಿಕ ವಿಭಜಿಸು_at_ne02 = ಸುಳ್ಳು;
    ಆದರೆ (src0->ne[3] > 1) {
        ಆದರೆ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE) {
            ವಿಭಜಿಸು_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST) {
            ವಿಭಜಿಸು_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ವಿಭಜಿಸು_at_ne02 = ನಿಜ;
        }
    }
    kparams->ವಿಭಜಿಸು_at_ne02 = ವಿಭಜಿಸು_at_ne02 ? 1 : 0;

    ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout L;
    htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_ಗಾತ್ರ);
    ಆದರೆ (L.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ == 0 || L.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
        ಆದರೆ (!is_ಕೂಡಿಸು_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ಪ್ರಸಾರ)) {
            ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_elems = (uint32_t) ggml_nelements(src0);
            ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;
            ಸ್ಥಿರಾಂಕ uint32_t max_chunk_elems = 32768 / elem_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t min_chunk_elems = 256;
            ಸ್ಥಿರಾಂಕ uint32_t ಗುರಿ_chunk_elems = hex_ಸುತ್ತು_ಮೇಲೆ((ಒಟ್ಟು_elems + (2 * n_ಎಳೆಗಳು) - 1) / (2 * n_ಎಳೆಗಳು), 32);
            ಸ್ಥಿರಾಂಕ uint32_t chunk_ಗಾತ್ರ = (std::min)(max_chunk_elems, (std::max)(ಗುರಿ_chunk_elems, min_chunk_elems));
            ಸ್ಥಿರಾಂಕ uint32_t chunk_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * elem_ಗಾತ್ರ, 128);

            kparams->kernel_type           = HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED;
            kparams->n_ಎಳೆಗಳು             = n_ಎಳೆಗಳು;
            kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್       = 1;
            kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = is_scalar_ಪ್ರಸಾರ ? 0 : src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
            kparams->src1_ಗಾತ್ರ             = 0;
            kparams->chunk_ಗಾತ್ರ            = chunk_ಗಾತ್ರ;
            kparams->chunk_ಬೈಟುಗಳು           = chunk_ಬೈಟುಗಳು;
            kparams->is_scalar             = is_scalar_ಪ್ರಸಾರ ? 1 : 0;

            htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_ಗಾತ್ರ);
            ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು == 0 || L.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ = L.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್;
    kparams->vtcm_ಗಾತ್ರ       = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_unary_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    uint32_t op,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t)sess->n_ಎಳೆಗಳು, src0_nrows);

    kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);

    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst->ne[0]  * ggml_type_ಗಾತ್ರ(dst->type);

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,  128);

    kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    size_t src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
    size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = 0;
    ತಾರ್ಕಿಕ ಪ್ರಸಾರ_ತೂಕ = ಸುಳ್ಳು;

    ಆದರೆ (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != ಶೂನ್ಯಸೂಚಕ);
        src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1->ne[0] * ggml_type_ಗಾತ್ರ(src1->type);
        src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
        ಪ್ರಸಾರ_ತೂಕ = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->ಪ್ರಸಾರ_ತೂಕ      = ಪ್ರಸಾರ_ತೂಕ;

    ರಚನೆ htp_unary_vtcm_layout L;
    uint32_t ಲಂಬಸಾಲು_tile = 0;
    uint32_t vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                ಪ್ರಸಾರ_ತೂಕ, n_ಎಳೆಗಳು, sess->vtcm_ಗಾತ್ರ, elem_ಗಾತ್ರ,
                                &ಲಂಬಸಾಲು_tile, &vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread);

    kparams->ಲಂಬಸಾಲು_tile = ಲಂಬಸಾಲು_tile;
    kparams->vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
    kparams->vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src0_ಬೈಟುಗಳು;
    kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src1_ಬೈಟುಗಳು;
    kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.dst_ಬೈಟುಗಳು;

    kparams->vtcm_src0_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;
    kparams->vtcm_src1_ಗಾತ್ರ = L.src1_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;
    kparams->vtcm_dst_ಗಾತ್ರ  = L.dst_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;

    kparams->ಖಂಡ = ಲಂಬಸಾಲು_tile ? 0 : ((L.src0_ಬೈಟುಗಳು / 2) / src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ uint32_t tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ಲಂಬಸಾಲು_tile > 0 ? (src0->ne[0] + ಲಂಬಸಾಲು_tile - 1) / ಲಂಬಸಾಲು_tile : 1;
    kparams->div_ne01  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_ಮೌಲ್ಯಗಳು(tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_get_ಅಡ್ಡಸಾಲುಗಳು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t nr = ne10 * ne11 * ne12;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    ಆದರೆ (tiled) {
        kparams->kernel_type = HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_TILED;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE;
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->kernel_type = HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_FLAT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_tasks = nr;

    kparams->n_ಎಳೆಗಳು = (std::min)((uint32_t)sess->n_ಎಳೆಗಳು, ಒಟ್ಟು_tasks);

    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout vtcm_layout = {};
    ಆಗ (kparams->n_ಎಳೆಗಳು > 0) {
        htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ಒಟ್ಟು_tasks, src1->type, kparams->n_ಎಳೆಗಳು);
        ಆದರೆ (vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು <= sess->vtcm_ಗಾತ್ರ) {
            ನಿಲ್ಲಿಸು;
        }
        --kparams->n_ಎಳೆಗಳು;
    }

    ಆದರೆ (kparams->n_ಎಳೆಗಳು == 0 && ಒಟ್ಟು_tasks > 0) {
        htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ಒಟ್ಟು_tasks, src1->type, 1);
    }

    kparams->vtcm_ಗಾತ್ರ = (ಒಟ್ಟು_tasks == 0) ? 0 : vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    kparams->tasks_ಪ್ರತಿ_thread = kparams->n_ಎಳೆಗಳು > 0 ? (ಒಟ್ಟು_tasks + kparams->n_ಎಳೆಗಳು - 1) / kparams->n_ಎಳೆಗಳು : 0;
    kparams->ಒಟ್ಟು_tasks = ಒಟ್ಟು_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne10) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne10 * ne11) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne02) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne03) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0, //ಮೌಲ್ಯಗಳು
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1, // indices
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,  // destination
    ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ uint32_t nr = src0->ne[1];

    kparams->n_ಎಳೆಗಳು = (std::min)((uint32_t)sess->n_ಎಳೆಗಳು, nr);
    kparams->tasks_ಪ್ರತಿ_thread = (nr + kparams->n_ಎಳೆಗಳು - 1) / kparams->n_ಎಳೆಗಳು;
    kparams->ಒಟ್ಟು_tasks = nr;

    kparams->div_ne11 = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[2]);
    kparams->div_tasks_ಪ್ರತಿ_thread = init_fastdiv_ಮೌಲ್ಯಗಳು(kparams->tasks_ಪ್ರತಿ_thread);
    kparams->div_ne02 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);

    ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout vtcm_layout;
    htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_ಎಳೆಗಳು);
    kparams->vtcm_ಗಾತ್ರ = vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_softmax_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, src0_nrows);

    ದಶಮಾಂಶ scale = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale,    &op->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, &op->op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));
    kparams->n_ತಲೆ      = n_ತಲೆ;
    kparams->n_ತಲೆ_log2 = n_ತಲೆ_log2;

    ಆದರೆ (max_bias > 0.0f && n_ತಲೆ_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != ಶೂನ್ಯಸೂಚಕ) ? 1 : 0;
    kparams->use_f16  = (src1 != ಶೂನ್ಯಸೂಚಕ && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ರಚನೆ htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_ಎಳೆಗಳು);

    kparams->n_ಎಳೆಗಳು                 = n_ಎಳೆಗಳು;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_ಪ್ರತಿ_thread     = (src0_nrows + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
    kparams->vtcm_ಗಾತ್ರ                 = (uint32_t) layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = (uint32_t) layout.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = (uint32_t) layout.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = (uint32_t) layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned     = (uint32_t) layout.src0_spad_half_ಗಾತ್ರ;
    kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned     = (uint32_t) layout.src1_spad_half_ಗಾತ್ರ;
    kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned      = (uint32_t) layout.dst_spad_half_ಗಾತ್ರ;
    kparams->src0_spad_half_ಗಾತ್ರ       = (uint32_t) layout.src0_spad_half_ಗಾತ್ರ;
    kparams->src1_spad_half_ಗಾತ್ರ       = (uint32_t) layout.src1_spad_half_ಗಾತ್ರ;
    kparams->dst_spad_half_ಗಾತ್ರ        = (uint32_t) layout.dst_spad_half_ಗಾತ್ರ;
    ಆದರೆ (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F16;
    } ಇಲ್ಲದಿದ್ದರೆ {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F32;
    }

    ಆದರೆ (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1]);
    ಆದರೆ (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1 ? src1->ne[3] : 1;
    ಆದರೆ (ne12 > 0) kparams->div_ne12 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12);
    ಆದರೆ (ne13 > 0) kparams->div_ne13 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne13);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_rope_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, src0_nrows);
    ಸ್ಥಿರಾಂಕ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ರಚನೆ htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_ಎಳೆಗಳು, n_freq_factors);

    kparams->n_ಎಳೆಗಳು              = n_ಎಳೆಗಳು;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_ಪ್ರತಿ_thread  = (src0_nrows + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
    kparams->vtcm_ಗಾತ್ರ              = (uint32_t) layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    kparams->spad_ಪ್ರತಿ_thread        = (uint32_t) layout.ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    kparams->theta_cache_ಸ್ಥಾನಾಂತರ     = (uint32_t) layout.theta_cache_ಗಾತ್ರ_aligned;
    kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = (uint32_t) layout.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    kparams->freq_factors_ಸ್ಥಾನಾಂತರ    = (uint32_t) (layout.ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    kparams->freq_factors_ಗಾತ್ರ      = (uint32_t) layout.freq_factors_ಗಾತ್ರ_aligned;

    ಆದರೆ (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[1]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ssm_conv_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ uint32_t d_conv  = (uint32_t) src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t d_inner = (uint32_t) src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_t     = (uint32_t) dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_s     = (uint32_t) dst->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ncs     = (uint32_t) src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, (d_inner + 31) / 32);

    kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ಸ್ಥಿರಾಂಕ uint32_t raw_rpt = (d_inner + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(raw_rpt, 32);

    ಸ್ಥಿರಾಂಕ uint32_t src1_raw_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_ಪ್ರತಿ_thread * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ), 128) + 128;
    ಸ್ಥಿರಾಂಕ uint32_t src1_T_ಬೈಟುಗಳು   = hex_ಸುತ್ತು_ಮೇಲೆ(d_conv * d_inner_ಪ್ರತಿ_thread * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    ಸ್ಥಿರಾಂಕ uint32_t vtcm_src1_ಪ್ರತಿ_thread = src1_raw_ಬೈಟುಗಳು + src1_T_ಬೈಟುಗಳು;

    uint32_t vtcm_src0_ಪ್ರತಿ_thread = 0;
    uint32_t vtcm_dst_ಪ್ರತಿ_thread  = 0;

    ಆದರೆ (n_t == 1) {
        kparams->d_inner_tile = d_inner_ಪ್ರತಿ_thread;

        ಸ್ಥಿರಾಂಕ uint32_t src0_tile_raw_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_ಪ್ರತಿ_thread * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        ಸ್ಥಿರಾಂಕ uint32_t src0_T_ಬೈಟುಗಳು        = hex_ಸುತ್ತು_ಮೇಲೆ(d_conv * d_inner_ಪ್ರತಿ_thread * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        vtcm_src0_ಪ್ರತಿ_thread = 2 * src0_tile_raw_ಬೈಟುಗಳು + src0_T_ಬೈಟುಗಳು;

        ಸ್ಥಿರಾಂಕ uint32_t dst_tile_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_ಪ್ರತಿ_thread * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        vtcm_dst_ಪ್ರತಿ_thread = 2 * dst_tile_ಬೈಟುಗಳು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ size_t vtcm_budget = (sess->vtcm_ಗಾತ್ರ > 0 ? sess->vtcm_ಗಾತ್ರ / n_ಎಳೆಗಳು : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ಸ್ಥಿರಾಂಕ uint32_t src0_ಖಂಡ_T    = hex_ಸುತ್ತು_ಮೇಲೆ(ncs * 32 * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        ಸ್ಥಿರಾಂಕ size_t   fixed_ಬೈಟುಗಳು     = vtcm_src1_ಪ್ರತಿ_thread + src0_ಖಂಡ_T;
        ಸ್ಥಿರಾಂಕ size_t   avail_for_tiles = vtcm_budget > fixed_ಬೈಟುಗಳು ? vtcm_budget - fixed_ಬೈಟುಗಳು : (128 * 1024);

        uint32_t ಗುರಿ_max_tile = hex_ಸುತ್ತು_ಮೇಲೆ((d_inner_ಪ್ರತಿ_thread + 3) / 4, 32);
        ಗುರಿ_max_tile = (std::max)(ಗುರಿ_max_tile, 32u);
        ಗುರಿ_max_tile = (std::min)(ಗುರಿ_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * ಗಾತ್ರ(ದಶಮಾಂಶ)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        ಆದರೆ (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        ಆದರೆ (d_inner_tile > ಗುರಿ_max_tile) {
            d_inner_tile = ಗುರಿ_max_tile;
        }
        ಆದರೆ (d_inner_tile > d_inner_ಪ್ರತಿ_thread) {
            d_inner_tile = d_inner_ಪ್ರತಿ_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ಸ್ಥಿರಾಂಕ uint32_t src0_tile_raw = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * ncs * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        vtcm_src0_ಪ್ರತಿ_thread = 2 * src0_tile_raw + src0_ಖಂಡ_T;

        vtcm_dst_ಪ್ರತಿ_thread = 2 * hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * n_t * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    }

    kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = vtcm_src0_ಪ್ರತಿ_thread;
    kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = vtcm_src1_ಪ್ರತಿ_thread;
    kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = vtcm_dst_ಪ್ರತಿ_thread;

    kparams->vtcm_src0_ಗಾತ್ರ = vtcm_src0_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    kparams->vtcm_src1_ಗಾತ್ರ = vtcm_src1_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    kparams->vtcm_dst_ಗಾತ್ರ  = vtcm_dst_ಪ್ರತಿ_thread  * n_ಎಳೆಗಳು;
    kparams->vtcm_ಗಾತ್ರ      = kparams->vtcm_src0_ಗಾತ್ರ + kparams->vtcm_src1_ಗಾತ್ರ + kparams->vtcm_dst_ಗಾತ್ರ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_gated_delta_net_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * q     = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * k     = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * v     = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * g     = op->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * state = op->src[5];

    ಸ್ಥಿರಾಂಕ uint32_t S_v      = (uint32_t) v->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t H        = (uint32_t) v->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t) v->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = (uint32_t) v->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ಸ್ಥಿರಾಂಕ uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = H * n_seqs;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_ಟೋಕನುಗಳು >= HTP_GDN_MIN_ಟೋಕನುಗಳು) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ರಚನೆ htp_gdn_hmx_vtcm_layout hmx_layout;
    ರಚನೆ htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    ಆದರೆ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, sess->vtcm_ಗಾತ್ರ, n_ಎಳೆಗಳು, ನಿಜ, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_ಗಾತ್ರ      = HTP_GDN_CHUNK_ಗಾತ್ರ;
        kparams->n_chunks        = (n_ಟೋಕನುಗಳು + HTP_GDN_CHUNK_ಗಾತ್ರ - 1) / HTP_GDN_CHUNK_ಗಾತ್ರ;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_ಗಾತ್ರ       = (uint32_t) hmx_layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_ಬೈಟುಗಳು;
        kparams->vtcm_ಪ್ರತಿ_thread = (uint32_t) (hmx_layout.ಒಟ್ಟು_ಬೈಟುಗಳು / (n_ಎಳೆಗಳು > 0 ? n_ಎಳೆಗಳು : 1));
    } ಇಲ್ಲದಿದ್ದರೆ {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_ಎಳೆಗಳು);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_ಪ್ರತಿ_thread = (uint32_t) hvx_layout.ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
        kparams->vtcm_ಗಾತ್ರ       = (uint32_t) hvx_layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    }

    kparams->n_ಎಳೆಗಳು           = n_ಎಳೆಗಳು;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_ಟೋಕನುಗಳು            = n_ಟೋಕನುಗಳು;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು          = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread     = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + kparams->n_ಎಳೆಗಳು - 1) / kparams->n_ಎಳೆಗಳು;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    kparams->state_ಗಾತ್ರ_ಪ್ರತಿ_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ದಶಮಾಂಶ) S_v);

    ಆದರೆ (H > 0)                  kparams->div_H         = init_fastdiv_ಮೌಲ್ಯಗಳು(H);
    ಆದರೆ (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) q->ne[1]);
    ಆದರೆ (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) k->ne[1]);
    ಆದರೆ (rq3 > 0)                kparams->div_rq3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rq3);
    ಆದರೆ (rk3 > 0)                kparams->div_rk3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rk3);
    ಆದರೆ (kparams->n_ಎಳೆಗಳು > 0) kparams->div_n_ಎಳೆಗಳು = init_fastdiv_ಮೌಲ್ಯಗಳು(kparams->n_ಎಳೆಗಳು);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ವಿಂಗಡಿಸು_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ತಾರ್ಕಿಕ is_ತುದಿ_k,
    ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t ne00       = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t k          = dst->ne[0];

    int32_t ಕ್ರಮ = GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC;
    ಆದರೆ (!is_ತುದಿ_k) {
        ಕ್ರಮ = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು_max = sess->n_ಎಳೆಗಳು > 0 ? sess->n_ಎಳೆಗಳು : 4;
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget = sess->vtcm_ಗಾತ್ರ > 0 ? sess->vtcm_ಗಾತ್ರ : (8 * 1024 * 1024);

    ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout layout;
    ತಾರ್ಕಿಕ ok = htp_ವಿಂಗಡಿಸು_solve_layout(&layout, ne00, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, k, n_ಎಳೆಗಳು_max, vtcm_budget, is_ತುದಿ_k);
    GGML_ASSERT(ok);

    kparams->n_ಎಳೆಗಳು         = (int32_t) layout.n_ಎಳೆಗಳು;
    kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು        = (int32_t) ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    kparams->ಅಡ್ಡಸಾಲು_ಆರಂಭ         = 0;
    kparams->ಅಡ್ಡಸಾಲು_ಕೊನೆ           = (int32_t) ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->ಕ್ರಮ             = ಕ್ರಮ;
    kparams->is_ತುದಿ_k          = is_ತುದಿ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_ಗಾತ್ರ         = (int32_t) layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ  = (int32_t) layout.phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off  = (int32_t) layout.ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off;
    kparams->ವಿಲೀನ_indices_off = (int32_t) layout.ವಿಲೀನ_indices_off;
    kparams->ವಿಲೀನ_elems       = (int32_t) layout.ವಿಲೀನ_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ಸ್ಥಿರಾಂಕ uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ಸ್ಥಿರಾಂಕ int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    ಆದರೆ (numer_hi < 0) {
        *ox_hi = 0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    ಆದರೆ (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_pool_shape_ಹೊಂದುತ್ತದೆ(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst
) {
    ಹಿಂತಿರುಗಿ ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_pool_2d_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    ರಚನೆ htp_pool_2d_kernel_params * kparams,
    ತಾರ್ಕಿಕ is_pool_1d
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));

    ಸ್ಥಿರಾಂಕ uint32_t src_x = (uint32_t) src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t dst_x = (uint32_t) dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t src_plane_ಬೈಟುಗಳು = src_x * src_y * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t dst_plane_ಬೈಟುಗಳು = dst_x * dst_y * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, planes > 0 ? planes : 1);

    ರಚನೆ htp_pool_vtcm_layout layout;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_ಎಳೆಗಳು, sess->vtcm_ಗಾತ್ರ);
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
    kparams->src_plane_ಬೈಟುಗಳು = src_plane_ಬೈಟುಗಳು;
    kparams->dst_plane_ಬೈಟುಗಳು = dst_plane_ಬೈಟುಗಳು;
    kparams->src_plane_ಬೈಟುಗಳು_aligned = (uint32_t) layout.src_spad_half_ಗಾತ್ರ;
    kparams->dst_plane_ಬೈಟುಗಳು_aligned = (uint32_t) layout.dst_spad_half_ಗಾತ್ರ;
    kparams->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_ಮಾರ್ಗ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_ಮಾರ್ಗ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->ಖಂಡ_ಮಾರ್ಗ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_ಎಣಿಕೆ = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_ಮಾರ್ಗ = (kparams->fast_ಮಾರ್ಗ && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ದಶಮಾಂಶ) (kparams->kernel_x * kparams->kernel_y);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_concat_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!src0 || !src1 || !dst) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಪೂರ್ಣಾಂಕ dim = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
    ಆದರೆ (dim < 0 || dim >= GGML_MAX_DIMS) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    kparams->dim = dim;

    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->type != dst->type || src1->type != dst->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < GGML_MAX_DIMS; d++) {
        ಸ್ಥಿರಾಂಕ int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        ಆದರೆ (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ uint32_t max_gather_ne0 = 128 / type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ b_dense = (src1->nb[0] == type_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_ಗಾತ್ರ &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_ಗಾತ್ರ
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_ಗಾತ್ರ);

    ಆದರೆ (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_ಗಾತ್ರ && dst->nb[1] == (size_t) dst->ne[0] * type_ಗಾತ್ರ &&
        src0->nb[0] == type_ಗಾತ್ರ &&
        (src1->nb[0] == type_ಗಾತ್ರ || src1->nb[1] == type_ಗಾತ್ರ) &&
        gather_strides_ok &&
        htp_gather_ಅಡ್ಡಸಾಲುಗಳು_tensor_outer_contiguous(dst) &&
        htp_gather_ಅಡ್ಡಸಾಲುಗಳು_tensor_outer_contiguous(src0) &&
        htp_gather_ಅಡ್ಡಸಾಲುಗಳು_tensor_outer_contiguous(src1)) {

        ಆದರೆ (htp_gather_ಅಡ್ಡಸಾಲುಗಳು_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
                                         sess->n_ಎಳೆಗಳು, sess->mdev.count, sess->vtcm_ಗಾತ್ರ)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು;
            kparams->n_ಎಳೆಗಳು   = (uint8_t) sess->n_ಎಳೆಗಳು;
            kparams->vtcm_ಗಾತ್ರ   = kparams->u.gather.ತುಂಡು_2 * sess->n_ಎಳೆಗಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dma_strides_ok = (src0->nb[0] == type_ಗಾತ್ರ && src1->nb[0] == type_ಗಾತ್ರ && dst->nb[0] == type_ಗಾತ್ರ);

    ಆದರೆ (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_ಎಳೆಗಳು   = 1;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ transposed_ಅಡ್ಡಸಾಲುಗಳು_ok = (src0->nb[0] == type_ಗಾತ್ರ && src1->nb[1] == type_ಗಾತ್ರ && dst->nb[0] == type_ಗಾತ್ರ);

    ಆದರೆ (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_ಅಡ್ಡಸಾಲುಗಳು_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು > 0 ? (uint32_t) sess->n_ಎಳೆಗಳು : 8;
        ರಚನೆ htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_ಗಾತ್ರ, n_ಎಳೆಗಳು);

        ಆದರೆ (sess->vtcm_ಗಾತ್ರ > 0 && layout.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_ಎಳೆಗಳು                         = n_ಎಳೆಗಳು;
        kparams->vtcm_ಗಾತ್ರ                         = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
        kparams->u.transposed.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread = layout.src0_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        kparams->u.transposed.spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread = layout.src1_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_cpy_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!src0 || !dst) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t nelem_src = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ int64_t nelem_dst = ggml_nelements(dst);
    ಆದರೆ (nelem_src != nelem_dst || nelem_src < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    kparams->src0_type_ಗಾತ್ರ = (uint8_t) src_type_ಗಾತ್ರ;
    kparams->dst_type_ಗಾತ್ರ  = (uint8_t) dst_type_ಗಾತ್ರ;
    kparams->ಒಟ್ಟು_elems    = (uint32_t) nelem_src;

    ಆದರೆ (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (nelem_src == 1) {
        ಆದರೆ (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sametype = (src0->type == dst->type);

    ತಾರ್ಕಿಕ same_extents = ನಿಜ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < GGML_MAX_DIMS; d++) {
        ಆದರೆ (src0->ne[d] != dst->ne[d]) {
            same_extents = ಸುಳ್ಳು;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_ಗಾತ್ರ) || (dst->nb[0] != dst_type_ಗಾತ್ರ) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_ಗಾತ್ರ) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sameshape  = same_extents && !transposed;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src_is_contiguous = ggml_is_contiguous(src0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dst_is_contiguous = ggml_is_contiguous(dst);

    ಆದರೆ (sametype) {
        ಆದರೆ (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಸ್ಥಿರಾಂಕ uint32_t max_gather_ne0 = 128 / src_type_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ಆದರೆ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_ಗಾತ್ರ && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_ಗಾತ್ರ &&
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು > 1 &&
            htp_gather_ಅಡ್ಡಸಾಲುಗಳು_tensor_outer_contiguous(src0)) {

            ಆದರೆ (htp_gather_ಅಡ್ಡಸಾಲುಗಳು_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, ಸುಳ್ಳು,
                                             src_type_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
                                             sess->n_ಎಳೆಗಳು, sess->mdev.count, sess->vtcm_ಗಾತ್ರ)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು;
                kparams->n_ಎಳೆಗಳು   = (uint8_t) sess->n_ಎಳೆಗಳು;
                kparams->vtcm_ಗಾತ್ರ   = kparams->u.gather.ತುಂಡು_2 * sess->n_ಎಳೆಗಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }

        ಆದರೆ (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_ಎಳೆಗಳು   = sess->n_ಎಳೆಗಳು > 0 ? (uint8_t) sess->n_ಎಳೆಗಳು : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (!sameshape) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    ಆದರೆ (!valid_conversion) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು > 0 ? (uint32_t) sess->n_ಎಳೆಗಳು : 4;
    ರಚನೆ htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_ಗಾತ್ರ, dst_type_ಗಾತ್ರ, n_ಎಳೆಗಳು);

    ಆದರೆ (sess->vtcm_ಗಾತ್ರ > 0 && layout.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_ಎಳೆಗಳು               = (uint8_t) n_ಎಳೆಗಳು;
    kparams->vtcm_ಗಾತ್ರ               = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
    kparams->u.convert.src0_buf_ಗಾತ್ರ = layout.src0_buf_ಗಾತ್ರ;
    kparams->u.convert.dst_buf_ಗಾತ್ರ  = layout.dst_buf_ಗಾತ್ರ;
    kparams->u.convert.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread = layout.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    kparams->u.convert.spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread = layout.spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    kparams->u.convert.src0_ಅಡ್ಡಸಾಲು_stride = layout.src0_ಅಡ್ಡಸಾಲು_stride;
    kparams->u.convert.dst_ಅಡ್ಡಸಾಲು_stride  = layout.dst_ಅಡ್ಡಸಾಲು_stride;
    kparams->u.convert.blk_ಅಡ್ಡಸಾಲುಗಳು        = layout.blk_ಅಡ್ಡಸಾಲುಗಳು;
    kparams->u.convert.div_ne01      = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[2] * src0->ne[1]));

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmnx_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0, // W0
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * act,  // x
    int32_t n_weights,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->n_ಎಳೆಗಳು = sess->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10 = act->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11 = act->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12 = act->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne13 = act->ne[3];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_padded = is_tiled ? hex_ಸುತ್ತು_ಮೇಲೆ(ne00, 32) : ne00;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11_padded = hex_ಸುತ್ತು_ಮೇಲೆ(ne11, 32);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01_tiled  = hex_ಸುತ್ತು_ಮೇಲೆ(ne01, 32);

    ಸ್ಥಿರಾಂಕ size_t vtcm_budget = sess->vtcm_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ತಾರ್ಕಿಕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ಆದರೆ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, ಶೂನ್ಯಸೂಚಕ, ಸುಳ್ಳು, is_batched)) {
        ಆದರೆ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, ಶೂನ್ಯಸೂಚಕ, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, ಸುಳ್ಳು, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            ಹೋಗು finalize;
        }
    }

    ಆದರೆ (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಹಿಂತಿರುಗಿ;
    }

    {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ act_nrows = ne11 * ne12 * ne13;
        ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
        ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->nb[1];

        ಸ್ಥಿರಾಂಕ uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t ಉತ್ತಮ_n_prefetch = 2;
        ಗಾಗಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ರಚನೆ htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು, wtype, ne10, act_nrows, sess->n_ಎಳೆಗಳು,
                0, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, d, ಸುಳ್ಳು, ನಿಜ
            );
            ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= sess->vtcm_ಗಾತ್ರ) {
                ಉತ್ತಮ_n_prefetch = d;
                ನಿಲ್ಲಿಸು;
            }
        }

        ರಚನೆ htp_mm_hvx_vtcm_layout L;
        ತಾರ್ಕಿಕ try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು, wtype, ne10, act_nrows, sess->n_ಎಳೆಗಳು,
            0, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, ಉತ್ತಮ_n_prefetch, ಸುಳ್ಳು, ನಿಜ
        );

        ಆದರೆ (try_tiled && L.ಒಟ್ಟು_ಬೈಟುಗಳು <= sess->vtcm_ಗಾತ್ರ) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು;
            kparams->tile_ಗಾತ್ರ         = htp_tensor_get_tile_ಗಾತ್ರ(wtype);
            kparams->aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype);
            kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ      = act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            kparams->vtcm_src0_ಗಾತ್ರ    = L.src0_ಬೈಟುಗಳು;
            kparams->vtcm_act_ಗಾತ್ರ     = L.act_ಬೈಟುಗಳು;
            kparams->vtcm_bias_ಗಾತ್ರ    = 0;
            kparams->vtcm_dst_ಗಾತ್ರ     = L.dst_ಬೈಟುಗಳು;
            kparams->vtcm_ಗಾತ್ರ         = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->n_prefetch        = ಉತ್ತಮ_n_prefetch;
            kparams->n_weights         = n_weights;
        } ಇಲ್ಲದಿದ್ದರೆ {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ಹಿಂತಿರುಗಿ;
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
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0, // W0
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * act,  // x
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ರಚನೆ htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_op_mul_mat_use_fwht(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst) {
    ಆದರೆ (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ int64_t n = src1->ne[0];
    ಆದರೆ (n < 32 || (n & (n - 1)) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_mul_mat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    ಆದರೆ (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ಆದರೆ (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ರಚನೆ htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, ಶೂನ್ಯಸೂಚಕ, dst, &ukparams);
        ಹಿಂತಿರುಗಿ ukparams.vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread > 0 && (size_t) ukparams.vtcm_ಗಾತ್ರ <= sess->vtcm_ಗಾತ್ರ;
    }

    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K: {
            ಆದರೆ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ int64_t qk = ggml_blck_ಗಾತ್ರ(src0->type);
            ಆದರೆ (src0->ne[0] % qk) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        }

        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಆದರೆ (src0->nb[1] < src0->nb[0]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಆದರೆ (src1->type != GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src0->nb[1] < src0->nb[0]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಆದರೆ (src0->nb[1] < src0->nb[0]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ಆದರೆ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_mul_mat_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಆದರೆ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (src0->ne[0] % ggml_blck_ಗಾತ್ರ(src0->type) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಆದರೆ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (src0->ne[0] % 32 != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ಆದರೆ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_ಗಾತ್ರ, sess->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ದ್ವಿಮಾನ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type == GGML_TYPE_F32) {
        ಆದರೆ (src1->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (dst->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
        ಆದರೆ (src1->type != GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (dst->type != GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_are_same_shape(src0, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess, op_remap_ಗಮ್ಯ_htp(op), src0, src1, dst, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ಕೂಡಿಸು_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!src2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_are_same_shape(src0, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src2->nb[0] != ಗಾತ್ರ(int32_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // REVISIT: add support for non-contiguous tensors
    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src2->nb[1] < src2->ne[0] * ಗಾತ್ರ(int32_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess, HTP_OP_ಕೂಡಿಸು_ID, src0, src1, dst, &kparams, src2);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_unary(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != src0->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    ಆದರೆ (src0->type == GGML_TYPE_F16) {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_NORM:
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ಸ್ಥಿತಿ GGML_OP_SCALE:
            ಸ್ಥಿತಿ GGML_OP_CLAMP:
            ಸ್ಥಿತಿ GGML_OP_SQR:
            ಸ್ಥಿತಿ GGML_OP_SQRT:
            ಸ್ಥಿತಿ GGML_OP_LOG:
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_UNARY:
                ಆದರೆ (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_ಹೆಜ್ಜೆ) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (!ggml_are_same_shape(src0, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // dst must be contiguous; src0 may be non-contiguous
    ಆದರೆ (!ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ಮೊತ್ತ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // TODO: add support for non-contigiuos tensors
    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_argmax(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pool_2d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ int32_t * params = op->op_params;

    ಆದರೆ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_hexagon_pool_shape_ಹೊಂದುತ್ತದೆ(src0, op)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t kernel_x = params[1];
    ಸ್ಥಿರಾಂಕ int32_t kernel_y = params[2];
    ಸ್ಥಿರಾಂಕ int32_t stride_x = params[3];
    ಸ್ಥಿರಾಂಕ int32_t stride_y = params[4];
    ಸ್ಥಿರಾಂಕ int32_t pad_x    = params[5];
    ಸ್ಥಿರಾಂಕ int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    ಆದರೆ (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, planes > 0 ? planes : 1);

    ರಚನೆ htp_pool_vtcm_layout layout;
    ಹಿಂತಿರುಗಿ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_ಎಳೆಗಳು, sess->vtcm_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pool_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ int32_t * params = op->op_params;

    ಆದರೆ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_hexagon_pool_shape_ಹೊಂದುತ್ತದೆ(src0, op)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t kernel = params[1];
    ಸ್ಥಿರಾಂಕ int32_t stride = params[2];
    ಸ್ಥಿರಾಂಕ int32_t pad    = params[3];
    ಆದರೆ (kernel <= 0 || stride <= 0 || pad < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, planes > 0 ? planes : 1);

    ರಚನೆ htp_pool_vtcm_layout layout;
    ಹಿಂತಿರುಗಿ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_ಎಳೆಗಳು, sess->vtcm_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_activations(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != src0->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous_1(src0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1) {
        ಆದರೆ (src1->type != src0->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_are_same_shape(src0, src1)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_is_contiguous_1(src1)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_softmax(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;  // FIXME: add support for sinks
    }

    ಆದರೆ (src0->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->ne[2] > 512) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1) {
        ಆದರೆ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src0->ne[0] != src1->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src1->ne[1] < src0->ne[1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src0->ne[2] % src1->ne[2] != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src0->ne[3] % src1->ne[3] != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (src1) {
        ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ಸ್ಥಿರಾಂಕ int64_t ne0 = src0->ne[0];
    ಆದರೆ (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // HVX vector size constraints for softmax
    #ವ್ಯಾಖ್ಯಾನಿಸು SOFTMAX_MAX_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    ಆದರೆ (ne0 > SOFTMAX_MAX_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು  = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, src0_nrows);
    ರಚನೆ htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != ಶೂನ್ಯಸೂಚಕ, src1 && src1->type == GGML_TYPE_F16, n_ಎಳೆಗಳು);
    ಆದರೆ (layout.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1]; // indices
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    ಆದರೆ (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_get_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1]; // indices
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src1) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->type == dst->type) {
        ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(op)) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params kparams;
    ggml_hexagon_precompute_get_ಅಡ್ಡಸಾಲುಗಳು_params(sess, src0, src1, dst, &kparams);
    ಆದರೆ (kparams.n_ಎಳೆಗಳು == 0 || (size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_argsort(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;         // indices

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು_max = sess->n_ಎಳೆಗಳು > 0 ? sess->n_ಎಳೆಗಳು : 4;
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget = sess->vtcm_ಗಾತ್ರ > 0 ? sess->vtcm_ಗಾತ್ರ : (8 * 1024 * 1024);

    ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout layout;
    ಆದರೆ (!htp_ವಿಂಗಡಿಸು_solve_layout(&layout, src0->ne[0], ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, dst->ne[0], n_ಎಳೆಗಳು_max, vtcm_budget, ಸುಳ್ಳು)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ತುದಿ_k(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;         // indices

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು_max = sess->n_ಎಳೆಗಳು > 0 ? sess->n_ಎಳೆಗಳು : 4;
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget = sess->vtcm_ಗಾತ್ರ > 0 ? sess->vtcm_ಗಾತ್ರ : (8 * 1024 * 1024);

    ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout layout;
    ಆದರೆ (!htp_ವಿಂಗಡಿಸು_solve_layout(&layout, src0->ne[0], ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, dst->ne[0], n_ಎಳೆಗಳು_max, vtcm_budget, ನಿಜ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_rope(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src2 = op->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!ggml_are_same_shape(src0, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->ne[0] <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಆದರೆ (src0_nrows == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t * op_params = &op->op_params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims = op_params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode   = op_params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    ಆದರೆ (n_dims < 0 || n_dims % 2 != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    ಆದರೆ (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ freq_ತಳ;
    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (freq_ತಳ < 0.0f) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (mode != GGML_ROPE_TYPE_ಸಾಮಾನ್ಯ &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    ಆದರೆ (mode == GGML_ROPE_TYPE_VISION) {
        ಆದರೆ (n_dims != (ಪೂರ್ಣಾಂಕ) (src0->ne[0] / 2) || n_offs != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (is_mrope) {
        ಸ್ಥಿರಾಂಕ int32_t * sections = op_params + 11;
        ಆದರೆ (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    ಆದರೆ (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src2) {
        ಆದರೆ (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src2->ne[0] < (n_dims / 2)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->nb[1] < src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[1] < dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (std::min)((uint32_t) sess->n_ಎಳೆಗಳು, src0_nrows);
    ಸ್ಥಿರಾಂಕ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ರಚನೆ htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_ಎಳೆಗಳು, n_freq_factors);
    ಆದರೆ (layout.ಒಟ್ಟು_ಬೈಟುಗಳು > sess->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ssm_conv(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    // Only support FP32 for now
    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Check IO tensor shapes and dims
    ಆದರೆ (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; // src0 should be effectively 3D
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_conv  = src1->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_inner = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_t = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_s = dst->ne[2];

    ಆದರೆ (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src0->nb[1] != src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) || src1->nb[1] != src1->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    ಆದರೆ ((size_t) kparams.vtcm_ಗಾತ್ರ > sess->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ im2col_3d_calc_output_ಗಾತ್ರ(uint32_t ಒಳಹರಿವು, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    ಆದರೆ (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t numerator = (int64_t) ಒಳಹರಿವು + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    ಆದರೆ (numerator < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint64_t size = (uint64_t) (numerator / stride) + 1;
    ಆದರೆ (size == 0 || size > UINT32_MAX) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    *output = (uint32_t) size;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->kernel_type = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_UNSUPPORTED;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!src0 || !src1 || !dst) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t * op_params = (ಸ್ಥಿರಾಂಕ int32_t *) op->op_params;
    ಸ್ಥಿರಾಂಕ int32_t s0 = op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = op_params[5];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = (op_params[6] == 1);

    ಆದರೆ (s0 <= 0 || d0 <= 0 || p0 < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t IW = (uint32_t) src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t KW = (uint32_t) src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t OW = (uint32_t) dst->ne[1];

    ಆದರೆ (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64   = (uint64_t) N * OH;
    ಆದರೆ (patch_stride_64 > UINT32_MAX || ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64 > UINT32_MAX) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು   = (uint32_t) ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64;

    ಆದರೆ (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (is_2D && dst->ne[2] != OH) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ಗಾತ್ರ(ggml_fp16_t) : ಗಾತ್ರ(ದಶಮಾಂಶ);

    ತಾರ್ಕಿಕ is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    ಆದರೆ (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ಸ್ಥಿರಾಂಕ uint32_t max_ಎಳೆಗಳು = (sess && sess->n_ಎಳೆಗಳು > 0) ? (uint32_t) sess->n_ಎಳೆಗಳು : 8;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು < max_ಎಳೆಗಳು) ? ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : max_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t vtcm_ಗಾತ್ರ   = (sess && sess->vtcm_ಗಾತ್ರ > 0) ? sess->vtcm_ಗಾತ್ರ : 0;

    ಆದರೆ (is_patchembed) {
        ಸ್ಥಿರಾಂಕ uint64_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) IC * KH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ uint64_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) OW * patch_stride * dst_elem;
        ಆದರೆ (src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw <= UINT32_MAX && dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw <= UINT32_MAX) {
            ಸ್ಥಿರಾಂಕ uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);
            ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);

            ರಚನೆ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಎಳೆಗಳು);

            ಆದರೆ (vtcm_ಗಾತ್ರ == 0 || layout.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_ಗಾತ್ರ) {
                kparams->kernel_type          = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_ಎಳೆಗಳು            = (uint8_t) n_ಎಳೆಗಳು;
                kparams->vtcm_ಗಾತ್ರ            = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
                kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
                kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
                kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
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
                kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು           = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }

    ಗಾಗಿ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ಸ್ಥಿರಾಂಕ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ಆದರೆ (Wb > UINT32_MAX) {
            ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ uint64_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) IC * KH * Wb * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ uint64_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) owb * patch_stride * dst_elem;
        ಆದರೆ (src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw > UINT32_MAX || dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw > UINT32_MAX) {
            ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);
        ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);

        ರಚನೆ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಎಳೆಗಳು);

        ಆದರೆ (vtcm_ಗಾತ್ರ == 0 || layout.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_ಗಾತ್ರ) {
            kparams->kernel_type          = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_ಎಳೆಗಳು            = (uint8_t) n_ಎಳೆಗಳು;
            kparams->vtcm_ಗಾತ್ರ            = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
            kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
            kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
            kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
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
            kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು           = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_3d_params(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
    ರಚನೆ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ಗಾತ್ರ(*kparams));
    kparams->kernel_type = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_UNSUPPORTED;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (!src0 || !src1 || !dst) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = (ಸ್ಥಿರಾಂಕ int32_t *) op->op_params;
    ಸ್ಥಿರಾಂಕ int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ಸ್ಥಿರಾಂಕ int32_t ic = params[9];

    ಆದರೆ (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->ne[3] % ic != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t IC = (uint32_t) ic;
    ಸ್ಥಿರಾಂಕ uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    ಆದರೆ (N == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    uint32_t OW, OH, OD;
    ಆದರೆ (!im2col_3d_calc_output_ಗಾತ್ರ(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_ಗಾತ್ರ(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_ಗಾತ್ರ(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t KD = (uint32_t) src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t KH = (uint32_t) src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t KW = (uint32_t) src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ID = (uint32_t) src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t IH = (uint32_t) src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t IW = (uint32_t) src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64   = (uint64_t) N * OD * OH;
    ಸ್ಥಿರಾಂಕ uint64_t dst_ne3_64      = (uint64_t) OD * N;

    ಆದರೆ (patch_stride_64 > UINT32_MAX || ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು   = (uint32_t) ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_64;
    ಸ್ಥಿರಾಂಕ uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    ಆದರೆ (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ಗಾತ್ರ(ggml_fp16_t) : ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ಸ್ಥಿರಾಂಕ uint32_t max_ಎಳೆಗಳು = (sess && sess->n_ಎಳೆಗಳು > 0) ? (uint32_t) sess->n_ಎಳೆಗಳು : 8;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು < max_ಎಳೆಗಳು) ? ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : max_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t vtcm_ಗಾತ್ರ   = (sess && sess->vtcm_ಗಾತ್ರ > 0) ? sess->vtcm_ಗಾತ್ರ : 0;

    ಆದರೆ (is_patchembed) {
        ಸ್ಥಿರಾಂಕ uint64_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) IC * KD * KH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ uint64_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) OW * patch_stride * dst_elem;
        ಆದರೆ (src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw <= UINT32_MAX && dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw <= UINT32_MAX) {
            ಸ್ಥಿರಾಂಕ uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);
            ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);

            ರಚನೆ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಎಳೆಗಳು);

            ಆದರೆ (vtcm_ಗಾತ್ರ == 0 || layout.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_ಗಾತ್ರ) {
                kparams->kernel_type          = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_ಎಳೆಗಳು            = (uint8_t) n_ಎಳೆಗಳು;
                kparams->vtcm_ಗಾತ್ರ            = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
                kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
                kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
                kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
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
                kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು           = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }

    ಗಾಗಿ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ಸ್ಥಿರಾಂಕ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ಆದರೆ (Wb > UINT32_MAX) {
            ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ uint64_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) IC * KD * KH * Wb * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ uint64_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw = (uint64_t) owb * patch_stride * dst_elem;
        ಆದರೆ (src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw > UINT32_MAX || dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw > UINT32_MAX) {
            ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ uint32_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);
        ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_raw, 256);

        ರಚನೆ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಎಳೆಗಳು);

        ಆದರೆ (vtcm_ಗಾತ್ರ == 0 || layout.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_ಗಾತ್ರ) {
            kparams->kernel_type          = HTP_IM2_ಲಂಬಸಾಲು_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_ಎಳೆಗಳು            = (uint8_t) n_ಎಳೆಗಳು;
            kparams->vtcm_ಗಾತ್ರ            = layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
            kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
            kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
            kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
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
            kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು           = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (owb == 1) ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_im2col(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ರಚನೆ htp_im2col_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_im2col_3d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ರಚನೆ htp_im2col_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pad(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t lp0 = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
    ಸ್ಥಿರಾಂಕ int32_t rp0 = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[1];
    ಸ್ಥಿರಾಂಕ int32_t circular = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[8];

    ಆದರೆ (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cumsum(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_diag(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    // diag only supports F32 currently
    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Input must have ne[1] == 1 (vector input)
    ಆದರೆ (src0->ne[1] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Output must be square in first two dimensions
    ಆದರೆ (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_solve_tri(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0]; // A
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1]; // B
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;         // X

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->ne[0] != src0->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->ne[1] != src1->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_tri(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    ಆದರೆ (dst->type  != GGML_TYPE_F32) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    ಆದರೆ (!ggml_are_same_shape(src0, dst)) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_hexagon_name(ggml_backend_t backend) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ಹಿಂತಿರುಗಿ sess->c_name();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ htp_op_ಸಂಕೇತ op_remap_ಗಮ್ಯ_htp(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಆಯ್ಕೆ (t->op) {
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:  ಹಿಂತಿರುಗಿ HTP_OP_FLASH_ATTN_EXT;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಆದರೆ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ಹಿಂತಿರುಗಿ HTP_OP_HADAMARD;
            }
            ಹಿಂತಿರುಗಿ HTP_OP_MUL_MAT;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:      ಹಿಂತಿರುಗಿ HTP_OP_MUL_MAT_ID;
        ಸ್ಥಿತಿ GGML_OP_MUL:             ಹಿಂತಿರುಗಿ HTP_OP_MUL;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:             ಹಿಂತಿರುಗಿ HTP_OP_ಕೂಡಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:          ಹಿಂತಿರುಗಿ HTP_OP_ಕೂಡಿಸು_ID;
        ಸ್ಥಿತಿ GGML_OP_SUB:             ಹಿಂತಿರುಗಿ HTP_OP_SUB;
        ಸ್ಥಿತಿ GGML_OP_DIV:             ಹಿಂತಿರುಗಿ HTP_OP_DIV;
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_CPY:             ಹಿಂತಿರುಗಿ HTP_OP_CPY;
        ಸ್ಥಿತಿ GGML_OP_CONT:            ಹಿಂತಿರುಗಿ HTP_OP_CPY;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:        ಹಿಂತಿರುಗಿ HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:        ಹಿಂತಿರುಗಿ HTP_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:             ಹಿಂತಿರುಗಿ HTP_OP_ಮೊತ್ತ;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:        ಹಿಂತಿರುಗಿ HTP_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:         ಹಿಂತಿರುಗಿ HTP_OP_POOL_2D;
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:         ಹಿಂತಿರುಗಿ HTP_OP_POOL_1D;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:         ಹಿಂತಿರುಗಿ HTP_OP_ARGSORT;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:           ಹಿಂತಿರುಗಿ HTP_OP_ತುದಿ_K;
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:          ಹಿಂತಿರುಗಿ HTP_OP_ARGMAX;
        ಸ್ಥಿತಿ GGML_OP_NORM:            ಹಿಂತಿರುಗಿ HTP_OP_NORM;
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:         ಹಿಂತಿರುಗಿ HTP_OP_L2_NORM;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:        ಹಿಂತಿರುಗಿ HTP_OP_RMS_NORM;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:          ಹಿಂತಿರುಗಿ HTP_OP_CONCAT;
        ಸ್ಥಿತಿ GGML_OP_SCALE:           ಹಿಂತಿರುಗಿ HTP_OP_SCALE;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:           ಹಿಂತಿರುಗಿ HTP_OP_CLAMP;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:      ಹಿಂತಿರುಗಿ HTP_OP_LEAKY_RELU;
        ಸ್ಥಿತಿ GGML_OP_SQR:             ಹಿಂತಿರುಗಿ HTP_OP_SQR;
        ಸ್ಥಿತಿ GGML_OP_SQRT:            ಹಿಂತಿರುಗಿ HTP_OP_SQRT;
        ಸ್ಥಿತಿ GGML_OP_LOG:             ಹಿಂತಿರುಗಿ HTP_OP_UNARY_LOG;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:        ಹಿಂತಿರುಗಿ HTP_OP_SOFTMAX;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:        ಹಿಂತಿರುಗಿ HTP_OP_SSM_CONV;
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET: ಹಿಂತಿರುಗಿ HTP_OP_GATED_DELTA_NET;
        ಸ್ಥಿತಿ GGML_OP_ROPE:            ಹಿಂತಿರುಗಿ HTP_OP_ROPE;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:          ಹಿಂತಿರುಗಿ HTP_OP_REPEAT;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:          ಹಿಂತಿರುಗಿ HTP_OP_CUMSUM;
        ಸ್ಥಿತಿ GGML_OP_FILL:            ಹಿಂತಿರುಗಿ HTP_OP_FILL;
        ಸ್ಥಿತಿ GGML_OP_DIAG:            ಹಿಂತಿರುಗಿ HTP_OP_DIAG;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:       ಹಿಂತಿರುಗಿ HTP_OP_SOLVE_TRI;
        ಸ್ಥಿತಿ GGML_OP_TRI:             ಹಿಂತಿರುಗಿ HTP_OP_TRI;
        ಸ್ಥಿತಿ GGML_OP_PAD:             ಹಿಂತಿರುಗಿ HTP_OP_PAD;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:          ಹಿಂತಿರುಗಿ HTP_OP_IM2_ಲಂಬಸಾಲು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:       ಹಿಂತಿರುಗಿ HTP_OP_IM2_ಲಂಬಸಾಲು_3_D;
        ಸ್ಥಿತಿ GGML_OP_ROLL:            ಹಿಂತಿರುಗಿ HTP_OP_ROLL;

        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(t)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:       ಹಿಂತಿರುಗಿ HTP_OP_UNARY_SILU;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:       ಹಿಂತಿರುಗಿ HTP_OP_UNARY_GELU;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ: ಹಿಂತಿರುಗಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:   ಹಿಂತಿರುಗಿ HTP_OP_UNARY_GELU_ERF;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:    ಹಿಂತಿರುಗಿ HTP_OP_UNARY_SIGMOID;
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:        ಹಿಂತಿರುಗಿ HTP_OP_UNARY_NEG;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:        ಹಿಂತಿರುಗಿ HTP_OP_UNARY_EXP;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:   ಹಿಂತಿರುಗಿ HTP_OP_UNARY_SOFTPLUS;
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:       ಹಿಂತಿರುಗಿ HTP_OP_UNARY_TANH;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:        ಹಿಂತಿರುಗಿ HTP_OP_UNARY_ABS;
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:       ಹಿಂತಿರುಗಿ HTP_OP_UNARY_RELU;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:       ಹಿಂತಿರುಗಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(t)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:       ಹಿಂತಿರುಗಿ HTP_OP_GLU_SWIGLU;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:   ಹಿಂತಿರುಗಿ HTP_OP_GLU_SWIGLU_OAI;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP: ಹಿಂತಿರುಗಿ HTP_OP_GLU_SWIGLU_CLAMP;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:        ಹಿಂತಿರುಗಿ HTP_OP_GLU_GEGLU;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:  ಹಿಂತಿರುಗಿ HTP_OP_GLU_GEGLU_ಶೀಘ್ರ;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:    ಹಿಂತಿರುಗಿ HTP_OP_GLU_GEGLU_ERF;
                ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    ಹಿಂತಿರುಗಿ HTP_OP_INVALID;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ op_is_compute(ggml_tensor *ಗಂಟು)
{
    ಹಿಂತಿರುಗಿ !ggml_op_is_ಖಾಲಿಗಳು(ಗಂಟು->op) && !ggml_is_ಖಾಲಿಗಳು(ಗಂಟು) && (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ mm_is_hmx_eligible(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಆದರೆ (opt_nhmx == 0) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = t->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = t->src[1];

    ಆದರೆ (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ಸ್ಥಿರಾಂಕ ggml_tensor src1_c = ggml_hexagon_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(src1);
        ಹಿಂತಿರುಗಿ ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ಸುಳ್ಳು, ಸುಳ್ಳು);
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    ಹಿಂತಿರುಗಿ ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_nx_kernel(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams) {
    ಆದರೆ (kparams->n_hmx) {
        ಹಿಂತಿರುಗಿ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ಆದರೆ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_id_nx_kernel(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams) {
    ಆದರೆ (kparams->n_hmx) {
        ಹಿಂತಿರುಗಿ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ಆದರೆ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಆದರೆ (t->op != GGML_OP_MUL_MAT) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = t->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = t->src[1];
    ಆದರೆ (src0->ne[2] != 1 || src0->ne[3] != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಆದರೆ (mm_is_hmx_eligible(t)) {
        ಹಿಂತಿರುಗಿ htp_is_hmx_ತೂಕ_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    ಆದರೆ (src1->type != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಹಿಂತಿರುಗಿ ggml_hexagon_tensor_is_tiled(src0);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_pair(ಸ್ಥಿರಾಂಕ ggml_tensor * n1, ಸ್ಥಿರಾಂಕ ggml_tensor * n2) {
    ಆದರೆ (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[1] != n2->src[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[0]->type != n2->src[0]->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಆದರೆ (t->op != GGML_OP_MUL_MAT_ID) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = t->src[0];
    ಆದರೆ (mm_is_hmx_eligible(t)) {
        ಹಿಂತಿರುಗಿ htp_is_hmx_ತೂಕ_type(src0->type);
    }

    ಹಿಂತಿರುಗಿ ggml_hexagon_tensor_is_tiled(src0);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id_pair(ಸ್ಥಿರಾಂಕ ggml_tensor * n1, ಸ್ಥಿರಾಂಕ ggml_tensor * n2) {
    ಆದರೆ (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[1] != n2->src[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[2] != n2->src[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (n1->src[0]->type != n2->src[0]->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_hexagon_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);

    ಆದರೆ (sess->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ಸ್ಥಿರಾಂಕ std::vector<htp_opnode> * nodes_ptr = ಶೂನ್ಯಸೂಚಕ;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    ತಾರ್ಕಿಕ cache_ದೊರಕಿತು = (graph->uid != 0 && sess->cached_uid == graph->uid);
    ಆದರೆ (cache_ದೊರಕಿತು) {
        nodes_ptr = &sess->cached_nodes;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Tag fusable tensors in graph
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
            ಸ್ವಯಂಚಾಲಿತ * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            ಆದರೆ (!extra) ಮುಂದುವರಿ;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            ಆದರೆ (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                ಆದರೆ ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ಕೂಡಿಸು && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ಕೂಡಿಸು })) ||
                    ggml_ಗಂಟು_has_n_ಬಳಸುತ್ತದೆ(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            ಆದರೆ (!op_is_compute(n)) {
                ಮುಂದುವರಿ;
            }

            htp_opnode ಗಂಟು(HTP_OP_INVALID, n);
            ಗಂಟು.opcode = op_remap_ಗಮ್ಯ_htp(n);
            ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID || ಗಂಟು.opcode == HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
                ggml_tensor * src0 = ಗಂಟು.ಗಂಟು->src[0];
                ಆದರೆ (ggml_hexagon_needs_repack(src0)) {
                    ಆದರೆ (opt_verbose) {
                        ಅಕ್ಷರ dims_buf[64];
                        ಅಕ್ಷರ strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, ಗಾತ್ರ(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, ಗಾತ್ರ(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    ಸ್ಮರಣೆ_ನಕಲು(raw.ದತ್ತಾಂಶ(), src0->ದತ್ತಾಂಶ, raw.size());
                    htp_repack_tensor_tiled(src0, raw.ದತ್ತಾಂಶ(), raw.size());
                    ಸ್ವಯಂಚಾಲಿತ * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_mm_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MUL || ಗಂಟು.opcode == HTP_OP_ಕೂಡಿಸು || ಗಂಟು.opcode == HTP_OP_ಕೂಡಿಸು_ID || ಗಂಟು.opcode == HTP_OP_SUB || ಗಂಟು.opcode == HTP_OP_DIV) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = ಗಂಟು.ಗಂಟು->src[1];
                ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = (ಗಂಟು.opcode == HTP_OP_ಕೂಡಿಸು_ID) ? ಗಂಟು.ಗಂಟು->src[2] : ಶೂನ್ಯಸೂಚಕ;
                GGML_ASSERT(ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess,
                    ಗಂಟು.opcode, ಗಂಟು.ಗಂಟು->src[0], src1, ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params *) ಗಂಟು.kernel_params, src2));
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_fa_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (htp_op_is_unary(ಗಂಟು.opcode)) {
                ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();
                ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = inputs[0];
                ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : ಶೂನ್ಯಸೂಚಕ;
                ggml_hexagon_precompute_unary_params(sess,
                    ಗಂಟು.opcode, src0, src1, ಗಂಟು.dst(),
                    (ರಚನೆ htp_unary_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
                ggml_hexagon_precompute_get_ಅಡ್ಡಸಾಲುಗಳು_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
                ggml_hexagon_precompute_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_rope_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ರಚನೆ htp_ssm_conv_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_softmax_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_gdn_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_ARGSORT || ಗಂಟು.opcode == HTP_OP_ತುದಿ_K) {
                ggml_hexagon_precompute_ವಿಂಗಡಿಸು_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    ಗಂಟು.opcode == HTP_OP_ತುದಿ_K,
                    (ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_POOL_2D || ಗಂಟು.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.dst(),
                    (ರಚನೆ htp_pool_2d_kernel_params *)ಗಂಟು.kernel_params,
                    ಗಂಟು.opcode == HTP_OP_POOL_1D);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_concat_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_CPY || ಗಂಟು.opcode == HTP_OP_CPY_FENCE) {
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = ggml_hexagon_precompute_cpy_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_copy_kernel_params *) ಗಂಟು.kernel_params
                );
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params *) ಗಂಟು.kernel_params;
                ಆದರೆ (ok && ಗಂಟು.opcode == HTP_OP_CPY && kparams->ಒಟ್ಟು_elems == 0) {
                    ಮುಂದುವರಿ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_IM2_ಲಂಬಸಾಲು) {
                ggml_hexagon_precompute_im2col_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_im2col_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_IM2_ಲಂಬಸಾಲು_3_D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ರಚನೆ htp_im2col_kernel_params *) ಗಂಟು.kernel_params
                );
            }
            computed_nodes.push_back(std::move(ಗಂಟು));
        }

        ಆದರೆ (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } ಇಲ್ಲದಿದ್ದರೆ {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು : *nodes_ptr) {
        sess->ಸರತಿಗೆ_ಸೇರಿಸು_op(ಗಂಟು);
    }

    ಆದರೆ (sess->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ಆದರೆ (sess->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), ಸ್ಥಿತಿ_ಗಮ್ಯ_str(sess->ಕೊನೆಯ_error));
    }
}

ಎಣಿಕೆ ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ರಚನೆ ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ರಚನೆ ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    ಶೂನ್ಯ reset() {
        ranges.clear();
    }

    ಶೂನ್ಯ ಕೂಡಿಸು_2(ಸ್ಥಿರಾಂಕ ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    ತಾರ್ಕಿಕ check(ಸ್ಥಿರಾಂಕ ggml_hexagon_mem_range & mr) ಸ್ಥಿರಾಂಕ {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & cmp : ranges) {
            ಆದರೆ (mr.pb != cmp.pb) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
};

ಸ್ಥಿರ ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    ಆದರೆ (tensor->ಬಫರ್) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->ಬಫರ್,
            /*.p0 =*/ (uint64_t) tensor->ದತ್ತಾಂಶ,
            /*.p1 =*/ (uint64_t) tensor->ದತ್ತಾಂಶ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(tensor->ಬಫರ್->buft, tensor),
            /*.pt =*/ pt,
        };
    } ಇಲ್ಲದಿದ್ದರೆ {
        mr = {
            /*.pb =*/ (uint64_t) ತಳ,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    ಹಿಂತಿರುಗಿ mr;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_mem_ranges_ಕೂಡಿಸು_ಗಂಟು(ggml_hexagon_mem_ranges & mrs, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
    ಆದರೆ (ಗಂಟು.is_ಖಾಲಿಗಳು()) ಹಿಂತಿರುಗಿ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (ಗಂಟು.ಗಂಟು->src[i]) {
            mrs.ಕೂಡಿಸು_2(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.ಗಂಟು->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (fused->src[i]) {
                mrs.ಕೂಡಿಸು_2(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.ಕೂಡಿಸು_2(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_mem_ranges_check_ಗಂಟು(ಸ್ಥಿರಾಂಕ ggml_hexagon_mem_ranges & mrs, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
    ಆದರೆ (ಗಂಟು.is_ಖಾಲಿಗಳು()) ಹಿಂತಿರುಗಿ ನಿಜ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (ಗಂಟು.ಗಂಟು->src[i]) {
            ಆದರೆ (!mrs.check(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.ಗಂಟು->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (fused->src[i]) {
                ಆದರೆ (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        }
    }
    ಹಿಂತಿರುಗಿ mrs.check(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ಸ್ಥಿರ std::vector<ಪೂರ್ಣಾಂಕ> ggml_hexagon_ನಕ್ಷೆ_optimize_reorder(ಸ್ಥಿರಾಂಕ std::vector<htp_opnode> & nodes) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = nodes.size();

    std::vector<ಪೂರ್ಣಾಂಕ> res;
    res.reserve(n);

    std::vector<ತಾರ್ಕಿಕ> ಬಳಸಿದ(n, ಸುಳ್ಳು);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < n; i0++) {
        ಆದರೆ (ಬಳಸಿದ[i0]) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & node0 = nodes[i0];

        ಆದರೆ (!node0.stackable()) {
            res.push_back(i0);
            ಬಳಸಿದ[i0] = ನಿಜ;
            ಮುಂದುವರಿ;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr ಪೂರ್ಣಾಂಕ N_FORWARD = 16;

        std::vector<ಪೂರ್ಣಾಂಕ> ರಾಶಿ;
        ರಾಶಿ.push_back(i0);

        mrs.reset();

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            ಆದರೆ (ಬಳಸಿದ[i1]) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & node1 = nodes[i1];

            ಆದರೆ (node1.stackable() && node1.same_ಒಳಹರಿವು(node0) && ggml_hexagon_mem_ranges_check_ಗಂಟು(mrs, node1)) {
                ರಾಶಿ.push_back(i1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_hexagon_mem_ranges_ಕೂಡಿಸು_ಗಂಟು(mrs, node1);
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx : ರಾಶಿ) {
            res.push_back(idx);
            ಬಳಸಿದ[idx] = ನಿಜ;
        }
    }

    ಹಿಂತಿರುಗಿ res;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_ನಕ್ಷೆ_optimize_params * params) {
    GGML_ಬಳಸದ(params);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = gf->n_nodes;

    constexpr ಪೂರ್ಣಾಂಕ MAX_FUSE = 16;

    ಎಣಿಕೆ ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        htp_opnode ಗಂಟು(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        ಆದರೆ (ಗಂಟು.op() == GGML_OP_ಕೂಡಿಸು ||
            ಗಂಟು.op() == GGML_OP_NORM ||
            ಗಂಟು.op() == GGML_OP_RMS_NORM) {
            ops[0] = ಗಂಟು.op();

            ಪೂರ್ಣಾಂಕ f = i + 1;
            ಆಗ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                ಆದರೆ (gf->nodes[f]->op != GGML_OP_ಕೂಡಿಸು &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ನಿಲ್ಲಿಸು;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            ಗಾಗಿ (; f > 1; f--) {
                ಆದರೆ (ggml_can_fuse(gf, i, ops, f)) {
                    ನಿಲ್ಲಿಸು;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                ಗಂಟು.ಕೂಡಿಸು_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(ಗಂಟು));
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕ್ರಮ = ggml_hexagon_ನಕ್ಷೆ_optimize_reorder(nodes);

    // unfuse
    {
        ಪೂರ್ಣಾಂಕ j = 0;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ i : ಕ್ರಮ) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು = nodes[i];

            gf->nodes[j++] = ಗಂಟು.ಗಂಟು;

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * fused : ಗಂಟು.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ uint64_t ggml_hexagon_session_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ ggml_hexagon_session * sess) {
    ಹಿಂತಿರುಗಿ ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಸ್ವಯಂಚಾಲಿತ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ಸ್ವಯಂಚಾಲಿತ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    ಸ್ವಯಂಚಾಲಿತ sbuf_dst = sess_dst->mmap_tensor(dst);

    ಆದರೆ (!sess_src->clone_ಬಫರ್(sbuf_dst)) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

    ಸ್ಥಿರಾಂಕ uint64_t src_ಕೀಲಿ = ggml_hexagon_session_ಕೀಲಿ(sess_src);
    ಸ್ವಯಂಚಾಲಿತ & fence_ಸ್ಥಾನ = sess_dst->cpy_fence_slots[src_ಕೀಲಿ];
    ಆದರೆ (!fence_ಸ್ಥಾನ) {
        fence_ಸ್ಥಾನ = (ಅಸ್ಥಿರ uint32_t *) sess_dst->alloc_fence(1);
    }

    ಆದರೆ (!sess_src->clone_ಬಫರ್(sess_dst->fence_buf)) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

    ಆದರೆ (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.ಬಫರ್ = &sess_dst->fence_buf->backend_ಬಫರ್;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.ದತ್ತಾಂಶ   = (ಶೂನ್ಯ *) fence_ಸ್ಥಾನ;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = ಗಾತ್ರ(int32_t);
    fence_tensor.nb[1]  = ಗಾತ್ರ(int32_t);
    fence_tensor.nb[2]  = ಗಾತ್ರ(int32_t);
    fence_tensor.nb[3]  = ಗಾತ್ರ(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->ಸರತಿಗೆ_ಸೇರಿಸು_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->ಸರತಿಗೆ_ಸೇರಿಸು_fence(&fence_tensor, fence_seq, /* wait = */ ನಿಜ);

    sess_dst->ಕೂಡಿಸು_peer(sess_src);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಸ್ವಯಂಚಾಲಿತ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ಸ್ವಯಂಚಾಲಿತ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ಸ್ವಯಂಚಾಲಿತ sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    ಆದರೆ (!sess_dst->clone_ಬಫರ್(sbuf_src)) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->ಸರತಿಗೆ_ಸೇರಿಸು_cpy(src, dst);
    sess_dst->ಕೂಡಿಸು_peer(sess_src);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಆದರೆ (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    ಆದರೆ (!dst->extra) {
        ggml_backend_ಬಫರ್_init_tensor(dst->ಬಫರ್, dst);
    }

    ಸ್ವಯಂಚಾಲಿತ * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src_extra = static_cast<ಸ್ಥಿರಾಂಕ ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    ಸ್ವಯಂಚಾಲಿತ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ಸ್ವಯಂಚಾಲಿತ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ಆದರೆ (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->ಸರತಿಗೆ_ಸೇರಿಸು_cpy(src, dst);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (sess_src->phys_idx != sess_dst->phys_idx)
        ಹಿಂತಿರುಗಿ ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    ಹಿಂತಿರುಗಿ ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಸ್ವಯಂಚಾಲಿತ sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = ಹೊಸ ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_ಸ್ಥಾನ = (ಅಸ್ಥಿರ uint32_t *) sess->alloc_fence(1);

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.ಬಫರ್ = &sess->fence_buf->backend_ಬಫರ್;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.ದತ್ತಾಂಶ   = (ಶೂನ್ಯ *) hex_event->fence_ಸ್ಥಾನ;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = ಗಾತ್ರ(int32_t);
    hex_event->fence_tensor.nb[1]  = ಗಾತ್ರ(int32_t);
    hex_event->fence_tensor.nb[2]  = ಗಾತ್ರ(int32_t);
    hex_event->fence_tensor.nb[3]  = ಗಾತ್ರ(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event, (ಶೂನ್ಯ *)hex_event->fence_ಸ್ಥಾನ);

    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    ಆದರೆ (hex_event->seq == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_ಸ್ಥಾನ);

    ಸ್ವಯಂಚಾಲಿತ * fence = reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಸ್ಥಿರ std::atomic<uint32_t> *>(hex_event->fence_ಸ್ಥಾನ);

    ಆದರೆ ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    ಆಗ (ನಿಜ) {
        ಆದರೆ ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t ಸ್ಥಿತಿ_2 = fence[1].load(std::memory_order_acquire);
            ಆದರೆ (ಸ್ಥಿತಿ_2 > HTP_ಸ್ಥಿತಿ_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), ಸ್ಥಿತಿ_ಗಮ್ಯ_str(ಸ್ಥಿತಿ_2));
            }
            ನಿಲ್ಲಿಸು;
        }
        std::this_thread::yield();
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ಸ್ವಯಂಚಾಲಿತ * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event);
    hex_event->fence_sess->free_fence((ಶೂನ್ಯ *) hex_event->fence_ಸ್ಥಾನ, 1);
    ಅಳಿಸು hex_event;
    ಅಳಿಸು event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ಸ್ವಯಂಚಾಲಿತ * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ಸ್ವಯಂಚಾಲಿತ hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ಆದರೆ (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->ಸರತಿಗೆ_ಸೇರಿಸು_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ಸುಳ್ಳು);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_ಸ್ಥಾನ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ಸ್ವಯಂಚಾಲಿತ hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ಆದರೆ (hex_event->seq == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_ಸ್ಥಾನ);

    // same physical NPU runs sequentially in FIFO order
    ಆದರೆ (sess->phys_idx == hex_event->sess->phys_idx) {
        ಆದರೆ (sess != hex_event->sess) {
            sess->ಕೂಡಿಸು_peer(hex_event->sess);
        }
        ಹಿಂತಿರುಗಿ;
    }

    sess->clone_ಬಫರ್(hex_event->fence_sess->fence_buf);
    sess->ಕೂಡಿಸು_peer(hex_event->sess);
    sess->ಸರತಿಗೆ_ಸೇರಿಸು_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ನಿಜ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಗಣ_tensor_async(ggml_backend_t backend, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, tensor->ಬಫರ್ ? (ಪೂರ್ಣಾಂಕ) tensor->ಬಫರ್->usage : -1);
    ggml_backend_tensor_ಗಣ(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, tensor->ಬಫರ್ ? (ಪೂರ್ಣಾಂಕ) tensor->ಬಫರ್->usage : -1);
    sess->flush_sync();
    ಆದರೆ (sess->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), ಸ್ಥಿತಿ_ಗಮ್ಯ_str(sess->ಕೊನೆಯ_error));
    }
    ggml_backend_tensor_get(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_ಗಣ_tensor_2d_async(ggml_backend_t backend,
                                                     ರಚನೆ ggml_tensor * tensor,
                                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,
                                                     size_t ಸ್ಥಾನಾಂತರ,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ದತ್ತಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ, tensor->ಬಫರ್ ? (ಪೂರ್ಣಾಂಕ) tensor->ಬಫರ್->usage : -1);
    ggml_backend_tensor_ಗಣ_2d(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor,
                                                     ಶೂನ್ಯ * ದತ್ತಾಂಶ,
                                                     size_t ಸ್ಥಾನಾಂತರ,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ದತ್ತಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ, tensor->ಬಫರ್ ? (ಪೂರ್ಣಾಂಕ) tensor->ಬಫರ್->usage : -1);
    sess->flush_sync();
    ಆದರೆ (sess->ಕೊನೆಯ_error > HTP_ಸ್ಥಿತಿ_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), ಸ್ಥಿತಿ_ಗಮ್ಯ_str(sess->ಕೊನೆಯ_error));
    }
    ggml_backend_tensor_get_2d(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_ಗಣ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_hexagon_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_ನಕ್ಷೆ_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_hexagon_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    ಹಿಂತಿರುಗಿ &guid;
}

ತಾರ್ಕಿಕ ggml_backend_is_hexagon(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ಸ್ಥಿರ ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಸ್ವಯಂಚಾಲಿತ sess    = dev_ctx->session();

    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ dev_ctx->c_name();

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "Hexagon";
    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    *ಬಿಡುಗಡೆ  = opt_mbuf_ಒಟ್ಟು;
    *ಒಟ್ಟು = opt_mbuf_ಒಟ್ಟು;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ನಿಜ,
        /* .mmap_support          = */ ಸುಳ್ಳು,
    };
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_hexagon_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ &dev_ctx->ಬಫರ್_type;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_hexagon_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಆದರೆ (!opt_hostbuf) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ &dev_ctx->host_ಬಫರ್_type;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cpy(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ರಚನೆ htp_copy_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cont(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ರಚನೆ htp_copy_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_repeat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ಬಳಸದ(sess);
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    // Support f32 and f16
    ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    // src and dst must be the same type
    ಆದರೆ (src0->type != dst->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    // dst dims must be multiples of src dims
    ಆದರೆ (dst->ne[0] % src0->ne[0] != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (dst->ne[1] % src0->ne[1] != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (dst->ne[2] % src0->ne[2] != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (dst->ne[3] % src0->ne[3] != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    // require contiguous tensors (no transposition)
    ಆದರೆ (ggml_is_transposed(src0) || ggml_is_transposed(dst)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_concat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ರಚನೆ htp_concat_kernel_params kparams;
    ಹಿಂತಿರುಗಿ ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_fill(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst = op;

    ಆದರೆ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
    GGML_ಬಳಸದ(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_roll(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hexagon_session * sess, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ಬಳಸದ(sess);

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst  = op;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_are_same_shape(src0, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src0->nb[0] != ggml_type_ಗಾತ್ರ(src0->type) || dst->nb[0] != ggml_type_ಗಾತ್ರ(dst->type)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಸ್ವಯಂಚಾಲಿತ sess    = dev_ctx->session();

    // reject ops that match the filter
    ಆದರೆ (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ supp = ಸುಳ್ಳು;
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            supp = ನಿಜ;
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_DIV:
            supp = ggml_hexagon_supported_ದ್ವಿಮಾನ(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            supp = ggml_hexagon_supported_ಕೂಡಿಸು_id(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            supp = ggml_hexagon_supported_ಮೊತ್ತ(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            supp = ggml_hexagon_supported_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    supp = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    supp = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            supp = ggml_hexagon_supported_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            supp = ggml_hexagon_supported_get_ಅಡ್ಡಸಾಲುಗಳು(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            supp = ggml_hexagon_supported_ತುದಿ_k(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    ಹಿಂತಿರುಗಿ supp;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    ತಾರ್ಕಿಕ supp = (buft == &dev_ctx->host_ಬಫರ್_type) || (buft == &dev_ctx->ಬಫರ್_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    ಹಿಂತಿರುಗಿ supp;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr = */ ಖಾಲಿ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    ಗಾಗಿ (size_t i = 0; i < opt_ndev; i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & cfg = opt_device_configs[i];
        ಆದರೆ (cfg.mdev_ಗುಂಪು.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } ಇಲ್ಲದಿದ್ದರೆ {
            std::string peers_str;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & p : cfg.mdev_ಗುಂಪು) {
                ಆದರೆ (!peers_str.ಖಾಲಿಗಳು()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = ಹೊಸ ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    ಗಾಗಿ (size_t i = 0; i < opt_ndev; i++) {
        ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        ಅಳಿಸು dev_ctx;
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ "HTP";
    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ opt_ndev;
    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    ಸ್ವಯಂಚಾಲಿತ hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    ಆದರೆ (ಸೂಚ್ಯಂಕ >= opt_ndev || !hreg->devices[ಸೂಚ್ಯಂಕ].context) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ &hreg->devices[ಸೂಚ್ಯಂಕ];
}

// ** communication context for tensor-split allreduce

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ಆದರೆ (n_backends < 2 || n_backends > 4) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಗಾಗಿ (size_t i = 0; i < n_backends; ++i) {
        ಆದರೆ (!ggml_backend_is_hexagon(backends[i])) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ಗಾಗಿ (size_t j = i + 1; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            ಆದರೆ (sess_i->phys_idx == sess_j->phys_idx) {
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
    }

    ಸ್ವಯಂಚಾಲಿತ * ctx = ಹೊಸ ggml_backend_hexagon_comm_context();
    ctx->backends.ನಿಯೋಜಿಸು(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (ಅಸ್ಥಿರ uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].ಬಫರ್ = &sess_i->fence_buf->backend_ಬಫರ್;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].ದತ್ತಾಂಶ   = (ಶೂನ್ಯ *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = ಗಾತ್ರ(int32_t);
        ctx->fence_tensors[i].nb[1]  = ಗಾತ್ರ(int32_t);
        ctx->fence_tensors[i].nb[2]  = ಗಾತ್ರ(int32_t);
        ctx->fence_tensors[i].nb[3]  = ಗಾತ್ರ(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    ಹಿಂತಿರುಗಿ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_comm_free(ಶೂನ್ಯ * comm_ctx_v) {
    ಆದರೆ (!comm_ctx_v) ಹಿಂತಿರುಗಿ;
    ಸ್ವಯಂಚಾಲಿತ * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ಗಾಗಿ (size_t i = 0; i < ctx->n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((ಶೂನ್ಯ *) ctx->fence_slots[i], 1);
    }
    ಅಳಿಸು ctx;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_comm_allreduce_tensor(ಶೂನ್ಯ * comm_ctx_v, ರಚನೆ ggml_tensor ** tensors) {
    ಆದರೆ (opt_ar_select == 0 || !comm_ctx_v) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಸ್ವಯಂಚಾಲಿತ * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ಸ್ಥಿರಾಂಕ size_t n_backends = comm_ctx->n_backends;

    ಆದರೆ (n_backends < 2 || n_backends > 4) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ಗಾಗಿ (size_t j = i + 1; j < n_backends; j++) {
            ಸ್ವಯಂಚಾಲಿತ sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            ಆದರೆ (sess_i->phys_idx == sess_j->phys_idx) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಆದರೆ (!tensors[i] || !tensors[i]->ಬಫರ್ || !ggml_backend_ಬಫರ್_is_hexagon(tensors[i]->ಬಫರ್)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (tensors[i]->type != tensors[0]->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_is_contiguous(tensors[i])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (size_t r = 0; r < n_backends; r++) {
        ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ರಚನೆ htp_allreduce_kernel_params kparams;
        ಆದರೆ (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, ಸುಳ್ಳು, ಸುಳ್ಳು, /*is_shard_ok=*/ ಸುಳ್ಳು, &kparams)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    ಗಾಗಿ (size_t i = 1; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ಆದರೆ ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    ಆದರೆ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_ನಮೂದು = max_seq;
    ಆದರೆ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ದತ್ತಾಂಶ_tensors(n_backends);
    std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> sync_tensors(n_backends);
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ದತ್ತಾಂಶ_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    ಗಾಗಿ (size_t r = 0; r < n_backends; r++) {
        ಸ್ವಯಂಚಾಲಿತ sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->ಸರತಿಗೆ_ಸೇರಿಸು_allreduce(tensors[r], ದತ್ತಾಂಶ_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_ನಮೂದು, fence_seq_exit);
        ಗಾಗಿ (size_t j = 0; j < n_backends; j++) {
            ಆದರೆ (r != j) {
                sess->ಕೂಡಿಸು_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_hexagon_ವಿಭಜಿಸು_ಬಫರ್_type(ಪೂರ್ಣಾಂಕ main_device, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * tensor_ವಿಭಜಿಸು) {
    GGML_ಬಳಸದ(tensor_ವಿಭಜಿಸು);
    ಸ್ವಯಂಚಾಲಿತ reg = ggml_backend_hexagon_reg();
    ಸ್ವಯಂಚಾಲಿತ dev = ggml_backend_reg_dev_get(reg, main_device);
    ಆದರೆ (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    ಆದರೆ (!dev) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ವಯಂಚಾಲಿತ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ &dev_ctx->ಬಫರ್_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    GGML_ಬಳಸದ(reg);
    ಆದರೆ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_hexagon_ವಿಭಜಿಸು_ಬಫರ್_type;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_comm_init") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_init;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_comm_free") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_free;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T> std::vector<T> str_ಗಮ್ಯ_vec(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    ಆಗ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, ಶೂನ್ಯಸೂಚಕ, 0));
    }

    ಹಿಂತಿರುಗಿ v;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ತಳ_2=10> std::string vec_ಗಮ್ಯ_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(ತಳ_2) << std::showbase;
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ i : v) { ss << i << ','; }
    ಸ್ವಯಂಚಾಲಿತ str = ss.str(); str.pop_back(); // drop last comma
    ಹಿಂತಿರುಗಿ str;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, ತಾರ್ಕಿಕ discovery_supported, ಸ್ಥಿರಾಂಕ std::unordered_map<ಪೂರ್ಣಾಂಕ, fastrpc_domain> & cdsp_map) {
    ಆದರೆ (discovery_supported) {
        ಸ್ವಯಂಚಾಲಿತ it = cdsp_map.find(cfg.physical_idx);
        ಆದರೆ (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆಯ್ಕೆ (cfg.physical_idx) {
            ಸ್ಥಿತಿ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ನಿಲ್ಲಿಸು;
        }
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sub_cfg : cfg.mdev_ಗುಂಪು) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_discover_devices() {
    std::unordered_map<ಪೂರ್ಣಾಂಕ, fastrpc_domain> cdsp_map;
    ತಾರ್ಕಿಕ discovery_supported = ಸುಳ್ಳು;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = ಶೂನ್ಯಸೂಚಕ;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_ಗಣ_TYPE(0, FASTRPC_NSP);

    ಪೂರ್ಣಾಂಕ err = remote_system_request(&domain_info);
    ಆದರೆ (err == AEE_SUCCESS && domain_info.sys.ಅಂಕೆ_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.ಅಂಕೆ_domains);
        domain_info.sys.domains     = domains.ದತ್ತಾಂಶ();
        domain_info.sys.max_domains = (ಪೂರ್ಣಾಂಕ) domains.size();

        err = remote_system_request(&domain_info);
        ಆದರೆ (err == AEE_SUCCESS) {
            discovery_supported = ನಿಜ;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_domains = std::min(domain_info.sys.ಅಂಕೆ_domains, (ಪೂರ್ಣಾಂಕ) domains.size());
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (ಪೂರ್ಣಾಂಕ) domains[i].type, domains[i].id, domains[i].name, domains[i].ಸ್ಥಿತಿ_2, domains[i].instance_id);
                ಆದರೆ (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (ಪೂರ್ಣಾಂಕ) domains[i].type);
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (!domains[i].ಸ್ಥಿತಿ_2) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ಮುಂದುವರಿ;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (ಚಿಹ್ನೆರಹಿತ) err);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (ಚಿಹ್ನೆರಹಿತ) err);
    }

    // Populate domain IDs and names for all configured devices
    ಗಾಗಿ (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q4_0 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q4_1 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q8_0 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_MXFP4 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_IQ4_NL == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q4_K == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q5_K == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q6_K == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q3_K == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q2_K == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q1_0 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_Q2_0 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) HTP_TYPE_BF16 == (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_etm      = getenv("GGML_HEXAGON_ETM");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_fa_ತಲೆ_ವಿಭಜಿಸು = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_power_ಬಹುಲಕ = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    ಆದರೆ (!str_arch) {
        ಪೂರ್ಣಾಂಕ err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        ಆದರೆ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ಖಾಲಿ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    ಸ್ವಯಂಚಾಲಿತ RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? ಹೊಸ std::regex(str_opfilter, RE_ICASE) : ಖಾಲಿ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ಖಾಲಿ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ಖಾಲಿ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ಖಾಲಿ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ಖಾಲಿ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    ಆದರೆ (str_power_ಬಹುಲಕ) {
        ಪೂರ್ಣಾಂಕ m = htp_power_ಬಹುಲಕ_from_str(str_power_ಬಹುಲಕ);
        ಆದರೆ (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_ಬಹುಲಕ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            opt_power_ಬಹುಲಕ = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ಖಾಲಿ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_ತಲೆ_ವಿಭಜಿಸು = str_fa_ತಲೆ_ವಿಭಜಿಸು ? atoi(str_fa_ತಲೆ_ವಿಭಜಿಸು)       : opt_fa_ತಲೆ_ವಿಭಜಿಸು;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    ಆದರೆ (str_mbuf) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * p = str_mbuf;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = 0; idx < 3 && p && *p; idx++) {
            ಆಗ (*p == ' ') p++;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(p);
            ಆಗ (len > 0 && p[len - 1] == ' ') len--;
            ಆದರೆ (len > 0) {
                std::string token(p, len);
                ಆದರೆ (idx == 0) opt_mbuf_dyn    = parse_ಗಾತ್ರ(token.c_str());
                ಆದರೆ (idx == 1) opt_mbuf_static = parse_ಗಾತ್ರ(token.c_str());
                ಆದರೆ (idx == 2) opt_mbuf_ಒಟ್ಟು  = parse_ಗಾತ್ರ(token.c_str());
            }
            ಆದರೆ (!comma) ನಿಲ್ಲಿಸು;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_ಗಾತ್ರ(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    ಆದರೆ (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    ಆದರೆ (str_devices && str_devices[0] != '\0') {
        ತಾರ್ಕಿಕ is_single_ಸಂಖ್ಯೆ = ನಿಜ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; str_devices[i] != '\0'; i++) {
            ಆದರೆ (!ಅಂಕಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)str_devices[i])) {
                is_single_ಸಂಖ್ಯೆ = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (is_single_ಸಂಖ್ಯೆ) {
            ಪೂರ್ಣಾಂಕ n = atoi(str_devices);
            ಆದರೆ (n < 1) n = 1;
            ಆದರೆ (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            ಗಾಗಿ (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (ಪೂರ್ಣಾಂಕ)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_ಗುಂಪು.clear();
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            std::string s_devices(str_devices);
            std::vector<std::string> ವಸ್ತುಗಳು;
            std::string curr_ಅಂಶ;
            ಪೂರ್ಣಾಂಕ bracket_ಆಳ = 0;
            ಗಾಗಿ (ಅಕ್ಷರ ch : s_devices) {
                ಆದರೆ (ch == '[') {
                    bracket_ಆಳ++;
                    curr_ಅಂಶ += ch;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ch == ']') {
                    ಆದರೆ (bracket_ಆಳ > 0) bracket_ಆಳ--;
                    curr_ಅಂಶ += ch;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ch == ',' && bracket_ಆಳ == 0) {
                    size_t s = curr_ಅಂಶ.find_first_not_of(" \t\r\n");
                    size_t e = curr_ಅಂಶ.find_last_not_of(" \t\r\n");
                    ಆದರೆ (s != std::string::npos) {
                        ವಸ್ತುಗಳು.push_back(curr_ಅಂಶ.substr(s, e - s + 1));
                    }
                    curr_ಅಂಶ.clear();
                } ಇಲ್ಲದಿದ್ದರೆ {
                    curr_ಅಂಶ += ch;
                }
            }
            size_t s = curr_ಅಂಶ.find_first_not_of(" \t\r\n");
            size_t e = curr_ಅಂಶ.find_last_not_of(" \t\r\n");
            ಆದರೆ (s != std::string::npos) {
                ವಸ್ತುಗಳು.push_back(curr_ಅಂಶ.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಅಂಶ : ವಸ್ತುಗಳು) {
                size_t b_open  = ಅಂಶ.find('[');
                size_t b_close = ಅಂಶ.rfind(']');

                ಆದರೆ (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = ಅಂಶ.substr(0, b_open);
                    std::string content  = ಅಂಶ.substr(b_open + 1, b_close - b_open - 1);

                    ಪೂರ್ಣಾಂಕ virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    ಆದರೆ (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ಪ್ರಯತ್ನಿಸು {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } ಹಿಡಿ (...) {
                            virt = 0;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        size_t dev_colon = dev_name.find(':');
                        ಆದರೆ (dev_colon != std::string::npos) {
                            ಪ್ರಯತ್ನಿಸು {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } ಹಿಡಿ (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<ಪೂರ್ಣಾಂಕ> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    ಆಗ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        ಆದರೆ (ps == std::string::npos) ಮುಂದುವರಿ;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        ಆದರೆ (dash_pos != std::string::npos) {
                            ಪ್ರಯತ್ನಿಸು {
                                ಪೂರ್ಣಾಂಕ p_ಆರಂಭ = std::stoi(p_part.substr(0, dash_pos));
                                ಪೂರ್ಣಾಂಕ p_ಕೊನೆ   = std::stoi(p_part.substr(dash_pos + 1));
                                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ p = p_ಆರಂಭ; p <= p_ಕೊನೆ; p++) {
                                    ಆದರೆ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } ಹಿಡಿ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಪ್ರಯತ್ನಿಸು {
                                ಪೂರ್ಣಾಂಕ p = std::stoi(p_part);
                                ಆದರೆ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } ಹಿಡಿ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    ಆದರೆ (phys_list.ಖಾಲಿಗಳು()) {
                        phys_list.push_back(0);
                    }

                    ಆದರೆ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        ಸ್ವಯಂಚಾಲಿತ & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_ಗುಂಪು.clear();

                        ಗಾಗಿ (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_ಗುಂಪು.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, ಅಂಶ.c_str());
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಅಂಶ.rfind("HTP", 0) == 0) {
                    std::string rest = ಅಂಶ.substr(3);
                    size_t colon_pos = rest.find(':');
                    ಪೂರ್ಣಾಂಕ phys = 0;
                    ಪೂರ್ಣಾಂಕ virt = 0;
                    ಪ್ರಯತ್ನಿಸು {
                        ಆದರೆ (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } ಹಿಡಿ (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", ಅಂಶ.c_str());
                        ಮುಂದುವರಿ;
                    }

                    ಆದರೆ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_ಗುಂಪು.clear();
                        opt_ndev++;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, ಅಂಶ.c_str());
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", ಅಂಶ.c_str());
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_ಗುಂಪು.clear();
    }

#ಆದರೆ defined(__ANDROID__)
    ಆದರೆ (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#ಆದರೆ_ಅಂತ್ಯ

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    ಆದರೆ (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            ಸ್ವಯಂಚಾಲಿತ v  = str_ಗಮ್ಯ_vec<uint32_t>(str_profile);
            ಆಯ್ಕೆ (v.size()) {
                ಸ್ಥಿತಿ 1:  opt_profile = v[0]; ಹಿಂತಿರುಗಿ opt_pmu_evt; // mode with default pmu events
                ಸ್ಥಿತಿ 8:  opt_profile = 2;    ಹಿಂತಿರುಗಿ v;           // mode with custom  pmu events
                ಪೂರ್ವನಿಯೋಜಿತ: opt_profile = 0;    ಹಿಂತಿರುಗಿ {};          // garbage input
            }}();
        ಆದರೆ (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_ಗಮ್ಯ_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = ಹೊಸ ggml_hexagon_registry(reg);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    ಸ್ಥಿರ ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ಖಾಲಿ };

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (!initialized) {
            ಸ್ವಯಂಚಾಲಿತ nErr = htpdrv_init();
            ಆದರೆ (nErr != AEE_SUCCESS) {
                ಹಿಂತಿರುಗಿ ಖಾಲಿ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
