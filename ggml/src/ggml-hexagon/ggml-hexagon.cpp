#define _USE_MATH_DEFINES

#अन्तर्भाव <assert.h>
#अन्तर्भाव <inttypes.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <time.h>

#अन्तर्भाव <atomic>
#अन्तर्भाव <memory>
#अन्तर्भाव <chrono>
#अन्तर्भाव <mutex>
#अन्तर्भाव <thread>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <string>
#अन्तर्भाव <sstream>
#अन्तर्भाव <iomanip>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <regex>
#अन्तर्भाव <queue>
#अन्तर्भाव <deque>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <initializer_list>

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
#अन्यथा
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#अन्तर्भाव <AEEStdErr.h>
#अन्तर्भाव <dspqueue.h>
#अन्तर्भाव <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml-hexagon.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव "htp-opnode.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp/matmul-ops.h"
#अन्तर्भाव "htp/binary-ops.h"
#अन्तर्भाव "htp/flash-attn-ops.h"
#अन्तर्भाव "htp/unary-ops.h"
#अन्तर्भाव "htp/get-rows-ops.h"
#अन्तर्भाव "htp/set-rows-ops.h"
#अन्तर्भाव "htp/softmax-ops.h"
#अन्तर्भाव "htp/pool-ops.h"
#अन्तर्भाव "htp/rope-ops.h"
#अन्तर्भाव "htp/ssm-conv.h"
#अन्तर्भाव "htp/gated-delta-net-ops.h"
#अन्तर्भाव "htp/argsort-ops.h"
#अन्तर्भाव "htp/concat-ops.h"
#अन्तर्भाव "htp/cpy-ops.h"
#अन्तर्भाव "htp/im2col-ops.h"
#अन्तर्भाव "htp_iface.h"
#अन्तर्भाव "htp-drv.h"
#अन्तर्भाव "htp-repack.h"

using intvec  = std::vector<पूर्णाङ्क>;
using uintvec = std::vector<unsigned पूर्णाङ्क>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

संरचना ggml_hexagon_device_config {
    पूर्णाङ्क         physical_idx = 0;
    पूर्णाङ्क         virtual_idx  = 0;
    पूर्णाङ्क         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

स्थिर ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

स्थिर पूर्णाङ्क    opt_arch    = 0; // autodetect
स्थिर size_t opt_ndev    = 1;
स्थिर size_t opt_nhvx    = 0; // use all
स्थिर पूर्णाङ्क    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
स्थिर size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
स्थिर पूर्णाङ्क    opt_etm     = 0;
स्थिर पूर्णाङ्क    opt_verbose = 0;
स्थिर पूर्णाङ्क    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
स्थिर तार्किक   opt_hostbuf = असत्य;
स्थिर तार्किक   opt_dma64   = असत्य;
स्थिर पूर्णाङ्क    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

स्थिर ध्रुव std::pair<ध्रुव वर्ण *, पूर्णाङ्क> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_खालचा_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_वरचा_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_खालचा_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_वरचा_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_वरचा_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

स्थिर पूर्णाङ्क htp_power_mode_from_str(ध्रुव वर्ण * s) {
    प्रति (auto & e : htp_power_mode_names) {
#यदि defined(_MSC_VER)
        यदि (_stricmp(s, e.first) == 0) {
#अन्यथा
        यदि (strcasecmp(s, e.first) == 0) {
#endif
            प्रत्यागम e.second;
        }
    }
    प्रत्यागम -1;
}

स्थिर ध्रुव वर्ण * htp_power_mode_to_str(पूर्णाङ्क mode) {
    प्रति (auto & e : htp_power_mode_names) {
        यदि (e.second == mode) {
            प्रत्यागम e.first;
        }
    }
    प्रत्यागम "unknown";
}

स्थिर size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
स्थिर size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
स्थिर size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

स्थिर पूर्णाङ्क    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
स्थिर पूर्णाङ्क    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
स्थिर पूर्णाङ्क    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
स्थिर पूर्णाङ्क    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
स्थिर पूर्णाङ्क    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
स्थिर पूर्णाङ्क    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
स्थिर u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

स्थिर पूर्णाङ्क opt_opbatch  = 1280; // max number of ops in a batch
स्थिर पूर्णाङ्क opt_opqueue  = 8;   // max number of pending batches
स्थिर पूर्णाङ्क opt_optrace  = 0;    // trace buffer size per thread (0 means default)
स्थिर पूर्णाङ्क opt_oppoll   = 0;    // polling for batch completions
स्थिर पूर्णाङ्क opt_opfusion  = 1;    // enable/disable op fusion
स्थिर पूर्णाङ्क opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

गणना ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

स्थिर अन्तःस्थ तार्किक ggml_hexagon_is_fusion_enabled(पूर्णाङ्क flag) {
    यदि (opt_opfusion <= 0) प्रत्यागम असत्य;
    यदि (opt_opfusion == 1) प्रत्यागम सत्य; // 1 enables all
    प्रत्यागम (opt_opfusion & flag) != 0;
}

स्थिर std::regex* opt_opfilter = रिक्त; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    यदि (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

स्थिर ध्रुव वर्ण * status_to_str(uint32_t status) {
    विकल्प (status) {
        पक्ष HTP_STATUS_OK:
            प्रत्यागम "OK";
        पक्ष HTP_STATUS_NO_SUPPORT:
            प्रत्यागम "NO-SUPPORT";
        पक्ष HTP_STATUS_INVAL_PARAMS:
            प्रत्यागम "INVAL-PARAMS";
        पक्ष HTP_STATUS_VTCM_TOO_SMALL:
            प्रत्यागम "VTCM-TOO-SMALL";
        पक्ष HTP_STATUS_INTERNAL_ERR:
            प्रत्यागम "INTERNAL-ERROR";
        स्वभाव:
            प्रत्यागम "UNKNOWN";
    }
}

// ** debug helpers

स्थिर शून्य ggml_hexagon_dump_op_exec(ध्रुव std::string &sess_name, ध्रुव htp_opnode & गाठ, ध्रुव uint32_t req_flags) {
    यदि (!opt_verbose) प्रत्यागम;

    htp_opformat fmt(गाठ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                गाठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

स्थिर शून्य ggml_hexagon_dump_op_supp(ध्रुव std::string &sess_name, ध्रुव संरचना ggml_tensor * op, तार्किक supp) {
    यदि (!opt_verbose) प्रत्यागम;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

स्थिर ध्रुव वर्ण * htp_event_name(uint16_t id) {
    विकल्प (id) {
        पक्ष HTP_TRACE_EVT_DMA:            प्रत्यागम "DMA";
        पक्ष HTP_TRACE_EVT_HVX_COMP:       प्रत्यागम "HVX_COMP";
        पक्ष HTP_TRACE_EVT_HVX_A_QUANT:    प्रत्यागम "HVX_A_QUANT";
        पक्ष HTP_TRACE_EVT_HVX_A_PREP:     प्रत्यागम "HVX_A_PREP";
        पक्ष HTP_TRACE_EVT_HVX_W_DEQUANT:  प्रत्यागम "HVX_W_DEQUANT";
        पक्ष HTP_TRACE_EVT_HVX_W_PREP:     प्रत्यागम "HVX_W_PREP";
        पक्ष HTP_TRACE_EVT_HVX_O_PROC:     प्रत्यागम "HVX_O_PROC";
        पक्ष HTP_TRACE_EVT_HVX_FA_QK:      प्रत्यागम "HVX_QK_FA";
        पक्ष HTP_TRACE_EVT_HVX_FA_SFM:     प्रत्यागम "HVX_SFM_FA";
        पक्ष HTP_TRACE_EVT_HVX_FA_Q_PREP:  प्रत्यागम "HVX_Q_PREP";
        पक्ष HTP_TRACE_EVT_HVX_FA_K_PREP:  प्रत्यागम "HVX_K_PREP";
        पक्ष HTP_TRACE_EVT_HVX_FA_V_PREP:  प्रत्यागम "HVX_V_PREP";
        पक्ष HTP_TRACE_EVT_HVX_GDN_PREP:   प्रत्यागम "HVX_GDN_PREP";
        पक्ष HTP_TRACE_EVT_HVX_GDN_SOLVE:  प्रत्यागम "HVX_GDN_SOLVE";
        पक्ष HTP_TRACE_EVT_HVX_GDN_V_PREP: प्रत्यागम "HVX_GDN_V_PREP";
        पक्ष HTP_TRACE_EVT_HVX_GDN_D_PREP: प्रत्यागम "HVX_GDN_D_PREP";
        पक्ष HTP_TRACE_EVT_HVX_GDN_OUT:    प्रत्यागम "HVX_GDN_OUT";
        पक्ष HTP_TRACE_EVT_HVX_GDN_STATE:  प्रत्यागम "HVX_GDN_STATE";
        पक्ष HTP_TRACE_EVT_HVX_GDN_REM:    प्रत्यागम "HVX_GDN_REM";
        पक्ष HTP_TRACE_EVT_HMX_COMP:       प्रत्यागम "HMX_COMP";
        पक्ष HTP_TRACE_EVT_L2FLUSH:        प्रत्यागम "L2FLUSH";
        पक्ष HTP_TRACE_EVT_INIT:           प्रत्यागम "INIT";
        पक्ष HTP_TRACE_EVT_BUFF:           प्रत्यागम "BUFF";
        पक्ष HTP_TRACE_EVT_FENCE:          प्रत्यागम "FENCE";
        स्वभाव:                           प्रत्यागम "UNKNOWN";
    }
}

स्थिर शून्य ggml_hexagon_dump_op_prof(ध्रुव std::string &sess_name, ध्रुव htp_opnode & गाठ, ध्रुव htp_prof_desc & pd) {
    यदि (!opt_profile) प्रत्यागम;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    ध्रुव uint32_t * pmu = pd.pmu;

    वर्ण pmu_str[256] = "";
    यदि (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, परिमाण(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(गाठ);
    भिन्न mhz = op_usec > 0 ? (भिन्न) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            गाठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

स्थिर शून्य ggml_hexagon_dump_batch_prof(ध्रुव std::string & sess_name, ध्रुव htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    भिन्न batch_mhz = rsp.usecs > 0 ? (भिन्न) batch_cycles / rsp.usecs : 0.0f;

    वर्ण evt_str[256] = "----";
    यदि (opt_profile == 3) {
        snprintf(evt_str, परिमाण(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

स्थिर शून्य ggml_hexagon_dump_trace_events(ध्रुव std::string & sess_name, ध्रुव htp_opbatch_rsp & rsp,
                                           ध्रुव htp_trace_desc * trace_events, uint32_t n_traces) {
    यदि (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        प्रति (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        प्रति (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            प्रति (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ध्रुव auto & e = trace_events[t * n_traces + idx];
                तार्किक is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

गणना ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

संरचना ggml_hexagon_session;

स्थिर शून्य ggml_hexagon_precompute_matmul_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_matmul_add_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * src2,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_unary_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    uint32_t op,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_unary_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_द्विमान_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    uint32_t op,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_द्विमान_kernel_params * kparams,
    ध्रुव संरचना ggml_tensor * src2 = nullptr
);

स्थिर शून्य ggml_hexagon_precompute_get_rows_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_get_rows_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_set_rows_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_set_rows_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_softmax_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_softmax_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_rope_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_rope_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_ssm_conv_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_ssm_conv_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_gated_delta_net_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_gdn_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_sort_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    तार्किक is_शीर्ष_k,
    संरचना htp_sort_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_pool_2d_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_pool_2d_kernel_params * kparams,
    तार्किक is_pool_1d
);

स्थिर तार्किक ggml_hexagon_precompute_concat_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_concat_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_cpy_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_copy_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_im2col_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_im2col_3d_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_mmnx_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * act,
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_mmidnx_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * act,
    ध्रुव संरचना ggml_tensor * dst,
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_allreduce_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    तार्किक has_add,
    तार्किक is_row_bcast,
    तार्किक is_shard_ok,
    संरचना htp_allreduce_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_rows_stride(ध्रुव int64_t * ne, ध्रुव size_t * nb, size_t * stride);
स्थिर तार्किक ggml_hexagon_matmul_can_collapse(ध्रुव संरचना ggml_tensor * src0, ध्रुव संरचना ggml_tensor * src1, ध्रुव संरचना ggml_tensor * dst);
स्थिर ggml_tensor ggml_hexagon_tensor_collapse_rows(ध्रुव संरचना ggml_tensor * t);

स्थिर तार्किक mm_is_hmx_eligible(ध्रुव ggml_tensor * t);
स्थिर htp_op_code op_remap_to_htp(ध्रुव ggml_tensor * t);
स्थिर तार्किक is_supported_mul_mat_nx_kernel(ध्रुव ggml_tensor * src0, ध्रुव संरचना htp_mm_kernel_params * kparams);
स्थिर तार्किक is_supported_mul_mat_id_nx_kernel(ध्रुव ggml_tensor * src0, ध्रुव संरचना htp_mm_kernel_params * kparams);
स्थिर तार्किक is_mergeable_mul_mat(ध्रुव ggml_tensor * t);
स्थिर तार्किक is_mergeable_mul_mat_pair(ध्रुव ggml_tensor * n1, ध्रुव ggml_tensor * n2);
स्थिर तार्किक is_mergeable_mul_mat_id(ध्रुव ggml_tensor * t);
स्थिर तार्किक is_mergeable_mul_mat_id_pair(ध्रुव ggml_tensor * n1, ध्रुव ggml_tensor * n2);

// ** backend sessions

संरचना ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

स्थिर अन्तःस्थ तार्किक ggml_hexagon_tensor_is_fuseable(ध्रुव संरचना ggml_tensor * t) {
    यदि (!t->extra) प्रत्यागम असत्य;
    auto extra = (ध्रुव संरचना ggml_hexagon_tensor_extra *) t->extra;
    प्रत्यागम (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

स्थिर अन्तःस्थ तार्किक ggml_hexagon_can_rt_repack(ध्रुव संरचना ggml_tensor * t) {
    यदि (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        प्रत्यागम असत्य;
    }
    यदि (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

स्थिर अन्तःस्थ तार्किक ggml_hexagon_tensor_is_tiled(ध्रुव संरचना ggml_tensor * t) {
    ध्रुव auto * extra = (ध्रुव ggml_hexagon_tensor_extra *) t->extra;
    यदि (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        प्रत्यागम सत्य;
    }
    प्रत्यागम ggml_hexagon_can_rt_repack(t);
}

स्थिर अन्तःस्थ तार्किक ggml_hexagon_needs_repack(ध्रुव संरचना ggml_tensor * t) {
    ध्रुव auto * extra = (ध्रुव ggml_hexagon_tensor_extra *) t->extra;
    यदि (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम ggml_hexagon_can_rt_repack(t);
}

स्थिर अन्तःस्थ तार्किक ggml_hexagon_tensors_overlap(ध्रुव संरचना ggml_tensor * a, ध्रुव संरचना ggml_tensor * b) {
    ध्रुव uintptr_t a0 = (uintptr_t) a->data;
    ध्रुव uintptr_t b0 = (uintptr_t) b->data;
    ध्रुव uintptr_t a1 = a0 + ggml_nbytes(a);
    ध्रुव uintptr_t b1 = b0 + ggml_nbytes(b);

    प्रत्यागम a0 < b1 && b0 < a1;
}

स्थिर अन्तःस्थ तार्किक ggml_hexagon_can_row_partition(ध्रुव संरचना ggml_tensor * t) {
    यदि (t->ne[1] > 1 && (t->nb[1] & 127) != 0) प्रत्यागम असत्य;
    यदि (t->ne[2] > 1 && (t->nb[2] & 127) != 0) प्रत्यागम असत्य;
    यदि (t->ne[3] > 1 && (t->nb[3] & 127) != 0) प्रत्यागम असत्य;
    प्रत्यागम सत्य;
}

संरचना htp_opnode;

संरचना ggml_hexagon_opbatch;
संरचना ggml_hexagon_opqueue;
संरचना ggml_hexagon_shared_buffer;
संरचना ggml_hexagon_fence_buffer;
संरचना ggml_hexagon_session;
संरचना ggml_backend_hexagon_device_context;

संरचना ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

संरचना ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    चञ्चल uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

संरचना ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    चञ्चल uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

संरचना ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    पूर्णाङ्क              phys_idx;
    पूर्णाङ्क              virt_idx;
    तार्किक             valid_session;
    तार्किक             valid_handle;
    तार्किक             valid_queue;
    तार्किक             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<पूर्णाङ्क, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(ध्रुव ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_मोजणी = 0) noexcept(असत्य);
    ~ggml_hexagon_session() noexcept(सत्य);

    ध्रुव वर्ण* c_name() ध्रुव { प्रत्यागम name.c_str(); }

    शून्य allocate(ध्रुव ggml_hexagon_device_config & config) noexcept(असत्य);
    शून्य release() noexcept(सत्य);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    शून्य      free_fence(शून्य * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, चञ्चल uint32_t *> cpy_fence_slots;

    शून्य enqueue_mdev_group();
    शून्य enqueue_op(ध्रुव htp_opnode & गाठ);
    शून्य enqueue_cpy(ध्रुव ggml_tensor * src, ggml_tensor * dst, ध्रुव ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    शून्य enqueue_fence(ध्रुव ggml_tensor * sync_tensor, uint32_t fence_seq = 0, तार्किक wait = सत्य);
    शून्य enqueue_allreduce(ध्रुव ggml_tensor * dst, ध्रुव std::vector<ध्रुव ggml_tensor *> & src_tensors,
                           ध्रुव std::vector<ध्रुव ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    शून्य start_batch();
    शून्य flush_sync(तार्किक all = सत्य);
    शून्य flush_async();
    शून्य flush_batch(size_t min_ops = 1);
    शून्य flush_peers();
    शून्य flush_pending(तार्किक all = सत्य);

    ggml_hexagon_shared_buffer * mmap_tensor(ध्रुव ggml_tensor * t);
    तार्किक clone_buffer(ध्रुव ggml_hexagon_shared_buffer*);
    शून्य release_buffer(ध्रुव ggml_hexagon_shared_buffer*);
    शून्य unclone_buffer(ध्रुव ggml_hexagon_shared_buffer*);

    शून्य add_peer(ggml_hexagon_session * peer) {
        यदि (एतत्->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } अन्यथा {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

संरचना ggml_backend_hexagon_device_context {
    पूर्णाङ्क                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(पूर्णाङ्क dev_id, ध्रुव ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ध्रुव वर्ण * c_name() ध्रुव { प्रत्यागम config.name.c_str(); }

    ggml_hexagon_session * session() {
        यदि (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        प्रत्यागम sess.get();
    }
};

संरचना ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(ध्रुव std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        एतत्->dev_ctx = dev_ctx;
        एतत्->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

संरचना ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    पूर्णाङ्क       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        यदि (!base) {
            क्षिप std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        यदि (fd < 0) {
            rpcmem_free(base);
            क्षिप std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        एतत्->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        यदि (base) {
            rpcmem_free(base);
        }
    }
};

संरचना ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    तार्किक     mapped;
    तार्किक     pinned;
    तार्किक     extended;

    ध्रुव वर्ण * c_name() ध्रुव { प्रत्यागम sess->c_name(); }
    uint8_t *    base()   ध्रुव { प्रत्यागम mem ? mem->base : nullptr; }
    size_t       size()   ध्रुव { प्रत्यागम mem ? mem->size : 0;  }
    पूर्णाङ्क          fd()     ध्रुव { प्रत्यागम mem ? mem->fd   : -1; }

    शून्य mmap(तार्किक extended = असत्य) {
        यदि (!एतत्->mem)   प्रत्यागम;
        यदि (एतत्->mapped) प्रत्यागम;

        GGML_ASSERT(!एतत्->pinned || !extended);

        एतत्->extended = extended;
        fastrpc_map_flags flags = एतत्->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        पूर्णाङ्क err = fastrpc_mmap(sess->domain_id, fd(), (शून्य *) base(), 0, size(), flags);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            क्षिप std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (शून्य *) base(), size(), fd(), pinned, extended);

        एतत्->mapped = सत्य;
    }

    शून्य unmap() {
        यदि (!एतत्->mapped) प्रत्यागम;

        यदि (!एतत्->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        यदि (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (शून्य *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (शून्य *) base(), size(), fd());

        एतत्->mapped = असत्य;
    }

    शून्य alloc(size_t size) {
        यदि (एतत्->mem) प्रत्यागम;

        एतत्->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (शून्य *) base(), एतत्->size(), fd(), (पूर्णाङ्क) pinned);
        यदि (एतत्->pinned) {
            mmap();
        }
    }

    शून्य मुक्त() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (शून्य *) base(), size(), fd());
        एतत्->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, तार्किक pinned = असत्य) {
        एतत्->sess     = sess;
        एतत्->mapped   = असत्य;
        एतत्->pinned   = pinned;
        एतत्->extended = असत्य;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        यदि (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, ध्रुव ggml_hexagon_shared_buffer & other) {
        एतत्->sess   = sess;
        एतत्->mem    = other.mem;
        एतत्->mapped = असत्य;
        एतत्->pinned = other.pinned;
        एतत्->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        मुक्त();
        प्रति (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

संरचना ggml_hexagon_fence_buffer : सार्वजनिक ggml_hexagon_shared_buffer {
    uint32_t              slot_मोजणी = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, असत्य /* pinned */),
          slot_मोजणी(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(एतत्);
        backend_buffer.size    = size;
        mmap(असत्य);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        यदि (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } अन्यथा यदि (slot_head + n_slots <= slot_मोजणी) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        यदि (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        प्रत्यागम ptr;
    }

    शून्य free_slot(शून्य * ptr, uint32_t n_slots = 1) {
        यदि (!ptr) प्रत्यागम;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        प्रति (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

अन्तःस्थ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    प्रत्यागम ptr;
}

अन्तःस्थ शून्य ggml_hexagon_session::free_fence(शून्य * ptr, uint32_t n_slots) {
    यदि (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

स्थिर शून्य * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    प्रत्यागम sbuf->base();
}

स्थिर गणना ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (शून्य *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = नूतन ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   ध्रुव शून्य *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    यदि (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        यदि (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (पूर्णाङ्क) buffer->usage, extra->flags);

    यदि ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((वर्ण *) tensor->data + offset, data, size);
        प्रत्यागम;
    }

    यदि (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        प्रत्यागम;
    }

    यदि (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    यदि (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   ध्रुव ggml_tensor *   tensor,
                                                   शून्य *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (पूर्णाङ्क) buffer->usage, extra->flags);

    यदि ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (ध्रुव वर्ण *) tensor->data + offset, size);
        प्रत्यागम;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

स्थिर तार्किक ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   ध्रुव संरचना ggml_tensor * src,
                                                   संरचना ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    प्रत्यागम असत्य;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

स्थिर शून्य ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          ध्रुव शून्य *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    यदि (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        यदि (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (पूर्णाङ्क) buffer->usage, extra->flags);

    यदि ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (ध्रुव uint8_t *) data + i * stride_data, size);
        }
        प्रत्यागम;
    }

    यदि (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    प्रति (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (ध्रुव uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    यदि (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      ध्रुव ggml_tensor *   tensor,
                                                      शून्य *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (पूर्णाङ्क) buffer->usage);

    यदि ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        प्रति (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (ध्रुव uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        प्रत्यागम;
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

    प्रति (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (शून्य *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

स्थिर ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ रिक्त,
};

// ** backend buffer type

स्थिर शून्य ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        ध्रुव शून्य *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((वर्ण *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        ध्रुव ggml_tensor *   tensor,
                                                        शून्य *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (ध्रुव वर्ण *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

स्थिर ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ रिक्त,
};

// ** backend buffer type

स्थिर ध्रुव वर्ण * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    प्रत्यागम static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    यदि (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        प्रत्यागम nullptr;
    }
    यत्न {
        ggml_hexagon_shared_buffer * sbuf = नूतन ggml_hexagon_shared_buffer(sess, size, असत्य);
        प्रत्यागम ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } गृहाण (ध्रुव std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        प्रत्यागम nullptr;
    }
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    यदि (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        प्रत्यागम nullptr;
    }
    यत्न {
        ggml_hexagon_shared_buffer * sbuf = नूतन ggml_hexagon_shared_buffer(sess, size, असत्य);
        प्रत्यागम ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } गृहाण (ध्रुव std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        प्रत्यागम nullptr;
    }
}

स्थिर size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    प्रत्यागम 128;  // HVX alignment
    GGML_UNUSED(buft);
}

स्थिर size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ध्रुव संरचना ggml_tensor * t) {
    यदि (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        प्रत्यागम std::max(tiled_size, ggml_nbytes(t));
    }
    प्रत्यागम ggml_nbytes(t);
}

स्थिर size_t parse_size(ध्रुव वर्ण * str, size_t default_unit = 1024 * 1024) {
    यदि (!str || str[0] == '\0') {
        प्रत्यागम 0;
    }
    वर्ण * end = रिक्त;
    द्विगुण val = strtod(str, &end);
    यदि (val < 0) {
        प्रत्यागम 0;
    }
    यदि (end && *end) {
        यावत् (*end == ' ') end++;
        यदि (*end == 'k' || *end == 'K') {
            प्रत्यागम (size_t) (val * 1024);
        } अन्यथा यदि (*end == 'm' || *end == 'M') {
            प्रत्यागम (size_t) (val * 1024 * 1024);
        } अन्यथा यदि (*end == 'g' || *end == 'G') {
            प्रत्यागम (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    प्रत्यागम (size_t) (val * default_unit);
}

स्थिर size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    प्रत्यागम opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    प्रत्यागम असत्य;
    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    प्रत्यागम सत्य;
    GGML_UNUSED(buft);
}

संरचना ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    पूर्णाङ्क    first;
    पूर्णाङ्क    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

स्थिर ध्रुव वर्ण * ggml_hexagon_kv_layer_suffix(ध्रुव संरचना ggml_tensor * t) {
    यदि (strncmp(t->name, "cache_", 6) != 0) {
        प्रत्यागम रिक्त;
    }
    ध्रुव वर्ण * p = strstr(t->name, "_l");
    यदि (!p || !isdigit((unsigned वर्ण)p[2])) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम p;
}

संरचना ggml_backend_hexagon_alloc_unit {
    size_t size;
    पूर्णाङ्क    first;
    पूर्णाङ्क    last;
};

स्थिर ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    ध्रुव size_t alignment = ggml_backend_buft_get_alignment(buft);
    ध्रुव size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    पूर्णाङ्क i = 0;
    यावत् (i < n_tensors) {
        संरचना ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        पूर्णाङ्क unit_first = i;
        पूर्णाङ्क unit_last = i + 1;

        यदि (t->data == रिक्त && t->view_src == रिक्त) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        ध्रुव वर्ण * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        यावत् (unit_last < n_tensors) {
            संरचना ggml_tensor * next = tensors[unit_last];

            यदि (next->view_src != रिक्त) {
                unit_last++;
                अनुवर्त;
            }

            यदि (layer_suffix != रिक्त) {
                ध्रुव वर्ण * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                यदि (next_suffix != रिक्त && strcmp(layer_suffix, next_suffix) == 0) {
                    यदि (next->data == रिक्त) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    अनुवर्त;
                }
            }

            भङ्ग;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    पूर्णाङ्क    cur_buf_first = 0;

    प्रति (ध्रुव auto & unit : units) {
        यदि (unit.size == 0) {
            अनुवर्त;
        }

        यदि (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    यदि (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    प्रत्यागम plan;
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ध्रुव ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    प्रति (ध्रुव auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        यदि (buffer == रिक्त) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            प्रति (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            प्रत्यागम रिक्त;
        }

        संरचना ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        संरचना ggml_tensor * t_failed = रिक्त;
        प्रति (पूर्णाङ्क j = item.first; j < item.last; j++) {
            संरचना ggml_tensor * t = tensors[j];
            यदि (t->data == रिक्त) {
                यदि (t->view_src == रिक्त) {
                    यदि (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                } अन्यथा यदि (t->buffer == रिक्त) {
                    यदि (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                }
            } अन्यथा {
                यदि (t->view_src != रिक्त && t->buffer == रिक्त) {
                    यदि (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        भङ्ग;
                    }
                }
            }
        }
        यदि (t_failed != रिक्त) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            प्रति (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            प्रत्यागम रिक्त;
        }

        buffers.push_back(buffer);
    }

    यदि (buffers.empty()) {
        प्रत्यागम रिक्त;
    }

    यदि (buffers.size() == 1) {
        प्रत्यागम buffers[0];
    }

    प्रत्यागम ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

स्थिर size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors) {
    ध्रुव ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    प्रति (ध्रुव auto & item : plan) {
        total += item.size;
    }
    प्रत्यागम total;
}

स्थिर ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

स्थिर ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(पूर्णाङ्क dev_id, ध्रुव ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = नूतन ggml_backend_hexagon_buffer_type_context(config.name, एतत्);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = नूतन ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", एतत्);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = नूतन ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", एतत्);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

स्थिर तार्किक ggml_backend_buffer_is_hexagon(ध्रुव संरचना ggml_backend_buffer * b) {
    प्रत्यागम b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

संरचना ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<पूर्णाङ्क, पूर्णाङ्क>                b_map; // buffer fd   to index
    std::unordered_map<ध्रुव ggml_tensor*, पूर्णाङ्क> t_map; // tensor ptr  to index
    std::unordered_multimap<शून्य*, पूर्णाङ्क>         d_map; // tensor data to index

    unsigned पूर्णाङ्क n_bufs;     // num buffers in the batch
    unsigned पूर्णाङ्क n_tens;     // num tensors ...
    unsigned पूर्णाङ्क n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned पूर्णाङ्क n_bufs_max;
    unsigned पूर्णाङ्क n_tens_max;
    unsigned पूर्णाङ्क n_ops_max;
    size_t       b_vmem_max;

    शून्य reset() {
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
        एतत्->sess = sess;

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

    तार्किक empty() ध्रुव { प्रत्यागम n_ops == 0; }

    // add buffer and return its index
    पूर्णाङ्क add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        यदि (it != b_map.end()) { प्रत्यागम it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        पूर्णाङ्क bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        यदि (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (शून्य*) sbuf->base(), (size_t) b.size, b_vmem);

        प्रत्यागम bi;
    }

    तार्किक same_shape(ध्रुव htp_tensor * h, ध्रुव ggml_tensor * t) ध्रुव {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        ध्रुव तार्किक is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ध्रुव int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        ध्रुव int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? परिमाण(ggml_fp16_t) : t->nb[0];
        गणना ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        प्रत्यागम (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    पूर्णाङ्क add_tensor(ध्रुव ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        प्रति (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            यदि (same_shape(h, t)) { प्रत्यागम it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        यदि (it != t_map.end()) { प्रत्यागम it->second; }

        // Add new tensor to the batch
        पूर्णाङ्क ti = n_tens++;
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

        ध्रुव तार्किक is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        यदि (is_tiled) {
            यदि (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = परिमाण(ggml_fp16_t);
            } अन्यथा {
                h.nb[0] = t->nb[0];
            }
            ध्रुव int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            ध्रुव int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } अन्यथा {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        यदि ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        यदि ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        यदि ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (शून्य*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        प्रत्यागम ti;
    }

    तार्किक fit_op(ध्रुव htp_opnode & गाठ) ध्रुव {
        यदि (n_ops >= n_ops_max) प्रत्यागम असत्य;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        पूर्णाङ्क seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णाङ्क n_seen_bufs = 0;

        auto fit_tensor = [&](ध्रुव ggml_tensor *t) {
            यदि (!t) प्रत्यागम;
            यदि (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                पूर्णाङ्क fd = sbuf->fd();
                यदि (!b_map.count(fd)) {
                    प्रति (पूर्णाङ्क i = 0; i < n_seen_bufs; i++) {
                        यदि (seen_bufs[i] == fd) प्रत्यागम;
                    }
                    यदि (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    यदि (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        प्रति (ध्रुव auto * src : गाठ.get_inputs()) {
            fit_tensor(src);
        }
        प्रति (ध्रुव auto * output : गाठ.get_outputs()) {
            fit_tensor(output);
        }

        यदि ((extra_bufs + n_bufs) > n_bufs_max) प्रत्यागम असत्य;
        यदि ((extra_tens + n_tens) > n_tens_max) प्रत्यागम असत्य;
        यदि ((extra_vmem + b_vmem) > b_vmem_max) प्रत्यागम असत्य;

        प्रत्यागम सत्य;
    }

    // assumes that fit_op() was called first and returned true
    शून्य add_op(ध्रुव htp_opnode & गाठ) {
        // Add new op

        unsigned पूर्णाङ्क n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = गाठ;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        गाठ.गाठ->op_params, परिमाण(गाठ.गाठ->op_params));
        memcpy(o.kernel_params, गाठ.kernel_params,   परिमाण(o.kernel_params));
        o.opcode = गाठ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = गाठ.get_inputs();
        प्रति (unsigned पूर्णाङ्क i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = गाठ.get_outputs();
        प्रति (unsigned पूर्णाङ्क i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    शून्य sort_buffers() {
        यदि (n_bufs <= 1) प्रत्यागम;

        std::vector<पूर्णाङ्क> order(n_bufs);
        प्रति (unsigned पूर्णाङ्क i = 0; i < n_bufs; i++) { order[i] = (पूर्णाङ्क) i; }

        std::stable_sort(order.begin(), order.end(), [&](पूर्णाङ्क a, पूर्णाङ्क b) {
            प्रत्यागम h_bufs[a].size > h_bufs[b].size;
        });

        तार्किक already_sorted = सत्य;
        प्रति (unsigned पूर्णाङ्क i = 0; i < n_bufs; i++) {
            यदि (order[i] != (पूर्णाङ्क) i) {
                already_sorted = असत्य;
                भङ्ग;
            }
        }
        यदि (already_sorted) प्रत्यागम;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        प्रति (unsigned पूर्णाङ्क new_bi = 0; new_bi < n_bufs; new_bi++) {
            पूर्णाङ्क old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        प्रति (unsigned पूर्णाङ्क i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        प्रति (unsigned पूर्णाङ्क i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    शून्य update_mdev_group(uint32_t mdev_idx) {
        यदि (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    तार्किक try_fuse_common(std::initializer_list<ध्रुव ggml_tensor *> tensors) ध्रुव {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        पूर्णाङ्क seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णाङ्क n_seen_bufs = 0;

        प्रति (ध्रुव auto * t : tensors) {
            यदि (!t || t_map.count(t)) {
                अनुवर्त;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            पूर्णाङ्क fd = sbuf->fd();
            यदि (!b_map.count(fd)) {
                तार्किक found = असत्य;
                प्रति (पूर्णाङ्क i = 0; i < n_seen_bufs; i++) {
                    यदि (seen_bufs[i] == fd) {
                        found = सत्य;
                        भङ्ग;
                    }
                }
                यदि (!found) {
                    यदि (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    यदि (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        यदि ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            प्रत्यागम असत्य;
        }

        प्रत्यागम सत्य;
    }

    तार्किक try_fuse_common(ध्रुव ggml_tensor * t1, ध्रुव ggml_tensor * t2) ध्रुव {
        प्रत्यागम try_fuse_common({t1, t2});
    }

    तार्किक try_fuse_allreduce_add(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0 || opt_ar_select != 2) प्रत्यागम असत्य;
        यदि (गाठ.opcode != HTP_OP_ADD) प्रत्यागम असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        यदि (last_गाठ.opcode != HTP_OP_ALLREDUCE) प्रत्यागम असत्य;

        auto * ar_kparams = (संरचना htp_allreduce_kernel_params *) last_गाठ.kernel_params;
        ध्रुव uint32_t rank    = (uint32_t) ar_kparams->rank;
        ध्रुव uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ध्रुव ggml_tensor * ar_local = last_गाठ.inputs[rank];
        ध्रुव ggml_tensor * add_src0 = गाठ.src0();
        ध्रुव ggml_tensor * add_src1 = गाठ.src1();
        ध्रुव ggml_tensor * add_dst  = गाठ.dst();

        यदि (!ggml_hexagon_tensor_is_fuseable(ar_local)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * res_tensor;
        यदि (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } अन्यथा यदि (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } अन्यथा {
            प्रत्यागम असत्य;
        }

        यदि (ar_local->type != res_tensor->type) प्रत्यागम असत्य;

        ध्रुव तार्किक is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ध्रुव तार्किक is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        यदि (!is_same_shape && !is_row_bcast) प्रत्यागम असत्य;

        यदि (is_same_shape) {
            यदि (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                प्रत्यागम असत्य;
            }
            यदि (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                प्रत्यागम असत्य;
            }
        }
        यदि (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            प्रत्यागम असत्य;
        }

        // scatter is only valid for in-place add within max outputs
        ध्रुव तार्किक is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        संरचना htp_allreduce_kernel_params new_kparams;
        यदि (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, सत्य, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        प्रति (uint32_t r = 0; r < n_ranks; r++) {
            ध्रुव ggml_tensor * ar_src = last_गाठ.inputs[r];
            यदि (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                अनुवर्त;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            यदि (scatter_ok && r == rank) {
                अनुवर्त;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            प्रत्यागम असत्य;
        }

        यदि (!try_fuse_common(res_tensor, add_dst)) {
            प्रत्यागम असत्य;
        }

        last_गाठ.opcode = HTP_OP_ALLREDUCE_ADD;
        last_गाठ.name   = "ALLREDUCE+ADD";
        last_गाठ.inputs.push_back(res_tensor);
        last_गाठ.outputs.clear();
        last_गाठ.outputs.push_back(add_dst);
        last_गाठ.fused.push_back(गाठ.गाठ);
        memcpy(last_गाठ.kernel_params, &new_kparams, परिमाण(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, परिमाण(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        यदि (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            प्रति (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            प्रति (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } अन्यथा {
            o.dst[0] = add_tensor(add_dst);
            प्रति (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (पूर्णाङ्क) new_kparams.mode, (पूर्णाङ्क) new_kparams.n_dsts);
        प्रत्यागम सत्य;
    }

    तार्किक try_fuse_rms_norm_mul(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0) प्रत्यागम असत्य;
        यदि (गाठ.opcode != HTP_OP_MUL) प्रत्यागम असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        यदि (last_गाठ.opcode != HTP_OP_RMS_NORM) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * mul_src0 = गाठ.src0();
        ध्रुव ggml_tensor * mul_src1 = गाठ.src1();
        ध्रुव ggml_tensor * rms_out  = last_गाठ.dst();

        यदि (!ggml_hexagon_tensor_is_fuseable(rms_out)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * weight;
        यदि (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } अन्यथा यदि (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } अन्यथा {
            प्रत्यागम असत्य;
        }

        ध्रुव ggml_tensor * src0 = last_गाठ.src0();

        यदि (src0->ne[0] != weight->ne[0] || src0->ne[0] != गाठ.dst()->ne[0]) {
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        ध्रुव तार्किक is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        यदि (!is_row_bcast && !is_same_shape) प्रत्यागम असत्य;

        यदि (!ggml_are_same_shape(src0, गाठ.dst())) {
            प्रत्यागम असत्य;
        }
        यदि (ggml_is_contiguous(src0) != ggml_is_contiguous(गाठ.dst())) {
            प्रत्यागम असत्य;
        }

        संरचना htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, गाठ.dst(), &new_kparams
        );

        यदि ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            प्रत्यागम असत्य;
        }

        यदि (!try_fuse_common(weight, गाठ.dst())) {
            प्रत्यागम असत्य;
        }

        last_गाठ.opcode = HTP_OP_RMS_NORM_MUL;
        last_गाठ.name   = "RMS_NORM+MUL";
        last_गाठ.inputs.clear();
        last_गाठ.inputs.push_back(src0);
        last_गाठ.inputs.push_back(weight);
        last_गाठ.outputs.clear();
        last_गाठ.outputs.push_back(गाठ.dst());
        last_गाठ.fused.push_back(गाठ.गाठ);
        memcpy(last_गाठ.kernel_params, &new_kparams, परिमाण(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, परिमाण(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        प्रति (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाठ.dst());
        प्रति (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        प्रत्यागम सत्य;
    }

    तार्किक try_fuse_mul_mat_add(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0) प्रत्यागम असत्य;
        यदि (गाठ.opcode != HTP_OP_ADD) प्रत्यागम असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        यदि (last_गाठ.opcode != HTP_OP_MUL_MAT) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * add_src0 = गाठ.src0();
        ध्रुव ggml_tensor * add_src1 = गाठ.src1();
        ध्रुव ggml_tensor * mm_out   = last_गाठ.dst();

        यदि (!ggml_hexagon_tensor_is_fuseable(mm_out)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * src2;
        यदि (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } अन्यथा यदि (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } अन्यथा {
            प्रत्यागम असत्य;
        }

        ध्रुव ggml_tensor * src0 = last_गाठ.src0();
        ध्रुव ggml_tensor * src1 = last_गाठ.src1();

        यदि (src2->type != GGML_TYPE_F32) प्रत्यागम असत्य;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        यदि (!ggml_are_same_shape(mm_out, गाठ.dst())) प्रत्यागम असत्य;

        ध्रुव संरचना htp_mm_kernel_params * orig_kparams = (ध्रुव संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
        संरचना htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, गाठ.dst(), &kparams);
        यदि (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            प्रत्यागम असत्य;
        }

        यदि (orig_kparams->collapse != kparams.collapse) {
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ध्रुव तार्किक can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        यदि (!can_fuse) प्रत्यागम असत्य;

        यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            प्रत्यागम असत्य;
        }

        यदि (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            यदि (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                प्रत्यागम असत्य;
            }
        }

        यदि (!try_fuse_common(src2, गाठ.dst())) {
            प्रत्यागम असत्य;
        }

        last_गाठ.opcode = HTP_OP_MUL_MAT_ADD;
        last_गाठ.name   = "MUL_MAT+ADD";
        last_गाठ.inputs.clear();
        last_गाठ.inputs.push_back(src0);
        last_गाठ.inputs.push_back(src1);
        last_गाठ.inputs.push_back(src2);
        last_गाठ.outputs.clear();
        last_गाठ.outputs.push_back(गाठ.dst());
        last_गाठ.fused.push_back(गाठ.गाठ);
        memcpy(last_गाठ.kernel_params, &kparams, परिमाण(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, परिमाण(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        प्रति (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाठ.dst());
        प्रति (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        प्रत्यागम सत्य;
    }

    तार्किक try_fuse_mul_mat_nx(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0 || गाठ.opcode != HTP_OP_MUL_MAT) प्रत्यागम असत्य;
        यदि (!is_mergeable_mul_mat(गाठ.गाठ)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * w_in = गाठ.src0();
        ध्रुव ggml_tensor * x_in = गाठ.src1();
        ध्रुव ggml_tensor * d_in = गाठ.dst();

        htp_opnode & last_गाठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        यदि (last_गाठ.opcode == HTP_OP_MUL_MAT_NX) {
            ध्रुव uint32_t curr_n = (uint32_t) last_गाठ.outputs.size();
            यदि (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor * w0 = last_गाठ.inputs[0];
            ध्रुव ggml_tensor * x  = last_गाठ.inputs[curr_n];

            यदि (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                प्रत्यागम असत्य;
            }
            यदि (!last_गाठ.fused.empty() && (mm_is_hmx_eligible(last_गाठ.fused[0]) != mm_is_hmx_eligible(गाठ.गाठ))) {
                प्रत्यागम असत्य;
            }

            ध्रुव संरचना htp_mm_kernel_params * orig_kparams = (ध्रुव संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
            ध्रुव तार्किक collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            यदि (orig_kparams->collapse && !collapse) {
                प्रत्यागम असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            यदि (collapse) {
                ध्रुव ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            यदि (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                प्रत्यागम असत्य;
            }
            यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                प्रत्यागम असत्य;
            }

            यदि (!try_fuse_common(w_in, d_in)) {
                प्रत्यागम असत्य;
            }

            last_गाठ.inputs[curr_n] = w_in;
            last_गाठ.inputs.push_back(x);
            last_गाठ.outputs.push_back(d_in);
            last_गाठ.fused.push_back(गाठ.गाठ);
            memcpy(last_गाठ.kernel_params, &kparams, परिमाण(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, परिमाण(kparams));

            प्रति (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_गाठ.inputs[s]);
            }
            प्रति (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            प्रति (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाठ.outputs[d]);
            }
            प्रति (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            प्रत्यागम सत्य;
        }

        // Case 2: last_node is single MUL_MAT
        यदि (last_गाठ.opcode == HTP_OP_MUL_MAT) {
            यदि (!is_mergeable_mul_mat_pair(last_गाठ.गाठ, गाठ.गाठ)) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor * w0 = last_गाठ.src0();
            ध्रुव ggml_tensor * x  = last_गाठ.src1();
            ध्रुव ggml_tensor * w1 = गाठ.src0();
            ध्रुव ggml_tensor * dst_0 = last_गाठ.dst();
            ध्रुव ggml_tensor * dst_1 = गाठ.dst();

            ध्रुव संरचना htp_mm_kernel_params * orig_kparams = (ध्रुव संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
            ध्रुव तार्किक collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            यदि (orig_kparams->collapse && !collapse) {
                प्रत्यागम असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            यदि (collapse) {
                ध्रुव ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            यदि (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                प्रत्यागम असत्य;
            }
            यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                प्रत्यागम असत्य;
            }

            यदि (!try_fuse_common(w1, गाठ.dst())) {
                प्रत्यागम असत्य;
            }

            last_गाठ.opcode = HTP_OP_MUL_MAT_NX;
            last_गाठ.name   = "MUL_MAT_NX";
            last_गाठ.inputs.clear();
            last_गाठ.inputs.push_back(w0);
            last_गाठ.inputs.push_back(w1);
            last_गाठ.inputs.push_back(x);
            last_गाठ.outputs.clear();
            last_गाठ.outputs.push_back(dst_0);
            last_गाठ.outputs.push_back(dst_1);
            last_गाठ.fused.push_back(गाठ.गाठ);
            memcpy(last_गाठ.kernel_params, &kparams, परिमाण(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, परिमाण(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            प्रति (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            प्रति (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            प्रत्यागम सत्य;
        }

        प्रत्यागम असत्य;
    }

    तार्किक try_fuse_mul_mat_id_nx(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0 || गाठ.opcode != HTP_OP_MUL_MAT_ID) प्रत्यागम असत्य;
        यदि (!is_mergeable_mul_mat_id(गाठ.गाठ)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * w_in   = गाठ.src0();
        ध्रुव ggml_tensor * x_in   = गाठ.src1();
        ध्रुव ggml_tensor * ids_in = गाठ.गाठ->src[2];
        ध्रुव ggml_tensor * d_in   = गाठ.dst();

        htp_opnode & last_गाठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        यदि (last_गाठ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ध्रुव uint32_t curr_n = (uint32_t) last_गाठ.outputs.size();
            यदि (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor * w0  = last_गाठ.inputs[0];
            ध्रुव ggml_tensor * x   = last_गाठ.inputs[curr_n];
            ध्रुव ggml_tensor * ids = last_गाठ.inputs[curr_n + 1];

            यदि (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                प्रत्यागम असत्य;
            }
            यदि (!last_गाठ.fused.empty() && (mm_is_hmx_eligible(last_गाठ.fused[0]) != mm_is_hmx_eligible(गाठ.गाठ))) {
                प्रत्यागम असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            यदि (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                प्रत्यागम असत्य;
            }
            यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                प्रत्यागम असत्य;
            }

            यदि (!try_fuse_common(w_in, d_in)) {
                प्रत्यागम असत्य;
            }

            last_गाठ.inputs[curr_n] = w_in;
            last_गाठ.inputs[curr_n + 1] = x;
            last_गाठ.inputs.push_back(ids);
            last_गाठ.outputs.push_back(d_in);
            last_गाठ.fused.push_back(गाठ.गाठ);
            memcpy(last_गाठ.kernel_params, &kparams, परिमाण(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, परिमाण(kparams));

            प्रति (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_गाठ.inputs[s]);
            }
            प्रति (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            प्रति (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाठ.outputs[d]);
            }
            प्रति (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            प्रत्यागम सत्य;
        }

        // Case 2: last_node is single MUL_MAT_ID
        यदि (last_गाठ.opcode == HTP_OP_MUL_MAT_ID) {
            यदि (!is_mergeable_mul_mat_id_pair(last_गाठ.गाठ, गाठ.गाठ)) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor * w0  = last_गाठ.src0();
            ध्रुव ggml_tensor * x   = last_गाठ.src1();
            ध्रुव ggml_tensor * ids = last_गाठ.गाठ->src[2];
            ध्रुव ggml_tensor * w1  = गाठ.src0();

            संरचना htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, गाठ.dst(), 2, &kparams);
            यदि (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                प्रत्यागम असत्य;
            }
            यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                प्रत्यागम असत्य;
            }

            यदि (!try_fuse_common(w1, गाठ.dst())) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor * dst_0 = last_गाठ.dst();
            ध्रुव ggml_tensor * dst_1 = गाठ.dst();

            last_गाठ.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_गाठ.name   = "MUL_MAT_ID_NX";
            last_गाठ.inputs.clear();
            last_गाठ.inputs.push_back(w0);
            last_गाठ.inputs.push_back(w1);
            last_गाठ.inputs.push_back(x);
            last_गाठ.inputs.push_back(ids);
            last_गाठ.outputs.clear();
            last_गाठ.outputs.push_back(dst_0);
            last_गाठ.outputs.push_back(dst_1);
            last_गाठ.fused.push_back(गाठ.गाठ);
            memcpy(last_गाठ.kernel_params, &kparams, परिमाण(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, परिमाण(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            प्रति (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            प्रति (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            प्रत्यागम सत्य;
        }

        प्रत्यागम असत्य;
    }

    तार्किक try_fuse_gdn_cpy(ध्रुव htp_opnode & गाठ) {
        यदि (n_ops == 0 || गाठ.opcode != HTP_OP_CPY) प्रत्यागम असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        यदि (last_गाठ.opcode != HTP_OP_GATED_DELTA_NET) प्रत्यागम असत्य;
        यदि (last_गाठ.outputs.size() != 1) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * gdn_out  = last_गाठ.dst();
        ध्रुव ggml_tensor * cpy_गाठ = गाठ.गाठ;
        ध्रुव ggml_tensor * cpy_src  = गाठ.src0();
        ध्रुव ggml_tensor * cpy_dst  = गाठ.dst();

        यदि (!cpy_src || !cpy_dst || !cpy_dst->data) प्रत्यागम असत्य;
        यदि (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) प्रत्यागम असत्य;
        यदि ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_गाठ->flags & GGML_TENSOR_FLAG_OUTPUT)) प्रत्यागम असत्य;

        ध्रुव ggml_tensor * v = last_गाठ.गाठ->src[2];
        यदि (!v) प्रत्यागम असत्य;

        ध्रुव int64_t S_v      = v->ne[0];
        ध्रुव int64_t H        = v->ne[1];
        ध्रुव int64_t n_tokens = v->ne[2];
        ध्रुव int64_t n_seqs   = v->ne[3];
        ध्रुव int64_t K        = ggml_get_op_params_i32(last_गाठ.गाठ, 0);
        ध्रुव size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * परिमाण(भिन्न);

        ध्रुव int64_t D         = S_v * S_v * H;
        ध्रुव int64_t n_written = std::min<int64_t>(n_tokens, K);

        यदि (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            प्रत्यागम असत्य;
        }

        यदि (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != परिमाण(भिन्न)) {
            प्रत्यागम असत्य;
        }
        यदि (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * परिमाण(भिन्न)) {
            प्रत्यागम असत्य;
        }
        यदि (n_written > 1) {
            यदि (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * परिमाण(भिन्न)) {
                प्रत्यागम असत्य;
            }
        }

        यदि (!try_fuse_common({cpy_dst})) {
            प्रत्यागम असत्य;
        }

        last_गाठ.name += "+CPY";
        last_गाठ.outputs.push_back(cpy_dst);
        last_गाठ.fused.push_back(गाठ.गाठ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        प्रति (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        प्रत्यागम सत्य;
    }

    तार्किक try_fuse(ध्रुव htp_opnode & गाठ) {
        यदि (!opt_opfusion) प्रत्यागम असत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(गाठ)) प्रत्यागम सत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(गाठ))  प्रत्यागम सत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(गाठ))   प्रत्यागम सत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(गाठ))    प्रत्यागम सत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(गाठ)) प्रत्यागम सत्य;
        यदि (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(गाठ))       प्रत्यागम सत्य;
        प्रत्यागम असत्य;
    }
};

संरचना ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

संरचना ggml_hexagon_opqueue {
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
        यदि (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * परिमाण(htp_trace_desc);
        }

        shm_blk_size = परिमाण(htp_buf_desc)  * n_bufs    +
                       परिमाण(htp_tensor)    * n_tensors +
                       परिमाण(htp_op_desc)   * n_ops     +
                       परिमाण(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = नूतन ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, सत्य /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        यदि (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() ध्रुव { प्रत्यागम shm_buf ? shm_buf->size() : 0; }

    // push new batch
    तार्किक ढकला(htp_opbatch_req& req, dspqueue_buffer& dbuf, ध्रुव ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(परिमाण(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(परिमाण(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(परिमाण(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(परिमाण(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(परिमाण(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(परिमाण(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        यदि (seq - shm_buf->sess->batch_rsp_seq > depth) { प्रत्यागम असत्य; }

        ध्रुव uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        ध्रुव size_t b_size = परिमाण(htp_buf_desc)  * req.n_bufs;
        ध्रुव size_t t_size = परिमाण(htp_tensor)    * req.n_tensors;
        ध्रुव size_t o_size = परिमाण(htp_op_desc)   * req.n_ops;
        ध्रुव size_t p_size = परिमाण(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        यदि (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * परिमाण(htp_trace_desc);
        } अन्यथा {
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

        memcpy(b_ptr, (शून्य *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (शून्य *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (शून्य *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        यदि (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            प्रति (unsigned पूर्णाङ्क i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (शून्य *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            प्रति (unsigned पूर्णाङ्क i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        प्रत्यागम सत्य;
    }

    शून्य काढा(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        ध्रुव uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        ध्रुव size_t b_size = परिमाण(htp_buf_desc)  * rsp.n_bufs;
        ध्रुव size_t t_size = परिमाण(htp_tensor)    * rsp.n_tensors;
        ध्रुव size_t o_size = परिमाण(htp_op_desc)   * rsp.n_ops;
        ध्रुव size_t p_size = परिमाण(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        यदि (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * परिमाण(htp_trace_desc);
        }

        ध्रुव size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        यदि (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ध्रुव htp_prof_desc * pd = (ध्रुव htp_prof_desc *) p_ptr;
            ध्रुव htp_trace_desc * trace_events = nullptr;
            यदि (opt_profile == 3) {
                trace_events = (ध्रुव htp_trace_desc *) (p_ptr + p_size);
            }

            यदि (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            प्रति (uint32_t i = 0; i < rsp.n_ops; i++) {
                यदि (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            यदि (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

शून्य ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    प्रति (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    प्रति (auto * peer : ppeers) {
        peer->flush_async();
    }

    प्रति (auto & sub : एतत्->mdev.sessions) {
        sub->flush_peers();
    }
}

शून्य ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

शून्य ggml_hexagon_session::flush_pending(तार्किक all) {
    प्रति (auto & sub : एतत्->mdev.sessions) {
        sub->flush_pending(all);
        यदि (sub->last_error > HTP_STATUS_OK) {
            एतत्->last_error = sub->last_error.load();
        }
    }

    यावत् (एतत्->batch_rsp_seq < एतत्->batch_req_seq) {
        संरचना htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        संरचना dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ध्रुव uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        पूर्णाङ्क err = dspqueue_read(एतत्->queue, &flags, 1, &n_dbufs, &dbuf, परिमाण(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        यदि (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            अनुवर्त;
        }

        यदि (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        यदि (rsp_size != परिमाण(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", एतत्->c_name(), rsp_size, n_dbufs);
        }

        यदि (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", एतत्->c_name(), status_to_str(rsp.status));
            एतत्->last_error = rsp.status;
            प्रति (auto & sub : एतत्->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->काढा(rsp, dbuf);

        GGML_ASSERT(rsp.seq == एतत्->batch_rsp_seq + 1);
        एतत्->batch_rsp_seq = rsp.seq;

        यदि (!all) भङ्ग;
    }
}

शून्य ggml_hexagon_session::flush_sync(तार्किक all) {
    flush_async();
    flush_pending(all);
}

शून्य ggml_hexagon_session::start_batch() {
    यदि (एतत्->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

शून्य ggml_hexagon_session::flush_batch(size_t min_ops) {
    यदि (op_batch->n_ops < min_ops) { प्रत्यागम; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    ध्रुव uint64_t seq = ++एतत्->batch_req_seq;

    op_batch->update_mdev_group(एतत्->mdev.idx);

    यदि (!op_queue->ढकला(req, dbuf, op_batch, seq)) {
        flush_pending(असत्य);
        op_queue->ढकला(req, dbuf, op_batch, seq);
    }

    प्रति (auto & sub : एतत्->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        यदि (!sub->op_queue->ढकला(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(असत्य);
            sub->op_queue->ढकला(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        पूर्णाङ्क err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, परिमाण(sub_req), (ध्रुव uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        यदि (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", एतत्->c_name(), dbuf.ptr, dbuf.size);

    पूर्णाङ्क err = dspqueue_write(एतत्->queue, 0, 1, &dbuf, परिमाण(req), (ध्रुव uint8_t*) &req, DSPQUEUE_TIMEOUT);
    यदि (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", एतत्->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

शून्य ggml_hexagon_session::enqueue_op(ध्रुव htp_opnode & गाठ) {
    auto clone_tensor_buffer = [एतत्](ध्रुव ggml_tensor * t) {
        auto sbuf = एतत्->mmap_tensor(t);
        यदि (!sbuf) प्रत्यागम;
        यदि (sbuf->sess != एतत्) {
            एतत्->clone_buffer(sbuf);
        }
        प्रति (auto & sub : एतत्->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    प्रति (auto t : गाठ.get_inputs()) {
        clone_tensor_buffer(t);
    }
    प्रति (auto t : गाठ.get_outputs()) {
        clone_tensor_buffer(t);
    }

    यदि (opt_opfusion && op_batch->try_fuse(गाठ)) {
        प्रत्यागम;
    }

    यदि (!op_batch->fit_op(गाठ)) {
        flush_async();
    }

    यदि (op_batch->empty()) {
        start_batch();
    }

    यदि (!op_batch->fit_op(गाठ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(गाठ);
}

शून्य ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_गाठ(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = एतत्->mdev_fence_slot;

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &एतत्->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (शून्य *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) एतत्->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) एतत्->mdev.idx;

    ggml_tensor * गाठ = group_गाठ.add_dummy(dummy_t);
    गाठ->src[0] = गाठ;
    group_गाठ.init(गाठ);
    group_गाठ.outputs.clear();
    group_गाठ.name = "MDEV_GROUP";

    प्रति (auto & sub : एतत्->mdev.sessions) {
        sub->clone_buffer(एतत्->fence_buf);
    }

    op_batch->add_op(group_गाठ);
}

शून्य ggml_hexagon_session::enqueue_cpy(ध्रुव ggml_tensor * src, ggml_tensor * dst, ध्रुव ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ध्रुव तार्किक with_fence = sync_tensor != nullptr;
    htp_opnode cpy_गाठ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* गाठ = cpy_गाठ.add_dummy(*dst);
    गाठ->op     = GGML_OP_CPY;
    गाठ->src[0] = const_cast<ggml_tensor *>(src);
    गाठ->src[1] = with_fence ? cpy_गाठ.add_dummy(*sync_tensor) : nullptr;
    यदि (with_fence) {
        गाठ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_गाठ.init(गाठ);
    यदि (with_fence) {
        cpy_गाठ.name = "CPY+FENCE";
    }
    ध्रुव तार्किक ok = ggml_hexagon_precompute_cpy_params(एतत्, गाठ, (संरचना htp_copy_kernel_params *) cpy_गाठ.kernel_params);
    ध्रुव auto * kparams = (ध्रुव संरचना htp_copy_kernel_params *) cpy_गाठ.kernel_params;
    यदि (ok && !with_fence && kparams->total_elems == 0) {
        प्रत्यागम;
    }
    एतत्->enqueue_op(cpy_गाठ);
}

शून्य ggml_hexagon_session::enqueue_fence(ध्रुव ggml_tensor * sync_tensor, uint32_t fence_seq, तार्किक wait) {
    htp_opnode sync_गाठ(HTP_OP_FENCE);

    ggml_tensor* गाठ = sync_गाठ.add_dummy(*sync_tensor);
    गाठ->op           = GGML_OP_NONE;
    गाठ->src[0]       = गाठ;
    गाठ->op_params[0] = (int32_t) fence_seq;
    गाठ->op_params[1] = wait ? 0 : 1;

    sync_गाठ.init(गाठ);
    sync_गाठ.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    एतत्->enqueue_op(sync_गाठ);
}

स्थिर तार्किक ggml_hexagon_precompute_allreduce_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    तार्किक has_add,
    तार्किक is_row_bcast,
    तार्किक is_shard_ok,
    संरचना htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    ध्रुव uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ध्रुव uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? परिमाण(ggml_fp16_t) : परिमाण(भिन्न);
    ध्रुव तार्किक is_contiguous = ggml_is_contiguous(dst);

    ध्रुव uint32_t ne0 = (uint32_t) dst->ne[0];
    ध्रुव uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ध्रुव तार्किक use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    यदि (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        यदि (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            ध्रुव uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ध्रुव uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ध्रुव uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } अन्यथा यदि (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } अन्यथा {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } अन्यथा {
        kparams->n_dsts = (int32_t) n_ranks;
        यदि (use_1d) {
            ध्रुव uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ध्रुव uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ध्रुव uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            ध्रुव uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } अन्यथा {
            ध्रुव uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            ध्रुव uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            ध्रुव uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            ध्रुव uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    यदि (use_1d) {
        ध्रुव uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        ध्रुव size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_मोजणी(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        यदि (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        यावत् ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            ध्रुव size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            यदि (block_elems < 128) भङ्ग;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        यदि (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            प्रत्यागम असत्य;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        प्रत्यागम सत्य;
    } अन्यथा {
        ध्रुव uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        ध्रुव size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_मोजणी(n_ranks, n_threads, has_add, is_row_bcast);

        ध्रुव uint32_t row_bytes = ne0 * elem_size;
        ध्रुव uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        ध्रुव uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        यावत् ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            ध्रुव size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            यदि (max_rows_per_buf == 0) भङ्ग;
        }

        यदि (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            प्रत्यागम असत्य;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        प्रत्यागम सत्य;
    }
}

शून्य ggml_hexagon_session::enqueue_allreduce(
    ध्रुव ggml_tensor * dst,
    ध्रुव std::vector<ध्रुव ggml_tensor *> & src_tensors,
    ध्रुव std::vector<ध्रुव ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_गाठ(HTP_OP_ALLREDUCE);

    ggml_tensor* गाठ = ar_गाठ.add_dummy(*dst);
    गाठ->op           = GGML_OP_NONE;
    गाठ->op_params[0] = (int32_t) fence_seq_entry;
    गाठ->op_params[1] = (int32_t) fence_seq_exit;

    ar_गाठ.init(गाठ);

    ar_गाठ.inputs.clear();
    प्रति (size_t i = 0; i < src_tensors.size(); i++) {
        ar_गाठ.inputs.push_back(src_tensors[i]);
    }
    प्रति (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_गाठ.inputs.push_back(ar_गाठ.add_dummy(*sync_tensors[i]));
    }

    ar_गाठ.outputs.clear();
    प्रति (size_t i = 0; i < src_tensors.size(); i++) {
        ar_गाठ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        एतत्, dst, rank, n_ranks, असत्य, असत्य, /*is_shard_ok=*/ असत्य,
        (संरचना htp_allreduce_kernel_params *) ar_गाठ.kernel_params);

    ar_गाठ.name = "ALLREDUCE";
    एतत्->enqueue_op(ar_गाठ);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(ध्रुव ggml_tensor * t) {
    यदि (!t) प्रत्यागम nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    यदि (!sbuf->mapped) {
        ध्रुव तार्किक is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ध्रुव तार्किक extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    प्रत्यागम sbuf;
}

तार्किक ggml_hexagon_session::clone_buffer(ध्रुव ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    यदि (sbuf->sess == एतत्) प्रत्यागम सत्य;

    auto mem = sbuf->mem;
    पूर्णाङ्क   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    यदि (एतत्->cloned_buffers.find(fd) != एतत्->cloned_buffers.end()) प्रत्यागम सत्य;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", एतत्->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(एतत्, *sbuf);
    यत्न {
        clone->mmap(sbuf->extended);
    } गृहाण (ध्रुव std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", एतत्->c_name(), exc.what());
        प्रत्यागम असत्य;
    }

    एतत्->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(एतत्);
    प्रत्यागम सत्य;
}

शून्य ggml_hexagon_session::release_buffer(ध्रुव ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    पूर्णाङ्क   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = एतत्->cloned_buffers.find(fd);
    यदि (it != एतत्->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        एतत्->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(एतत्);
}

शून्य ggml_hexagon_session::unclone_buffer(ध्रुव ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    प्रति (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

स्थिर size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    ध्रुव size_t MiB = 1024 * 1024;
    ध्रुव size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    यत्न {
        sbufs.push_back(नूतन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;
        sbufs.push_back(नूतन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;
        sbufs.push_back(नूतन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;

        यावत् (1) {
            sbufs.push_back(नूतन ggml_hexagon_shared_buffer(sess, step, सत्य));
            vmem += step;
        }
    } गृहाण (...) { }

    प्रति (auto b : sbufs) { delete b; }

    प्रत्यागम vmem - step; // backoff to account for overhead from internal mappings
}

शून्य ggml_hexagon_session::allocate(ध्रुव ggml_hexagon_device_config & config) noexcept(असत्य) {
    पूर्णाङ्क phys_idx = config.physical_idx;
    पूर्णाङ्क virt_idx = config.virtual_idx;

    एतत्->valid_session = असत्य;
    एतत्->valid_handle  = असत्य;
    एतत्->valid_queue   = असत्य;
    एतत्->valid_iface   = असत्य;

    एतत्->name          = config.name;
    एतत्->phys_idx      = phys_idx;
    एतत्->virt_idx      = virt_idx;
    एतत्->domain_id     = config.domain_id;
    एतत्->session_id    = 0;
    एतत्->batch_req_seq = 0;
    एतत्->batch_rsp_seq = 0;
    एतत्->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", एतत्->name.c_str(), एतत्->domain_id, phys_idx, virt_idx);

    यदि (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        क्षिप std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ध्रुव std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    यदि (virt_idx > 0) {
        संरचना remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<वर्ण *>(dom_name.c_str());
        n.session_name     = const_cast<वर्ण *>(एतत्->name.c_str());
        n.session_name_len = एतत्->name.size();
        n.session_id       = virt_idx;

        पूर्णाङ्क err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (शून्य *) &n, परिमाण(n));
        यदि (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           एतत्->c_name(), phys_idx, virt_idx, err);
            क्षिप std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        एतत्->session_id    = n.session_id;
        एतत्->domain_id     = n.effective_domain_id;
        एतत्->valid_session = सत्य;
    } अन्यथा {
        संरचना remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<वर्ण *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        पूर्णाङ्क err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (शून्य *) &eff, परिमाण(eff));
        यदि (err == AEE_SUCCESS) {
            एतत्->domain_id = eff.effective_domain_id;
        } अन्यथा {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           एतत्->name.c_str(), err, एतत्->domain_id);
        }
    }

    // Enable unsigned modules
    {
        संरचना remote_rpc_control_unsigned_module u;
        u.domain = एतत्->domain_id;
        u.enable = 1;
        पूर्णाङ्क err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (शून्य *) &u, परिमाण(u));
        यदि (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", एतत्->c_name(), err);
            क्षिप std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    वर्ण session_uri[256];
    {
        वर्ण htp_uri[256];
        snprintf(htp_uri, परिमाण(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        संरचना remote_rpc_get_uri u = {};
        u.session_id      = एतत्->session_id;
        u.domain_name     = const_cast<वर्ण *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<वर्ण *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = परिमाण(session_uri);

        पूर्णाङ्क err = remote_session_control(FASTRPC_GET_URI, (शून्य *) &u, परिमाण(u));
        यदि (err != AEE_SUCCESS) {
            snprintf(session_uri, परिमाण(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), एतत्->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          एतत्->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    पूर्णाङ्क err = htp_iface_open(session_uri, &एतत्->handle);
    यदि (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", एतत्->c_name(), session_uri, err);
        क्षिप std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    एतत्->valid_handle = सत्य;

    // Query HW info and resolve session options
    {
        unsigned पूर्णाङ्क hw_n_threads = 0;
        unsigned पूर्णाङ्क hw_n_hvx     = 0;
        unsigned पूर्णाङ्क hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        पूर्णाङ्क hw_err = htp_iface_hwinfo(एतत्->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        यदि (hw_err == 0) {
            ध्रुव uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            एतत्->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            एतत्->n_hvx     = एतत्->n_threads;
            एतत्->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            एतत्->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          एतत्->c_name(), एतत्->n_threads, एतत्->n_hvx, एतत्->n_hmx,
                          (unsigned long long)(एतत्->vtcm_size / (1024 * 1024)));
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", एतत्->c_name(), hw_err);
            ध्रुव uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            एतत्->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            एतत्->n_hvx     = एतत्->n_threads;
            एतत्->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            एतत्->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        संरचना remote_rpc_control_latency l;
        l.enable = 1;

        पूर्णाङ्क err = remote_handle64_control(एतत्->handle, DSPRPC_CONTROL_LATENCY, (शून्य *) &l, परिमाण(l));
        यदि (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", एतत्->c_name(),
                  एतत्->session_id, एतत्->domain_id, session_uri, (unsigned long) एतत्->handle);

    ध्रुव size_t req_q_size = (परिमाण(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ध्रुव size_t rsp_q_size = (परिमाण(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(एतत्->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (शून्य *) एतत्,  // Callback context
                          &queue);
    यदि (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", एतत्->name.c_str(), (unsigned) err);
        क्षिप std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    एतत्->valid_queue = सत्य;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &एतत्->queue_id);
    यदि (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        क्षिप std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    यदि (opt_etm) {
        err = htp_iface_etm(एतत्->handle, 1);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    एतत्->op_queue = नूतन ggml_hexagon_opqueue(एतत्, opt_opbatch, opt_opqueue);

    एतत्->fence_buf = नूतन ggml_hexagon_fence_buffer(एतत्, &dev_ctx->fence_buffer_type, 64 * 1024);
    यदि (एतत्->mdev.count > 1) {
        एतत्->mdev_fence_slot = एतत्->alloc_fence(एतत्->mdev.count);
    }

    यदि (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(एतत्);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", एतत्->c_name(), opt_vmem);
    }
    ध्रुव size_t shm_size = एतत्->op_queue->shm_size();
    एतत्->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    एतत्->op_batch = नूतन ggml_hexagon_opbatch(एतत्, opt_opbatch, एतत्->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(एतत्->handle, एतत्->session_id, एतत्->queue_id, एतत्->n_threads, opt_nhmx, एतत्->max_vmem, (uint32_t) opt_power_mode);
    यदि (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", एतत्->c_name(), (unsigned) err);
        क्षिप std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    एतत्->valid_iface = सत्य;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", एतत्->c_name(), htp_power_mode_to_str(opt_power_mode));

    यदि (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(एतत्->handle, opt_profile, &pmu_conf);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

शून्य ggml_hexagon_session::release() noexcept(सत्य) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", एतत्->name.c_str());

    एतत्->mdev.sessions.clear();

    पूर्णाङ्क err;

    यदि (एतत्->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(एतत्->handle);
        यदि (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete एतत्->op_batch;
    delete एतत्->op_queue;
    प्रति (auto & it : एतत्->cpy_fence_slots) {
        free_fence((शून्य *) it.second, 1);
    }
    एतत्->cpy_fence_slots.clear();

    यदि (एतत्->fence_buf) {
        unclone_buffer(एतत्->fence_buf);
        delete एतत्->fence_buf;
        एतत्->fence_buf = nullptr;
    }
    यावत् (!एतत्->cloned_buffers.empty()) {
        release_buffer(एतत्->cloned_buffers.begin()->second.get());
    }

    यदि (opt_etm) {
        err = htp_iface_etm(एतत्->handle, 0);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    यदि (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(एतत्->handle, 0, &pmu_conf);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    यदि (एतत्->valid_queue) {
        err = dspqueue_close(queue);
        यदि (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    यदि (एतत्->valid_handle) {
        htp_iface_close(एतत्->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ध्रुव ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_मोजणी) noexcept(असत्य) {
    एतत्->dev        = dev;
    एतत्->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    एतत्->mdev.idx   = mdev_idx;
    एतत्->mdev.count = mdev_मोजणी > 0 ? mdev_मोजणी : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)एतत्) & 0xFFFF;

    यत्न {
        allocate(config);
        यदि (एतत्->mdev.idx == 0 && !config.mdev_group.empty()) {
            प्रति (size_t i = 0; i < config.mdev_group.size(); i++) {
                एतत्->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], एतत्->dev, (uint32_t) (i + 1), एतत्->mdev.count));
            }
        }
    } गृहाण (ध्रुव std::exception & exc) {
        release();
        क्षिप;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(सत्य) {
    release();
}

// ** backend interface

स्थिर तार्किक ggml_hexagon_flash_attn_is_hmx_eligible(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * q,
    ध्रुव संरचना ggml_tensor * k,
    ध्रुव संरचना ggml_tensor * v,
    ध्रुव संरचना ggml_tensor * sinks
) {
    यदि (sess->n_hmx == 0) {
        प्रत्यागम असत्य;
    }

    यदि (opt_fa_select < 2) {
        प्रत्यागम असत्य;
    }

    यदि ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t DK = q->ne[0];
    ध्रुव uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    यदि (DK % 8 != 0 || DV % 8 != 0) {
        प्रत्यागम असत्य;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    यदि (opt_fa_select > 2) {
        प्रत्यागम सत्य;
    }

    ध्रुव uint32_t M = q->ne[1];

    // Prefill or batched decode
    यदि (M > 1) {
        प्रत्यागम सत्य;
    }

    ध्रुव uint32_t n_head = q->ne[2];
    ध्रुव uint32_t n_kv_heads = k->ne[2];
    ध्रुव uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    ध्रुव uint32_t S = k->ne[1];
    ध्रुव uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    यदि (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        प्रत्यागम सत्य;
    }

    // Tile alignment for 32-row HMX tiles
    ध्रुव तार्किक is_tile_aligned = (G > 0 && (32 % G == 0));
    यदि (!is_tile_aligned) {
        यदि (sess->n_threads >= 6) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    यदि (DK == 256) {
        s_cross = 4096;
    } अन्यथा यदि (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } अन्यथा {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    प्रत्यागम S >= s_cross;
}

स्थिर तार्किक ggml_hexagon_precompute_flash_attn_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_fa_kernel_params * kparams
) {
    यदि (opt_fa_select < 1) {
        प्रत्यागम असत्य;
    }

    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव संरचना ggml_tensor * q    = op->src[0];
    ध्रुव संरचना ggml_tensor * k    = op->src[1];
    ध्रुव संरचना ggml_tensor * v    = op->src[2];
    ध्रुव संरचना ggml_tensor * mask = op->src[3];

    ध्रुव uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ध्रुव uint32_t neq1 = q->ne[1];  // n_tokens
    ध्रुव uint32_t neq2 = q->ne[2];  // n_heads

    ध्रुव uint32_t nek1 = k->ne[1];  // kv_len

    ध्रुव uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ध्रुव uint32_t DK = neq0;
    ध्रुव uint32_t DV = nev0;

    ध्रुव uint32_t n_kv_heads = k->ne[2];
    ध्रुव uint32_t G          = neq2 / n_kv_heads;

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], परिमाण(भिन्न));
    memcpy(&max_bias,      &op->op_params[1], परिमाण(भिन्न));
    memcpy(&logit_softcap, &op->op_params[2], परिमाण(भिन्न));

    यदि (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ध्रुव uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    ध्रुव संरचना ggml_tensor * sinks = op->src[4];
    यदि (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ध्रुव uint32_t DK_pad = hex_round_up(DK, 64);
        ध्रुव uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        पूर्णाङ्क ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        यदि (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            यदि (logit_softcap == 0.0f) {
                kparams->scale = scale * (भिन्न) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } अन्यथा {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (भिन्न) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            ध्रुव size_t row_vec_bytes = hex_align_up(Bc * परिमाण(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            ध्रुव size_t m_ओळ_bytes = hex_align_up(Bc * परिमाण(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_ओळ_bytes / परिमाण(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_मूल्ये(G);
            यदि (mask) {
                kparams->src3_div2 = init_fastdiv_मूल्ये(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_मूल्ये(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            प्रत्यागम सत्य;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    ध्रुव size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ध्रुव size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    ध्रुव size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_मूल्ये(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_मूल्ये(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_मूल्ये(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_मूल्ये(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_मूल्ये(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_मूल्ये(q->ne[3]/v->ne[3]);
    यदि (mask) {
        kparams->src3_div2 = init_fastdiv_मूल्ये(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_मूल्ये(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_flash_attn_ext(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * src3 = op->src[3];
    ध्रुव संरचना ggml_tensor * src4 = op->src[4];
    ध्रुव संरचना ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    यदि ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        प्रत्यागम असत्य;
    }

    यदि (src3 && src3->type != GGML_TYPE_F16) {  // mask
        प्रत्यागम असत्य;
    }

    यदि (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        प्रत्यागम असत्य;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }

    संरचना htp_fa_kernel_params kparams;
    यदि (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        प्रत्यागम असत्य;
    }

    यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_gated_delta_net(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    यदि (opt_gdn_select < 1) {
        प्रत्यागम असत्य;
    }

    ध्रुव संरचना ggml_tensor * q     = op->src[0];
    ध्रुव संरचना ggml_tensor * k     = op->src[1];
    ध्रुव संरचना ggml_tensor * v     = op->src[2];
    ध्रुव संरचना ggml_tensor * g     = op->src[3];
    ध्रुव संरचना ggml_tensor * beta  = op->src[4];
    ध्रुव संरचना ggml_tensor * state = op->src[5];
    ध्रुव संरचना ggml_tensor * dst   = op;

    यदि (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t S_v      = v->ne[0];
    ध्रुव int64_t H        = v->ne[1];
    ध्रुव int64_t n_tokens = v->ne[2];
    ध्रुव int64_t n_seqs   = v->ne[3];
    ध्रुव int64_t K        = ggml_get_op_params_i32(op, 0);

    यदि (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        प्रत्यागम असत्य;
    }
    यदि (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        प्रत्यागम असत्य;
    }
    यदि ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        प्रत्यागम असत्य;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    यदि (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        प्रत्यागम असत्य;
    }
    यदि (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t total_rows = (uint32_t) (H * n_seqs);
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ध्रुव तार्किक can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    यदि (can_use_hmx) {
        संरचना htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        यदि (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, सत्य, &n_heads_batch)) {
            प्रत्यागम असत्य;
        }
    } अन्यथा {
        संरचना htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        यदि (layout.total_bytes > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_matmul_is_hmx_eligible(
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    तार्किक is_matmul_id,
    तार्किक is_batched
) {
    GGML_UNUSED(dst);

    यदि (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        प्रत्यागम असत्य;
    }

    ध्रुव पूर्णाङ्क ne00  = src0->ne[0];
    ध्रुव पूर्णाङ्क ne01  = src0->ne[1];
    ध्रुव पूर्णाङ्क ne11  = src1->ne[1];
    ध्रुव पूर्णाङ्क ne12  = src1->ne[2];
    ध्रुव पूर्णाङ्क wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    यदि (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        प्रत्यागम असत्य;
    }

    // HMX supports F16, F32, and repack quantized types.
    यदि (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        प्रत्यागम असत्य;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    यदि (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        प्रत्यागम असत्य;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ध्रुव तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    यदि (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        प्रत्यागम असत्य;
    }
    यदि (!is_matmul_id && is_batched && is_tiled) {
        प्रत्यागम असत्य;
    }

    // HMX assumes contiguous row-major layout.
    यदि (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        प्रत्यागम असत्य;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ध्रुव पूर्णाङ्क m = is_matmul_id ? ne12 : ne11;
    यदि (m <= HTP_MM_HMX_MIN_NROWS) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(dst);
}

स्थिर तार्किक ggml_hexagon_precompute_hmx_mm_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * act,
    ध्रुव संरचना ggml_tensor * dst,
    पूर्णाङ्क wtype,
    पूर्णाङ्क ne00_padded,
    पूर्णाङ्क ne01_padded,
    पूर्णाङ्क ne02,
    पूर्णाङ्क ne11,
    पूर्णाङ्क ne12,
    पूर्णाङ्क ne11_padded,
    तार्किक is_matmul_id,
    तार्किक is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    संरचना htp_mm_kernel_params * kparams
) {
    ध्रुव तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ध्रुव पूर्णाङ्क aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    ध्रुव पूर्णाङ्क n_threads = (पूर्णाङ्क)sess->n_threads;
    ध्रुव पूर्णाङ्क ne10 = act->ne[0];

    पूर्णाङ्क m_for_solver = ne11;
    पूर्णाङ्क m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    यदि (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ध्रुव तार्किक dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : सत्य;
        ध्रुव तार्किक act_row_split = ggml_hexagon_can_row_partition(act);
        यदि (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (पूर्णाङ्क) sess->mdev.count - 1) / (पूर्णाङ्क) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    ध्रुव तार्किक pipeline = is_matmul_id ? असत्य : htp_mm_hmx_pipeline(m_for_solver);

    ध्रुव तार्किक is_batched_val = is_matmul_id ? असत्य : is_batched;
    ध्रुव पूर्णाङ्क group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    तार्किक use_grouped = असत्य;
    पूर्णाङ्क act_threads_selected = 0;

    यदि (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        यदि (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = सत्य;
        }
    }

    यदि (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        पूर्णाङ्क m_id_rows = 0;
        यदि (dst && is_matmul_id) {
            ध्रुव पूर्णाङ्क n_experts = ne02 > 0 ? ne02 : 1;
            ध्रुव size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            पूर्णाङ्क m_per_expert = (पूर्णाङ्क) ((total_expert_rows + n_experts - 1) / n_experts);
            यदि (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (पूर्णाङ्क) sess->mdev.count - 1) / (पूर्णाङ्क) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        ध्रुव uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        यदि (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            प्रत्यागम असत्य;
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
    kparams->div_n_act_threads = init_fastdiv_मूल्ये(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_मूल्ये(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    यदि (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    प्रत्यागम सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_hvx_mm_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    पूर्णाङ्क wtype,
    तार्किक is_tiled,
    पूर्णाङ्क ne02,
    पूर्णाङ्क ne03,
    पूर्णाङ्क ne10,
    पूर्णाङ्क ne11,
    पूर्णाङ्क ne12,
    पूर्णाङ्क ne13,
    तार्किक is_matmul_id,
    ध्रुव size_t src2_row_size,
    size_t vtcm_budget,
    संरचना htp_mm_kernel_params * kparams
) {
    यदि (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    ध्रुव पूर्णाङ्क src1_nrows = ne11 * ne12 * ne13;

    यदि (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        ध्रुव तार्किक k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        यदि (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (पूर्णाङ्क) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            संरचना htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            प्रति (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, सत्य, असत्य
                );
                यदि (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    भङ्ग;
                }
            }
            यदि (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                प्रत्यागम;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            प्रत्यागम;
        } अन्यथा {
            तार्किक try_tiled = (k_align && opt_mm_select >= 1);
            यदि (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                यदि (src1_nrows < (पूर्णाङ्क) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } अन्यथा {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                संरचना htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                प्रति (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, असत्य, असत्य
                    );
                    यदि (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        भङ्ग;
                    }
                }

                uint32_t m_chunk = 0;
                यदि (htp_mm_hvx_solve_vtcm_params(
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
                    प्रत्यागम;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            प्रत्यागम;
        }
    } अन्यथा यदि (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    } अन्यथा यदि (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        यदि (htp_mm_hvx_solve_vtcm_params(
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
            प्रत्यागम;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    } अन्यथा यदि (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        यदि (htp_mm_hvx_solve_vtcm_params(
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
            प्रत्यागम;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    } अन्यथा यदि (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        यदि (htp_mm_hvx_solve_vtcm_params(
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
            प्रत्यागम;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    }
}

स्थिर शून्य ggml_hexagon_precompute_matmul_params_impl(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    ध्रुव size_t src2_row_size,
    ध्रुव size_t src2_size,
    संरचना htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव पूर्णाङ्क ne00 = src0->ne[0];
    ध्रुव पूर्णाङ्क ne01 = src0->ne[1];
    ध्रुव पूर्णाङ्क ne02 = src0->ne[2];
    ध्रुव पूर्णाङ्क ne03 = src0->ne[3];

    ध्रुव पूर्णाङ्क ne10 = src1->ne[0];
    ध्रुव पूर्णाङ्क ne11 = src1->ne[1];
    ध्रुव पूर्णाङ्क ne12 = src1->ne[2];
    ध्रुव पूर्णाङ्क ne13 = src1->ne[3];

    ध्रुव तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ध्रुव पूर्णाङ्क wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ध्रुव पूर्णाङ्क ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ध्रुव पूर्णाङ्क ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ध्रुव पूर्णाङ्क ne01_tiled  = hex_round_up(ne01, 32);

    ध्रुव तार्किक is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ध्रुव तार्किक is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ध्रुव size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    तार्किक hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    यदि (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        यदि (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            गच्छ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_मूल्ये(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_मूल्ये(ne11);
    kparams->div_r2       = init_fastdiv_मूल्ये(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_मूल्ये(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_मूल्ये(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
स्थिर तार्किक ggml_hexagon_rows_stride(ध्रुव int64_t * ne, ध्रुव size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        यदि (ne[i] == 1) अनुवर्त;
        यदि (s == 0) { s = nb[i]; next = s * ne[i]; अनुवर्त; }
        यदि (nb[i] != next) प्रत्यागम असत्य;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    प्रत्यागम सत्य;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
स्थिर तार्किक ggml_hexagon_matmul_can_collapse(ध्रुव संरचना ggml_tensor * src0, ध्रुव संरचना ggml_tensor * src1, ध्रुव संरचना ggml_tensor * dst) {
    size_t s1, sd;
    प्रत्यागम (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

स्थिर तार्किक ggml_hexagon_matmul_add_can_collapse(
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * src2,
    ध्रुव संरचना ggml_tensor * dst
) {
    यदि (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        प्रत्यागम असत्य;
    }
    यदि (!src2) {
        प्रत्यागम सत्य;
    }
    ध्रुव int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    यदि (src2_nrows == 1) {
        प्रत्यागम सत्य;
    }
    size_t s2;
    प्रत्यागम src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

स्थिर ggml_tensor ggml_hexagon_tensor_collapse_rows(ध्रुव संरचना ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    प्रत्यागम c;
}

स्थिर शून्य ggml_hexagon_precompute_matmul_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
) {
    यदि (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ध्रुव ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ध्रुव ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        प्रत्यागम;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

स्थिर शून्य ggml_hexagon_precompute_fused_matmul_add_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * src2,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
) {
    यदि (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        ध्रुव ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ध्रुव ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ध्रुव ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ध्रुव संरचना ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        ध्रुव size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        प्रत्यागम;
    }
    ध्रुव size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

स्थिर तार्किक ggml_hexagon_precompute_द्विमान_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    uint32_t op,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_द्विमान_kernel_params * kparams,
    ध्रुव संरचना ggml_tensor * src2
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव size_t elem_size = ggml_type_size(src0->type);
    ध्रुव size_t src0_row_size = src0->ne[0] * elem_size;
    ध्रुव size_t src1_row_size = src1->ne[0] * elem_size;
    ध्रुव size_t dst_row_size  = dst->ne[0]  * elem_size;

    ध्रुव size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    ध्रुव size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    ध्रुव size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    यदि (!htp_द्विमान_strides_safe(src0, src1, dst, elem_size)) {
        प्रत्यागम असत्य;
    }

    ध्रुव तार्किक is_add_id = op == HTP_OP_ADD_ID;
    ध्रुव तार्किक is_scalar = !is_add_id && src1->ne[0] == 1;
    ध्रुव तार्किक is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ध्रुव तार्किक is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ध्रुव तार्किक is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    ध्रुव तार्किक is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ध्रुव तार्किक is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    यदि (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        ध्रुव uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        ध्रुव uint32_t n_threads = sess->n_threads;
        ध्रुव uint32_t max_chunk_elems = 32768 / elem_size;
        ध्रुव uint32_t min_chunk_elems = 256;
        ध्रुव uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        ध्रुव uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        ध्रुव uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_द्विमान_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        संरचना htp_द्विमान_vtcm_layout L;
        htp_द्विमान_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        यदि (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }

        kparams->vtcm_size = L.total_bytes;
        प्रत्यागम सत्य;
    }

    गणना htp_द्विमान_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    यदि (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_द्विमान_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * परिमाण(int32_t), 128);
        यदि (src1_size + src2_size > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }
    } अन्यथा यदि (is_row_bcast) {
        kernel_type = HTP_द्विमान_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } अन्यथा यदि (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        यदि (src1_size > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }
        kernel_type = HTP_द्विमान_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा यदि (is_same_shape) {
        kernel_type = HTP_द्विमान_KERNEL_SAME_SHAPE;
    } अन्यथा यदि (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        यदि (src1_size > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }
        kernel_type = HTP_द्विमान_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        यदि (src1_size > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }
        kernel_type = HTP_द्विमान_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    ध्रुव तार्किक src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ध्रुव तार्किक src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ध्रुव तार्किक dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ध्रुव तार्किक src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ध्रुव तार्किक src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ध्रुव तार्किक dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    तार्किक split_at_ne01 = असत्य;
    यदि (src0->ne[2] > 1) {
        यदि (kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } अन्यथा यदि (kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } अन्यथा {
            split_at_ne01 = सत्य;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    तार्किक split_at_ne02 = असत्य;
    यदि (src0->ne[3] > 1) {
        यदि (kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } अन्यथा यदि (kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } अन्यथा {
            split_at_ne02 = सत्य;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    संरचना htp_द्विमान_vtcm_layout L;
    htp_द्विमान_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    यदि (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        यदि (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            ध्रुव uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            ध्रुव uint32_t n_threads = sess->n_threads;
            ध्रुव uint32_t max_chunk_elems = 32768 / elem_size;
            ध्रुव uint32_t min_chunk_elems = 256;
            ध्रुव uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            ध्रुव uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            ध्रुव uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_द्विमान_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_द्विमान_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            यदि (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                प्रत्यागम असत्य;
            }

            kparams->vtcm_size = L.total_bytes;
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    प्रत्यागम सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_unary_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    uint32_t op,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    ध्रुव size_t elem_size = ggml_type_size(src0->type);

    ध्रुव size_t src0_data_row_size = src0->ne[0] * elem_size;
    ध्रुव size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    ध्रुव size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ध्रुव size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    तार्किक broadcast_weight = असत्य;

    यदि (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    संरचना htp_unary_vtcm_layout L;
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

    ध्रुव uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_मूल्ये(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_मूल्ये(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_मूल्ये(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_मूल्ये(tiles_per_row);
}

स्थिर शून्य ggml_hexagon_precompute_get_rows_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t ne02 = src0->ne[2];
    ध्रुव uint32_t ne03 = src0->ne[3];

    ध्रुव uint32_t ne10 = src1->ne[0];
    ध्रुव uint32_t ne11 = src1->ne[1];
    ध्रुव uint32_t ne12 = src1->ne[2];
    ध्रुव uint32_t nr = ne10 * ne11 * ne12;

    ध्रुव तार्किक tiled = ggml_hexagon_tensor_is_tiled(src0);
    ध्रुव पूर्णाङ्क wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    यदि (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } अन्यथा यदि (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } अन्यथा {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    ध्रुव uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    संरचना htp_get_rows_vtcm_layout vtcm_layout = {};
    यावत् (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        यदि (vtcm_layout.total_bytes <= sess->vtcm_size) {
            भङ्ग;
        }
        --kparams->n_threads;
    }

    यदि (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_मूल्ये(ne10) : fastdiv_मूल्ये{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_मूल्ये(ne10 * ne11) : fastdiv_मूल्ये{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_मूल्ये(ne02) : fastdiv_मूल्ये{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_मूल्ये(ne03) : fastdiv_मूल्ये{0, 0};
}

स्थिर शून्य ggml_hexagon_precompute_set_rows_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0, //मूल्ये
    ध्रुव संरचना ggml_tensor * src1, // indices
    ध्रुव संरचना ggml_tensor * dst,  // destination
    संरचना htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_मूल्ये(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_मूल्ये(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_मूल्ये(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_मूल्ये(src0->ne[2]);

    संरचना htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

स्थिर शून्य ggml_hexagon_precompute_softmax_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    भिन्न scale = 1.0f;
    भिन्न max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], परिमाण(भिन्न));
    memcpy(&max_bias, &op->op_params[1], परिमाण(भिन्न));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ध्रुव uint32_t n_head = src0->ne[2];
    ध्रुव uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    यदि (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } अन्यथा {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t ne10 = src1 ? src1->ne[0] : 1;

    संरचना htp_softmax_vtcm_layout layout;
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
    यदि (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } अन्यथा यदि (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } अन्यथा {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    यदि (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_मूल्ये(src0->ne[1]);
    यदि (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_मूल्ये(src0->ne[2]);
    ध्रुव uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ध्रुव uint32_t ne13 = src1 ? src1->ne[3] : 1;
    यदि (ne12 > 0) kparams->div_ne12 = init_fastdiv_मूल्ये(ne12);
    यदि (ne13 > 0) kparams->div_ne13 = init_fastdiv_मूल्ये(ne13);
}

स्थिर शून्य ggml_hexagon_precompute_rope_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * dst  = op;

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ध्रुव uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    संरचना htp_rope_vtcm_layout layout;
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

    यदि (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_मूल्ये(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_मूल्ये(dst->ne[1]);
    }
}

स्थिर शून्य ggml_hexagon_precompute_ssm_conv_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव uint32_t d_conv  = (uint32_t) src1->ne[0];
    ध्रुव uint32_t d_inner = (uint32_t) src0->ne[1];
    ध्रुव uint32_t n_t     = (uint32_t) dst->ne[1];
    ध्रुव uint32_t n_s     = (uint32_t) dst->ne[2];
    ध्रुव uint32_t ncs     = (uint32_t) src0->ne[0];

    ध्रुव uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ध्रुव uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    ध्रुव uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    ध्रुव uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * परिमाण(भिन्न), 128) + 128;
    ध्रुव uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * परिमाण(भिन्न), 128);
    ध्रुव uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    यदि (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        ध्रुव uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * परिमाण(भिन्न), 128);
        ध्रुव uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * परिमाण(भिन्न), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        ध्रुव uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * परिमाण(भिन्न), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } अन्यथा {
        ध्रुव size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ध्रुव uint32_t src0_block_T    = hex_round_up(ncs * 32 * परिमाण(भिन्न), 128);
        ध्रुव size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        ध्रुव size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * परिमाण(भिन्न)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        यदि (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        यदि (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        यदि (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ध्रुव uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * परिमाण(भिन्न), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * परिमाण(भिन्न), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

स्थिर शून्य ggml_hexagon_precompute_gated_delta_net_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव संरचना ggml_tensor * q     = op->src[0];
    ध्रुव संरचना ggml_tensor * k     = op->src[1];
    ध्रुव संरचना ggml_tensor * v     = op->src[2];
    ध्रुव संरचना ggml_tensor * g     = op->src[3];
    ध्रुव संरचना ggml_tensor * state = op->src[5];

    ध्रुव uint32_t S_v      = (uint32_t) v->ne[0];
    ध्रुव uint32_t H        = (uint32_t) v->ne[1];
    ध्रुव uint32_t n_tokens = (uint32_t) v->ne[2];
    ध्रुव uint32_t n_seqs   = (uint32_t) v->ne[3];
    ध्रुव uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ध्रुव uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ध्रुव uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ध्रुव uint32_t total_rows = H * n_seqs;
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ध्रुव तार्किक can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    संरचना htp_gdn_hmx_vtcm_layout hmx_layout;
    संरचना htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    यदि (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, सत्य, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } अन्यथा {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / परिमाण(भिन्न));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((भिन्न) S_v);

    यदि (H > 0)                  kparams->div_H         = init_fastdiv_मूल्ये(H);
    यदि (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_मूल्ये((uint32_t) q->ne[1]);
    यदि (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_मूल्ये((uint32_t) k->ne[1]);
    यदि (rq3 > 0)                kparams->div_rq3       = init_fastdiv_मूल्ये(rq3);
    यदि (rk3 > 0)                kparams->div_rk3       = init_fastdiv_मूल्ये(rk3);
    यदि (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_मूल्ये(kparams->n_threads);
}

स्थिर शून्य ggml_hexagon_precompute_sort_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    तार्किक is_शीर्ष_k,
    संरचना htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    ध्रुव uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t ne00       = src0->ne[0];
    ध्रुव uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    यदि (!is_शीर्ष_k) {
        order = ((ध्रुव int32_t *) op->op_params)[0];
    }

    ध्रुव uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ध्रुव size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    संरचना htp_sort_vtcm_layout layout;
    तार्किक ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_शीर्ष_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_शीर्ष_k          = is_शीर्ष_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_मूल्ये_off  = (int32_t) layout.merge_मूल्ये_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

स्थिर शून्य ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ध्रुव uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ध्रुव int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    यदि (numer_hi < 0) {
        *ox_hi = 0;
    } अन्यथा {
        ध्रुव uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    यदि (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

स्थिर तार्किक ggml_hexagon_pool_shape_fits(
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * dst
) {
    प्रत्यागम ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

स्थिर शून्य ggml_hexagon_precompute_pool_2d_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * dst,
    संरचना htp_pool_2d_kernel_params * kparams,
    तार्किक is_pool_1d
) {
    memset(kparams, 0, परिमाण(*kparams));

    ध्रुव uint32_t src_x = (uint32_t) src0->ne[0];
    ध्रुव uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ध्रुव uint32_t dst_x = (uint32_t) dst->ne[0];
    ध्रुव uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ध्रुव uint32_t src_plane_bytes = src_x * src_y * परिमाण(भिन्न);
    ध्रुव uint32_t dst_plane_bytes = dst_x * dst_y * परिमाण(भिन्न);
    ध्रुव uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ध्रुव uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    ध्रुव तार्किक ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_मोजणी = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ध्रुव तार्किक narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (भिन्न) (kparams->kernel_x * kparams->kernel_y);
}

स्थिर तार्किक ggml_hexagon_precompute_concat_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!src0 || !src1 || !dst) {
        प्रत्यागम असत्य;
    }

    पूर्णाङ्क dim = ((ध्रुव int32_t *) op->op_params)[0];
    यदि (dim < 0 || dim >= GGML_MAX_DIMS) {
        प्रत्यागम असत्य;
    }
    kparams->dim = dim;

    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }
    यदि (src0->type != dst->type || src1->type != dst->type) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t type_size = ggml_type_size(dst->type);

    प्रति (पूर्णाङ्क d = 0; d < GGML_MAX_DIMS; d++) {
        ध्रुव int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        यदि (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव uint32_t max_gather_ne0 = 128 / type_size;
    ध्रुव uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ध्रुव तार्किक b_dense = (src1->nb[0] == type_size);
    ध्रुव तार्किक gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    यदि (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        यदि (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            प्रत्यागम सत्य;
        }
    }

    ध्रुव तार्किक dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    यदि (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        प्रत्यागम सत्य;
    }

    ध्रुव तार्किक is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ध्रुव तार्किक is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ध्रुव तार्किक transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    यदि (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ध्रुव uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        संरचना htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        यदि (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            प्रत्यागम असत्य;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        प्रत्यागम सत्य;
    }

    प्रत्यागम असत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_cpy_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!src0 || !dst) {
        प्रत्यागम असत्य;
    }

    यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t nelem_src = ggml_nelements(src0);
    ध्रुव int64_t nelem_dst = ggml_nelements(dst);
    यदि (nelem_src != nelem_dst || nelem_src < 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t src_type_size = ggml_type_size(src0->type);
    ध्रुव uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    यदि (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        प्रत्यागम सत्य;
    }

    यदि (nelem_src == 1) {
        यदि (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            प्रत्यागम सत्य;
        }
        यदि ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    }

    ध्रुव तार्किक sametype = (src0->type == dst->type);

    तार्किक same_extents = सत्य;
    प्रति (पूर्णाङ्क d = 0; d < GGML_MAX_DIMS; d++) {
        यदि (src0->ne[d] != dst->ne[d]) {
            same_extents = असत्य;
            भङ्ग;
        }
    }

    ध्रुव तार्किक transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    ध्रुव तार्किक sameshape  = same_extents && !transposed;

    ध्रुव तार्किक src_is_contiguous = ggml_is_contiguous(src0);
    ध्रुव तार्किक dst_is_contiguous = ggml_is_contiguous(dst);

    यदि (sametype) {
        यदि (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            प्रत्यागम सत्य;
        }

        ध्रुव uint32_t max_gather_ne0 = 128 / src_type_size;
        ध्रुव uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        यदि ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            यदि (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, असत्य,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                प्रत्यागम सत्य;
            }
        }

        यदि (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            प्रत्यागम सत्य;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_मूल्ये((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_मूल्ये((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_मूल्ये((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_मूल्ये((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_मूल्ये((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_मूल्ये((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        प्रत्यागम सत्य;
    }

    यदि (!sameshape) {
        प्रत्यागम असत्य;
    }

    ध्रुव तार्किक valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    यदि (!valid_conversion) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    संरचना htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    यदि (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        प्रत्यागम असत्य;
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
    kparams->u.convert.div_ne01      = init_fastdiv_मूल्ये((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_मूल्ये((uint32_t) (src0->ne[2] * src0->ne[1]));

    प्रत्यागम सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_fused_mmnx_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0, // W0
    ध्रुव संरचना ggml_tensor * act,  // x
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->n_threads = sess->n_threads;

    ध्रुव पूर्णाङ्क ne00 = src0->ne[0];
    ध्रुव पूर्णाङ्क ne01 = src0->ne[1];
    ध्रुव पूर्णाङ्क ne02 = src0->ne[2];
    ध्रुव पूर्णाङ्क ne03 = src0->ne[3];

    ध्रुव पूर्णाङ्क ne10 = act->ne[0];
    ध्रुव पूर्णाङ्क ne11 = act->ne[1];
    ध्रुव पूर्णाङ्क ne12 = act->ne[2];
    ध्रुव पूर्णाङ्क ne13 = act->ne[3];

    ध्रुव तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ध्रुव पूर्णाङ्क wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ध्रुव पूर्णाङ्क ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    ध्रुव पूर्णाङ्क ne11_padded = hex_round_up(ne11, 32);
    ध्रुव पूर्णाङ्क ne01_tiled  = hex_round_up(ne01, 32);

    ध्रुव size_t vtcm_budget = sess->vtcm_size;
    ध्रुव तार्किक is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    तार्किक hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    यदि (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, असत्य, is_batched)) {
        यदि (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, असत्य, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            गच्छ finalize;
        }
    }

    यदि (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        प्रत्यागम;
    }

    {
        ध्रुव पूर्णाङ्क act_nrows = ne11 * ne12 * ne13;
        ध्रुव size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        ध्रुव size_t src0_row_size = src0->nb[1];

        ध्रुव uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        प्रति (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            संरचना htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, असत्य, सत्य
            );
            यदि (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                भङ्ग;
            }
        }

        संरचना htp_mm_hvx_vtcm_layout L;
        तार्किक try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, असत्य, सत्य
        );

        यदि (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } अन्यथा {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            प्रत्यागम;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_मूल्ये(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_मूल्ये(ne11);
    kparams->div_r2       = init_fastdiv_मूल्ये(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_मूल्ये(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_मूल्ये(ne12);
}

स्थिर शून्य ggml_hexagon_precompute_fused_mmidnx_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * src0, // W0
    ध्रुव संरचना ggml_tensor * act,  // x
    ध्रुव संरचना ggml_tensor * dst,  // dst0
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

स्थिर तार्किक ggml_hexagon_op_mul_mat_use_fwht(ध्रुव संरचना ggml_tensor * dst) {
    यदि (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        प्रत्यागम असत्य;
    }

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    ध्रुव int64_t n = src1->ne[0];
    यदि (n < 32 || (n & (n - 1)) != 0) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

स्थिर तार्किक ggml_hexagon_supported_mul_mat(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * dst) {
    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    यदि (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        यदि (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            प्रत्यागम असत्य;
        }
        संरचना htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        प्रत्यागम ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q2_K: {
            यदि (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                प्रत्यागम असत्य;
            }

            ध्रुव int64_t qk = ggml_blck_size(src0->type);
            यदि (src0->ne[0] % qk) {
                प्रत्यागम असत्य;
            }

            यदि (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;
        }

        पक्ष GGML_TYPE_F16:
            यदि (src0->nb[1] < src0->nb[0]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;

        पक्ष GGML_TYPE_F32:
            यदि (src1->type != GGML_TYPE_F32) {
                प्रत्यागम असत्य;
            }
            यदि (src0->nb[1] < src0->nb[0]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;

        पक्ष GGML_TYPE_BF16:
            यदि (src0->nb[1] < src0->nb[0]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                प्रत्यागम असत्य;
            }
            यदि (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;

        स्वभाव:
            प्रत्यागम असत्य;
    }

    संरचना htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    यदि (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_mul_mat_id(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    यदि (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        प्रत्यागम असत्य;
    }

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q2_K:
            यदि (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                प्रत्यागम असत्य;
            }

            यदि (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;

        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_BF16:
            यदि (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                प्रत्यागम असत्य;
            }
            यदि (src0->ne[0] % 32 != 0) {
                प्रत्यागम असत्य;
            }
            भङ्ग;

        स्वभाव:
            प्रत्यागम असत्य;
    }

    संरचना htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    यदि (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_द्विमान(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type == GGML_TYPE_F32) {
        यदि (src1->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
        यदि (dst->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
    }
    अन्यथा यदि (src0->type == GGML_TYPE_F16) {
        यदि (src1->type != GGML_TYPE_F16) {
            प्रत्यागम असत्य;
        }
        यदि (dst->type != GGML_TYPE_F16) {
            प्रत्यागम असत्य;
        }
    }
    अन्यथा {
        प्रत्यागम असत्य;
    }

    यदि (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_are_same_shape(src0, dst)) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        प्रत्यागम असत्य;
    }

    संरचना htp_द्विमान_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_द्विमान_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_add_id(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!src2) {
        प्रत्यागम असत्य;
    }
    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_are_same_shape(src0, dst)) {
        प्रत्यागम असत्य;
    }
    यदि (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        प्रत्यागम असत्य;
    }
    यदि (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        प्रत्यागम असत्य;
    }
    यदि (src0->nb[0] != परिमाण(भिन्न) || src1->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(int32_t)) {
        प्रत्यागम असत्य;
    }

    // REVISIT: add support for non-contiguous tensors
    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }
    यदि (src2->nb[1] < src2->ne[0] * परिमाण(int32_t)) {
        प्रत्यागम असत्य;
    }

    संरचना htp_द्विमान_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_द्विमान_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

स्थिर तार्किक ggml_hexagon_supported_unary(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != src0->type) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous_rows(src0)) {
        प्रत्यागम असत्य;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    यदि (src0->type == GGML_TYPE_F16) {
        विकल्प (op->op) {
            पक्ष GGML_OP_NORM:
            पक्ष GGML_OP_RMS_NORM:
            पक्ष GGML_OP_L2_NORM:
            पक्ष GGML_OP_SCALE:
            पक्ष GGML_OP_CLAMP:
            पक्ष GGML_OP_SQR:
            पक्ष GGML_OP_SQRT:
            पक्ष GGML_OP_LOG:
                भङ्ग;
            पक्ष GGML_OP_UNARY:
                यदि (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    प्रत्यागम असत्य;
                }
                भङ्ग;
            स्वभाव:
                प्रत्यागम असत्य;
        }
    }

    यदि (!ggml_are_same_shape(src0, dst)) {
        प्रत्यागम असत्य;
    }

    // dst must be contiguous; src0 may be non-contiguous
    यदि (!ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_sum(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_sum_rows(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    // TODO: add support for non-contigiuos tensors
    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_argmax(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_pool_2d(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव int32_t * params = op->op_params;

    यदि (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        प्रत्यागम असत्य;
    }
    यदि (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_hexagon_pool_shape_fits(src0, op)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t kernel_x = params[1];
    ध्रुव int32_t kernel_y = params[2];
    ध्रुव int32_t stride_x = params[3];
    ध्रुव int32_t stride_y = params[4];
    ध्रुव int32_t pad_x    = params[5];
    ध्रुव int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    यदि (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ध्रुव uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    प्रत्यागम htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

स्थिर तार्किक ggml_hexagon_supported_pool_1d(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव int32_t * params = op->op_params;

    यदि (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        प्रत्यागम असत्य;
    }
    यदि (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_hexagon_pool_shape_fits(src0, op)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t kernel = params[1];
    ध्रुव int32_t stride = params[2];
    ध्रुव int32_t pad    = params[3];
    यदि (kernel <= 0 || stride <= 0 || pad < 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ध्रुव uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    प्रत्यागम htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

स्थिर तार्किक ggml_hexagon_supported_activations(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != src0->type) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous_1(src0)) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    यदि (src1) {
        यदि (src1->type != src0->type) {
            प्रत्यागम असत्य;
        }
        यदि (!ggml_are_same_shape(src0, src1)) {
            प्रत्यागम असत्य;
        }
        यदि (!ggml_is_contiguous_1(src1)) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_softmax(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src2) {
        प्रत्यागम असत्य;  // FIXME: add support for sinks
    }

    यदि (src0->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (src0->ne[2] > 512) {
        प्रत्यागम असत्य;
    }

    यदि (src1) {
        यदि (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            प्रत्यागम असत्य;
        }
        यदि (src0->ne[0] != src1->ne[0]) {
            प्रत्यागम असत्य;
        }
        यदि (src1->ne[1] < src0->ne[1]) {
            प्रत्यागम असत्य;
        }
        यदि (src0->ne[2] % src1->ne[2] != 0) {
            प्रत्यागम असत्य;
        }
        यदि (src0->ne[3] % src1->ne[3] != 0) {
            प्रत्यागम असत्य;
        }
    }

    यदि (src1) {
        यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            प्रत्यागम असत्य;
        }
    } अन्यथा {
        यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            प्रत्यागम असत्य;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ध्रुव int64_t ne0 = src0->ne[0];
    यदि (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        प्रत्यागम असत्य;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    यदि (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    संरचना htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    यदि (layout.total_bytes > sess->vtcm_size) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_set_rows(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    ध्रुव संरचना ggml_tensor * src1 = op->src[1]; // indices
    ध्रुव संरचना ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    यदि (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        प्रत्यागम असत्य;
    }

    यदि (src0->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        प्रत्यागम असत्य;
    }

    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_get_rows(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    ध्रुव संरचना ggml_tensor * src1 = op->src[1]; // indices
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        प्रत्यागम असत्य;
    }

    यदि (src0->type == dst->type) {
        यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            प्रत्यागम असत्य;
        }
    } अन्यथा यदि (src0->type == GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    } अन्यथा यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    यदि (ggml_is_empty(op)) {
        प्रत्यागम सत्य;
    }

    संरचना htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    यदि (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_argsort(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    ध्रुव संरचना ggml_tensor * dst  = op;         // indices

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ध्रुव size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    संरचना htp_sort_vtcm_layout layout;
    यदि (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, असत्य)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_शीर्ष_k(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    ध्रुव संरचना ggml_tensor * dst  = op;         // indices

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ध्रुव size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    संरचना htp_sort_vtcm_layout layout;
    यदि (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, सत्य)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_rope(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * src2 = op->src[2];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!ggml_are_same_shape(src0, dst)) {
        प्रत्यागम असत्य;
    }

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    यदि (src0->ne[0] <= 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    यदि (src0_nrows == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t * op_params = &op->op_params[0];
    ध्रुव पूर्णाङ्क n_dims = op_params[1];
    ध्रुव पूर्णाङ्क mode   = op_params[2];
    ध्रुव पूर्णाङ्क n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    यदि (n_dims < 0 || n_dims % 2 != 0) {
        प्रत्यागम असत्य;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    यदि (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        प्रत्यागम असत्य;
    }

    भिन्न freq_base;
    memcpy(&freq_base, op_params + 5, परिमाण(भिन्न));
    यदि (freq_base < 0.0f) {
        प्रत्यागम असत्य;
    }

    यदि (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        प्रत्यागम असत्य;
    }

    ध्रुव तार्किक is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    यदि (mode == GGML_ROPE_TYPE_VISION) {
        यदि (n_dims != (पूर्णाङ्क) (src0->ne[0] / 2) || n_offs != 0) {
            प्रत्यागम असत्य;
        }
    }

    यदि (is_mrope) {
        ध्रुव int32_t * sections = op_params + 11;
        यदि (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    यदि (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        प्रत्यागम असत्य;
    }

    यदि (src2) {
        यदि (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            प्रत्यागम असत्य;
        }
        यदि (src2->ne[0] < (n_dims / 2)) {
            प्रत्यागम असत्य;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    यदि (src0->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न)) {
        प्रत्यागम असत्य;
    }
    यदि (src0->nb[1] < src0->ne[0] * परिमाण(भिन्न) || dst->nb[1] < dst->ne[0] * परिमाण(भिन्न)) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ध्रुव uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    संरचना htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    यदि (layout.total_bytes > sess->vtcm_size) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_ssm_conv(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    // Only support FP32 for now
    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    // Check IO tensor shapes and dims
    यदि (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        प्रत्यागम असत्य; // src0 should be effectively 3D
    }

    ध्रुव पूर्णाङ्क d_conv  = src1->ne[0];
    ध्रुव पूर्णाङ्क d_inner = src0->ne[1];
    ध्रुव पूर्णाङ्क n_t = dst->ne[1];
    ध्रुव पूर्णाङ्क n_s = dst->ne[2];

    यदि (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        प्रत्यागम असत्य;
    }
    यदि (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        प्रत्यागम असत्य;
    }
    यदि (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        प्रत्यागम असत्य;
    }
    यदि (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        प्रत्यागम असत्य;
    }
    यदि (src0->nb[0] != परिमाण(भिन्न) || src1->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न)) {
        प्रत्यागम असत्य;
    }
    यदि (src0->nb[1] != src0->ne[0] * परिमाण(भिन्न) || src1->nb[1] != src1->ne[0] * परिमाण(भिन्न)) {
        प्रत्यागम असत्य;
    }

    संरचना htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    यदि ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक im2col_3d_calc_output_size(uint32_t आदान, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    यदि (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t numerator = (int64_t) आदान + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    यदि (numerator < 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint64_t size = (uint64_t) (numerator / stride) + 1;
    यदि (size == 0 || size > UINT32_MAX) {
        प्रत्यागम असत्य;
    }

    *output = (uint32_t) size;
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_im2col_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!src0 || !src1 || !dst) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        प्रत्यागम असत्य;
    }

    यदि (src1->nb[0] != परिमाण(भिन्न) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t * op_params = (ध्रुव int32_t *) op->op_params;
    ध्रुव int32_t s0 = op_params[0];
    ध्रुव int32_t s1 = op_params[1];
    ध्रुव int32_t p0 = op_params[2];
    ध्रुव int32_t p1 = op_params[3];
    ध्रुव int32_t d0 = op_params[4];
    ध्रुव int32_t d1 = op_params[5];
    ध्रुव तार्किक is_2D = (op_params[6] == 1);

    यदि (s0 <= 0 || d0 <= 0 || p0 < 0) {
        प्रत्यागम असत्य;
    }
    यदि (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ध्रुव uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ध्रुव uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ध्रुव uint32_t IW = (uint32_t) src1->ne[0];
    ध्रुव uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ध्रुव uint32_t KW = (uint32_t) src0->ne[0];
    ध्रुव uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ध्रुव uint32_t OW = (uint32_t) dst->ne[1];

    यदि (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ध्रुव uint64_t total_rows_64   = (uint64_t) N * OH;
    यदि (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        प्रत्यागम असत्य;
    }
    ध्रुव uint32_t patch_stride = (uint32_t) patch_stride_64;
    ध्रुव uint32_t total_rows   = (uint32_t) total_rows_64;

    यदि (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        प्रत्यागम असत्य;
    }
    यदि (is_2D && dst->ne[2] != OH) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? परिमाण(ggml_fp16_t) : परिमाण(भिन्न);

    तार्किक is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    यदि (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ध्रुव uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ध्रुव uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ध्रुव uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    यदि (is_patchembed) {
        ध्रुव uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * परिमाण(भिन्न);
        ध्रुव uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        यदि (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ध्रुव uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ध्रुव uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            संरचना htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            यदि (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                प्रत्यागम सत्य;
            }
        }
    }

    प्रति (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ध्रुव uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        यदि (Wb > UINT32_MAX) {
            यदि (owb == 1) भङ्ग;
            अनुवर्त;
        }

        ध्रुव uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * परिमाण(भिन्न);
        ध्रुव uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        यदि (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            यदि (owb == 1) भङ्ग;
            अनुवर्त;
        }

        ध्रुव uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ध्रुव uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        संरचना htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        यदि (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            प्रत्यागम सत्य;
        }
        यदि (owb == 1) भङ्ग;
    }

    प्रत्यागम असत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_im2col_3d_params(
    ध्रुव संरचना ggml_hexagon_session * sess,
    ध्रुव संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, परिमाण(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (!src0 || !src1 || !dst) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        प्रत्यागम असत्य;
    }

    यदि (src1->nb[0] != परिमाण(भिन्न) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t * params = (ध्रुव int32_t *) op->op_params;
    ध्रुव int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ध्रुव int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ध्रुव int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ध्रुव int32_t ic = params[9];

    यदि (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        प्रत्यागम असत्य;
    }

    यदि (src1->ne[3] % ic != 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t IC = (uint32_t) ic;
    ध्रुव uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    यदि (N == 0) {
        प्रत्यागम असत्य;
    }

    uint32_t OW, OH, OD;
    यदि (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t KD = (uint32_t) src0->ne[2];
    ध्रुव uint32_t KH = (uint32_t) src0->ne[1];
    ध्रुव uint32_t KW = (uint32_t) src0->ne[0];
    ध्रुव uint32_t ID = (uint32_t) src1->ne[2];
    ध्रुव uint32_t IH = (uint32_t) src1->ne[1];
    ध्रुव uint32_t IW = (uint32_t) src1->ne[0];

    ध्रुव uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ध्रुव uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    ध्रुव uint64_t dst_ne3_64      = (uint64_t) OD * N;

    यदि (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        प्रत्यागम असत्य;
    }
    ध्रुव uint32_t patch_stride = (uint32_t) patch_stride_64;
    ध्रुव uint32_t total_rows   = (uint32_t) total_rows_64;
    ध्रुव uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    यदि (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? परिमाण(ggml_fp16_t) : परिमाण(भिन्न);

    ध्रुव तार्किक is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ध्रुव uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ध्रुव uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ध्रुव uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    यदि (is_patchembed) {
        ध्रुव uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * परिमाण(भिन्न);
        ध्रुव uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        यदि (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ध्रुव uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ध्रुव uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            संरचना htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            यदि (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                प्रत्यागम सत्य;
            }
        }
    }

    प्रति (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ध्रुव uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        यदि (Wb > UINT32_MAX) {
            यदि (owb == 1) भङ्ग;
            अनुवर्त;
        }

        ध्रुव uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * परिमाण(भिन्न);
        ध्रुव uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        यदि (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            यदि (owb == 1) भङ्ग;
            अनुवर्त;
        }

        ध्रुव uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ध्रुव uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        संरचना htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        यदि (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            प्रत्यागम सत्य;
        }
        यदि (owb == 1) भङ्ग;
    }

    प्रत्यागम असत्य;
}

स्थिर तार्किक ggml_hexagon_supported_im2col(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    संरचना htp_im2col_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_im2col_3d(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    संरचना htp_im2col_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_pad(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t lp0 = ((ध्रुव int32_t *) op->op_params)[0];
    ध्रुव int32_t rp0 = ((ध्रुव int32_t *) op->op_params)[1];
    ध्रुव int32_t circular = ((ध्रुव int32_t *) op->op_params)[8];

    यदि (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_cumsum(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_diag(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    // diag only supports F32 currently
    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    // Input must have ne[1] == 1 (vector input)
    यदि (src0->ne[1] != 1) {
        प्रत्यागम असत्य;
    }

    // Output must be square in first two dimensions
    यदि (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_solve_tri(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0]; // A
    ध्रुव संरचना ggml_tensor * src1 = op->src[1]; // B
    ध्रुव संरचना ggml_tensor * dst  = op;         // X

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (src0->ne[0] != src0->ne[1]) {
        प्रत्यागम असत्य;
    }

    यदि (src0->ne[1] != src1->ne[1]) {
        प्रत्यागम असत्य;
    }

    यदि (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        प्रत्यागम असत्य;
    }

    यदि (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_tri(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32) { प्रत्यागम असत्य; }
    यदि (dst->type  != GGML_TYPE_F32) { प्रत्यागम असत्य; }
    यदि (!ggml_are_same_shape(src0, dst)) { प्रत्यागम असत्य; }
    यदि (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { प्रत्यागम असत्य; }

    प्रत्यागम सत्य;

    GGML_UNUSED(sess);
}

स्थिर ध्रुव वर्ण * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    प्रत्यागम sess->c_name();
}

स्थिर शून्य ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

स्थिर htp_op_code op_remap_to_htp(ध्रुव ggml_tensor * t) {
    विकल्प (t->op) {
        पक्ष GGML_OP_FLASH_ATTN_EXT:  प्रत्यागम HTP_OP_FLASH_ATTN_EXT;
        पक्ष GGML_OP_MUL_MAT:
            यदि (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                प्रत्यागम HTP_OP_HADAMARD;
            }
            प्रत्यागम HTP_OP_MUL_MAT;
        पक्ष GGML_OP_MUL_MAT_ID:      प्रत्यागम HTP_OP_MUL_MAT_ID;
        पक्ष GGML_OP_MUL:             प्रत्यागम HTP_OP_MUL;
        पक्ष GGML_OP_ADD:             प्रत्यागम HTP_OP_ADD;
        पक्ष GGML_OP_ADD_ID:          प्रत्यागम HTP_OP_ADD_ID;
        पक्ष GGML_OP_SUB:             प्रत्यागम HTP_OP_SUB;
        पक्ष GGML_OP_DIV:             प्रत्यागम HTP_OP_DIV;
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CPY:             प्रत्यागम HTP_OP_CPY;
        पक्ष GGML_OP_CONT:            प्रत्यागम HTP_OP_CPY;
        पक्ष GGML_OP_GET_ROWS:        प्रत्यागम HTP_OP_GET_ROWS;
        पक्ष GGML_OP_SET_ROWS:        प्रत्यागम HTP_OP_SET_ROWS;
        पक्ष GGML_OP_SUM:             प्रत्यागम HTP_OP_SUM;
        पक्ष GGML_OP_SUM_ROWS:        प्रत्यागम HTP_OP_SUM_ROWS;
        पक्ष GGML_OP_POOL_2D:         प्रत्यागम HTP_OP_POOL_2D;
        पक्ष GGML_OP_POOL_1D:         प्रत्यागम HTP_OP_POOL_1D;
        पक्ष GGML_OP_ARGSORT:         प्रत्यागम HTP_OP_ARGSORT;
        पक्ष GGML_OP_शीर्ष_K:           प्रत्यागम HTP_OP_शीर्ष_K;
        पक्ष GGML_OP_ARGMAX:          प्रत्यागम HTP_OP_ARGMAX;
        पक्ष GGML_OP_NORM:            प्रत्यागम HTP_OP_NORM;
        पक्ष GGML_OP_L2_NORM:         प्रत्यागम HTP_OP_L2_NORM;
        पक्ष GGML_OP_RMS_NORM:        प्रत्यागम HTP_OP_RMS_NORM;
        पक्ष GGML_OP_CONCAT:          प्रत्यागम HTP_OP_CONCAT;
        पक्ष GGML_OP_SCALE:           प्रत्यागम HTP_OP_SCALE;
        पक्ष GGML_OP_CLAMP:           प्रत्यागम HTP_OP_CLAMP;
        पक्ष GGML_OP_LEAKY_RELU:      प्रत्यागम HTP_OP_LEAKY_RELU;
        पक्ष GGML_OP_SQR:             प्रत्यागम HTP_OP_SQR;
        पक्ष GGML_OP_SQRT:            प्रत्यागम HTP_OP_SQRT;
        पक्ष GGML_OP_LOG:             प्रत्यागम HTP_OP_UNARY_LOG;
        पक्ष GGML_OP_SOFT_MAX:        प्रत्यागम HTP_OP_SOFTMAX;
        पक्ष GGML_OP_SSM_CONV:        प्रत्यागम HTP_OP_SSM_CONV;
        पक्ष GGML_OP_GATED_DELTA_NET: प्रत्यागम HTP_OP_GATED_DELTA_NET;
        पक्ष GGML_OP_ROPE:            प्रत्यागम HTP_OP_ROPE;
        पक्ष GGML_OP_REPEAT:          प्रत्यागम HTP_OP_REPEAT;
        पक्ष GGML_OP_CUMSUM:          प्रत्यागम HTP_OP_CUMSUM;
        पक्ष GGML_OP_FILL:            प्रत्यागम HTP_OP_FILL;
        पक्ष GGML_OP_DIAG:            प्रत्यागम HTP_OP_DIAG;
        पक्ष GGML_OP_SOLVE_TRI:       प्रत्यागम HTP_OP_SOLVE_TRI;
        पक्ष GGML_OP_TRI:             प्रत्यागम HTP_OP_TRI;
        पक्ष GGML_OP_PAD:             प्रत्यागम HTP_OP_PAD;
        पक्ष GGML_OP_IM2COL:          प्रत्यागम HTP_OP_IM2COL;
        पक्ष GGML_OP_IM2COL_3D:       प्रत्यागम HTP_OP_IM2COL_3D;
        पक्ष GGML_OP_ROLL:            प्रत्यागम HTP_OP_ROLL;

        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(t)) {
                पक्ष GGML_UNARY_OP_SILU:       प्रत्यागम HTP_OP_UNARY_SILU;
                पक्ष GGML_UNARY_OP_GELU:       प्रत्यागम HTP_OP_UNARY_GELU;
                पक्ष GGML_UNARY_OP_GELU_QUICK: प्रत्यागम HTP_OP_UNARY_GELU_QUICK;
                पक्ष GGML_UNARY_OP_GELU_ERF:   प्रत्यागम HTP_OP_UNARY_GELU_ERF;
                पक्ष GGML_UNARY_OP_SIGMOID:    प्रत्यागम HTP_OP_UNARY_SIGMOID;
                पक्ष GGML_UNARY_OP_NEG:        प्रत्यागम HTP_OP_UNARY_NEG;
                पक्ष GGML_UNARY_OP_EXP:        प्रत्यागम HTP_OP_UNARY_EXP;
                पक्ष GGML_UNARY_OP_SOFTPLUS:   प्रत्यागम HTP_OP_UNARY_SOFTPLUS;
                पक्ष GGML_UNARY_OP_TANH:       प्रत्यागम HTP_OP_UNARY_TANH;
                पक्ष GGML_UNARY_OP_ABS:        प्रत्यागम HTP_OP_UNARY_ABS;
                पक्ष GGML_UNARY_OP_RELU:       प्रत्यागम HTP_OP_UNARY_RELU;
                पक्ष GGML_UNARY_OP_STEP:       प्रत्यागम HTP_OP_UNARY_STEP;
            स्वभाव:
                भङ्ग;
            }
            भङ्ग;

        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(t)) {
                पक्ष GGML_GLU_OP_SWIGLU:       प्रत्यागम HTP_OP_GLU_SWIGLU;
                पक्ष GGML_GLU_OP_SWIGLU_OAI:   प्रत्यागम HTP_OP_GLU_SWIGLU_OAI;
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP: प्रत्यागम HTP_OP_GLU_SWIGLU_CLAMP;
                पक्ष GGML_GLU_OP_GEGLU:        प्रत्यागम HTP_OP_GLU_GEGLU;
                पक्ष GGML_GLU_OP_GEGLU_QUICK:  प्रत्यागम HTP_OP_GLU_GEGLU_QUICK;
                पक्ष GGML_GLU_OP_GEGLU_ERF:    प्रत्यागम HTP_OP_GLU_GEGLU_ERF;
                स्वभाव: भङ्ग;
            }
            भङ्ग;

        स्वभाव:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    प्रत्यागम HTP_OP_INVALID;
}

स्थिर अन्तःस्थ तार्किक op_is_compute(ggml_tensor *गाठ)
{
    प्रत्यागम !ggml_op_is_empty(गाठ->op) && !ggml_is_empty(गाठ) && (गाठ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

स्थिर तार्किक mm_is_hmx_eligible(ध्रुव ggml_tensor * t) {
    यदि (opt_nhmx == 0) { प्रत्यागम असत्य; }

    ध्रुव ggml_tensor * src0 = t->src[0];
    ध्रुव ggml_tensor * src1 = t->src[1];

    यदि (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ध्रुव ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        प्रत्यागम ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, असत्य, असत्य);
    }

    ध्रुव तार्किक is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ध्रुव तार्किक is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    प्रत्यागम ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

स्थिर तार्किक is_supported_mul_mat_nx_kernel(ध्रुव ggml_tensor * src0, ध्रुव संरचना htp_mm_kernel_params * kparams) {
    यदि (kparams->n_hmx) {
        प्रत्यागम kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    यदि (!ggml_hexagon_tensor_is_tiled(src0)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

स्थिर तार्किक is_supported_mul_mat_id_nx_kernel(ध्रुव ggml_tensor * src0, ध्रुव संरचना htp_mm_kernel_params * kparams) {
    यदि (kparams->n_hmx) {
        प्रत्यागम kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    यदि (!ggml_hexagon_tensor_is_tiled(src0)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

स्थिर तार्किक is_mergeable_mul_mat(ध्रुव ggml_tensor * t) {
    यदि (t->op != GGML_OP_MUL_MAT) प्रत्यागम असत्य;
    यदि (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) प्रत्यागम असत्य;

    ध्रुव ggml_tensor * src0 = t->src[0];
    ध्रुव ggml_tensor * src1 = t->src[1];
    यदि (src0->ne[2] != 1 || src0->ne[3] != 1) प्रत्यागम असत्य;

    यदि (mm_is_hmx_eligible(t)) {
        प्रत्यागम htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    यदि (src1->type != GGML_TYPE_F32) प्रत्यागम असत्य;

    प्रत्यागम ggml_hexagon_tensor_is_tiled(src0);
}

स्थिर तार्किक is_mergeable_mul_mat_pair(ध्रुव ggml_tensor * n1, ध्रुव ggml_tensor * n2) {
    यदि (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[1] != n2->src[1]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[0]->type != n2->src[0]->type) {
        प्रत्यागम असत्य;
    }
    यदि (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        प्रत्यागम असत्य;
    }
    यदि (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक is_mergeable_mul_mat_id(ध्रुव ggml_tensor * t) {
    यदि (t->op != GGML_OP_MUL_MAT_ID) प्रत्यागम असत्य;

    ध्रुव ggml_tensor * src0 = t->src[0];
    यदि (mm_is_hmx_eligible(t)) {
        प्रत्यागम htp_is_hmx_weight_type(src0->type);
    }

    प्रत्यागम ggml_hexagon_tensor_is_tiled(src0);
}

स्थिर तार्किक is_mergeable_mul_mat_id_pair(ध्रुव ggml_tensor * n1, ध्रुव ggml_tensor * n2) {
    यदि (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[1] != n2->src[1]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[2] != n2->src[2]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        प्रत्यागम असत्य;
    }
    यदि (n1->src[0]->type != n2->src[0]->type) {
        प्रत्यागम असत्य;
    }
    यदि (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        प्रत्यागम असत्य;
    }
    यदि (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    यदि (sess->last_error > HTP_STATUS_OK) {
        प्रत्यागम GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ध्रुव std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    तार्किक cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    यदि (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } अन्यथा {
        // Tag fusable tensors in graph
        प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            यदि (!extra) अनुवर्त;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            यदि (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } अन्यथा यदि ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                यदि ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_गाठ_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            यदि (!op_is_compute(n)) {
                अनुवर्त;
            }

            htp_opnode गाठ(HTP_OP_INVALID, n);
            गाठ.opcode = op_remap_to_htp(n);
            यदि (गाठ.opcode == HTP_OP_MUL_MAT || गाठ.opcode == HTP_OP_MUL_MAT_ID || गाठ.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = गाठ.गाठ->src[0];
                यदि (ggml_hexagon_needs_repack(src0)) {
                    यदि (opt_verbose) {
                        वर्ण dims_buf[64];
                        वर्ण strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, परिमाण(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, परिमाण(strides_buf), src0);
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

            यदि (गाठ.opcode == HTP_OP_MUL_MAT || गाठ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.गाठ,
                    (संरचना htp_mm_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_MUL || गाठ.opcode == HTP_OP_ADD || गाठ.opcode == HTP_OP_ADD_ID || गाठ.opcode == HTP_OP_SUB || गाठ.opcode == HTP_OP_DIV) {
                ध्रुव ggml_tensor * src1 = गाठ.गाठ->src[1];
                ध्रुव ggml_tensor * src2 = (गाठ.opcode == HTP_OP_ADD_ID) ? गाठ.गाठ->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_द्विमान_params(sess,
                    गाठ.opcode, गाठ.गाठ->src[0], src1, गाठ.गाठ,
                    (संरचना htp_द्विमान_kernel_params *) गाठ.kernel_params, src2));
            } अन्यथा यदि (गाठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_fa_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (htp_op_is_unary(गाठ.opcode)) {
                auto inputs = गाठ.get_inputs();
                ध्रुव संरचना ggml_tensor * src0 = inputs[0];
                ध्रुव संरचना ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    गाठ.opcode, src0, src1, गाठ.dst(),
                    (संरचना htp_unary_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_get_rows_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_set_rows_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_rope_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_ssm_conv_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_softmax_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_gdn_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_ARGSORT || गाठ.opcode == HTP_OP_शीर्ष_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    गाठ.गाठ,
                    गाठ.opcode == HTP_OP_शीर्ष_K,
                    (संरचना htp_sort_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_POOL_2D || गाठ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, गाठ.गाठ->src[0], गाठ.dst(),
                    (संरचना htp_pool_2d_kernel_params *)गाठ.kernel_params,
                    गाठ.opcode == HTP_OP_POOL_1D);
            } अन्यथा यदि (गाठ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_concat_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_CPY || गाठ.opcode == HTP_OP_CPY_FENCE) {
                ध्रुव तार्किक ok = ggml_hexagon_precompute_cpy_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_copy_kernel_params *) गाठ.kernel_params
                );
                ध्रुव auto * kparams = (ध्रुव संरचना htp_copy_kernel_params *) गाठ.kernel_params;
                यदि (ok && गाठ.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    अनुवर्त;
                }
            } अन्यथा यदि (गाठ.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_im2col_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा यदि (गाठ.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_im2col_kernel_params *) गाठ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(गाठ));
        }

        यदि (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } अन्यथा {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    प्रति (ध्रुव auto & गाठ : *nodes_ptr) {
        sess->enqueue_op(गाठ);
    }

    यदि (sess->last_error > HTP_STATUS_OK) {
        प्रत्यागम GGML_STATUS_FAILED;
    }

    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    यदि (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

गणना ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

संरचना ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

संरचना ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    शून्य reset() {
        ranges.clear();
    }

    शून्य add(ध्रुव ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    तार्किक check(ध्रुव ggml_hexagon_mem_range & mr) ध्रुव {
        प्रति (ध्रुव auto & cmp : ranges) {
            यदि (mr.pb != cmp.pb) {
                अनुवर्त;
            }
            यदि (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                अनुवर्त;
            }
            यदि (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                प्रत्यागम असत्य;
            }
        }
        प्रत्यागम सत्य;
    }
};

स्थिर ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ध्रुव ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ध्रुव ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    यदि (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } अन्यथा {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    प्रत्यागम mr;
}

स्थिर शून्य ggml_hexagon_mem_ranges_add_गाठ(ggml_hexagon_mem_ranges & mrs, ध्रुव htp_opnode & गाठ) {
    यदि (गाठ.is_empty()) प्रत्यागम;

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        यदि (गाठ.गाठ->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(गाठ.गाठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    प्रति (ध्रुव auto * fused : गाठ.fused) {
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            यदि (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(गाठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

स्थिर तार्किक ggml_hexagon_mem_ranges_check_गाठ(ध्रुव ggml_hexagon_mem_ranges & mrs, ध्रुव htp_opnode & गाठ) {
    यदि (गाठ.is_empty()) प्रत्यागम सत्य;

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        यदि (गाठ.गाठ->src[i]) {
            यदि (!mrs.check(ggml_hexagon_mem_range_from_tensor(गाठ.गाठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                प्रत्यागम असत्य;
            }
        }
    }
    प्रति (ध्रुव auto * fused : गाठ.fused) {
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            यदि (fused->src[i]) {
                यदि (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    प्रत्यागम असत्य;
                }
            }
        }
    }
    प्रत्यागम mrs.check(ggml_hexagon_mem_range_from_tensor(गाठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

स्थिर std::vector<पूर्णाङ्क> ggml_hexagon_graph_optimize_reorder(ध्रुव std::vector<htp_opnode> & nodes) {
    ध्रुव पूर्णाङ्क n = nodes.size();

    std::vector<पूर्णाङ्क> res;
    res.reserve(n);

    std::vector<तार्किक> used(n, असत्य);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    प्रति (पूर्णाङ्क i0 = 0; i0 < n; i0++) {
        यदि (used[i0]) {
            अनुवर्त;
        }

        ध्रुव auto & node0 = nodes[i0];

        यदि (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = सत्य;
            अनुवर्त;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr पूर्णाङ्क N_FORWARD = 16;

        std::vector<पूर्णाङ्क> चळत;
        चळत.push_back(i0);

        mrs.reset();

        प्रति (पूर्णाङ्क i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            यदि (used[i1]) {
                अनुवर्त;
            }

            ध्रुव auto & node1 = nodes[i1];

            यदि (node1.stackable() && node1.same_आदान(node0) && ggml_hexagon_mem_ranges_check_गाठ(mrs, node1)) {
                चळत.push_back(i1);
            } अन्यथा {
                ggml_hexagon_mem_ranges_add_गाठ(mrs, node1);
            }
        }

        प्रति (पूर्णाङ्क idx : चळत) {
            res.push_back(idx);
            used[idx] = सत्य;
        }
    }

    प्रत्यागम res;
}

स्थिर शून्य ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    ध्रुव पूर्णाङ्क n = gf->n_nodes;

    constexpr पूर्णाङ्क MAX_FUSE = 16;

    गणना ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        htp_opnode गाठ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        यदि (गाठ.op() == GGML_OP_ADD ||
            गाठ.op() == GGML_OP_NORM ||
            गाठ.op() == GGML_OP_RMS_NORM) {
            ops[0] = गाठ.op();

            पूर्णाङ्क f = i + 1;
            यावत् (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                यदि (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    भङ्ग;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            प्रति (; f > 1; f--) {
                यदि (ggml_can_fuse(gf, i, ops, f)) {
                    भङ्ग;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            प्रति (पूर्णाङ्क k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                गाठ.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(गाठ));
    }

    ध्रुव auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        पूर्णाङ्क j = 0;
        प्रति (ध्रुव auto i : order) {
            ध्रुव auto & गाठ = nodes[i];

            gf->nodes[j++] = गाठ.गाठ;

            प्रति (auto * fused : गाठ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

स्थिर uint64_t ggml_hexagon_session_किल्ली(ध्रुव ggml_hexagon_session * sess) {
    प्रत्यागम ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

स्थिर तार्किक ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    यदि (!sess_src->clone_buffer(sbuf_dst)) { प्रत्यागम असत्य; }

    ध्रुव uint64_t src_किल्ली = ggml_hexagon_session_किल्ली(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_किल्ली];
    यदि (!fence_slot) {
        fence_slot = (चञ्चल uint32_t *) sess_dst->alloc_fence(1);
    }

    यदि (!sess_src->clone_buffer(sess_dst->fence_buf)) { प्रत्यागम असत्य; }

    यदि (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (शून्य *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = परिमाण(int32_t);
    fence_tensor.nb[1]  = परिमाण(int32_t);
    fence_tensor.nb[2]  = परिमाण(int32_t);
    fence_tensor.nb[3]  = परिमाण(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ सत्य);

    sess_dst->add_peer(sess_src);

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    यदि (!sess_dst->clone_buffer(sbuf_src)) { प्रत्यागम असत्य; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    यदि (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        प्रत्यागम असत्य;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    यदि (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ध्रुव auto * src_extra = static_cast<ध्रुव ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    यदि (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        प्रत्यागम सत्य;
    }

    यदि (sess_src->phys_idx != sess_dst->phys_idx)
        प्रत्यागम ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    प्रत्यागम ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

स्थिर ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = नूतन ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (चञ्चल uint32_t *) sess->alloc_fence(1);

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (शून्य *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = परिमाण(int32_t);
    hex_event->fence_tensor.nb[1]  = परिमाण(int32_t);
    hex_event->fence_tensor.nb[2]  = परिमाण(int32_t);
    hex_event->fence_tensor.nb[3]  = परिमाण(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (शून्य *)hex_event, (शून्य *)hex_event->fence_slot);

    प्रत्यागम नूतन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

स्थिर शून्य ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    यदि (hex_event->seq == 0) {
        प्रत्यागम;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<ध्रुव चञ्चल std::atomic<uint32_t> *>(hex_event->fence_slot);

    यदि ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    यावत् (सत्य) {
        यदि ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            यदि (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            भङ्ग;
        }
        std::this_thread::yield();
    }
}

स्थिर शून्य ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (शून्य *)hex_event);
    hex_event->fence_sess->free_fence((शून्य *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

स्थिर शून्य ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

स्थिर शून्य ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    यदि (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ असत्य);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);
}

स्थिर शून्य ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    यदि (hex_event->seq == 0) {
        प्रत्यागम;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    यदि (sess->phys_idx == hex_event->sess->phys_idx) {
        यदि (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        प्रत्यागम;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ सत्य);
}

स्थिर शून्य ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, संरचना ggml_tensor * tensor, ध्रुव शून्य * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (पूर्णाङ्क) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ध्रुव संरचना ggml_tensor * tensor, शून्य * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (पूर्णाङ्क) tensor->buffer->usage : -1);
    sess->flush_sync();
    यदि (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     संरचना ggml_tensor * tensor,
                                                     ध्रुव शून्य * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (पूर्णाङ्क) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

स्थिर शून्य ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ध्रुव संरचना ggml_tensor * tensor,
                                                     शून्य * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (पूर्णाङ्क) tensor->buffer->usage : -1);
    sess->flush_sync();
    यदि (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

स्थिर संरचना ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

स्थिर ggml_guid_t ggml_backend_hexagon_guid() {
    स्थिर ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    प्रत्यागम &guid;
}

तार्किक ggml_backend_is_hexagon(ggml_backend_t backend) {
    प्रत्यागम backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

स्थिर ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    प्रत्यागम नूतन ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

स्थिर ध्रुव वर्ण * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    प्रत्यागम dev_ctx->c_name();

    GGML_UNUSED(dev);
}

स्थिर ध्रुव वर्ण * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    प्रत्यागम "Hexagon";
    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * मुक्त, size_t * total) {
    *मुक्त  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ सत्य,
        /* .mmap_support          = */ असत्य,
    };
}

स्थिर ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    प्रत्यागम &dev_ctx->buffer_type;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    यदि (!opt_hostbuf) {
        प्रत्यागम रिक्त;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    प्रत्यागम &dev_ctx->host_buffer_type;
}

स्थिर तार्किक ggml_hexagon_supported_cpy(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    संरचना htp_copy_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_cont(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    संरचना htp_copy_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_repeat(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    GGML_UNUSED(sess);
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    // Support f32 and f16
    यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) प्रत्यागम असत्य;

    // src and dst must be the same type
    यदि (src0->type != dst->type) प्रत्यागम असत्य;

    // dst dims must be multiples of src dims
    यदि (dst->ne[0] % src0->ne[0] != 0) प्रत्यागम असत्य;
    यदि (dst->ne[1] % src0->ne[1] != 0) प्रत्यागम असत्य;
    यदि (dst->ne[2] % src0->ne[2] != 0) प्रत्यागम असत्य;
    यदि (dst->ne[3] % src0->ne[3] != 0) प्रत्यागम असत्य;

    // require contiguous tensors (no transposition)
    यदि (ggml_is_transposed(src0) || ggml_is_transposed(dst)) प्रत्यागम असत्य;

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_concat(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    संरचना htp_concat_kernel_params kparams;
    प्रत्यागम ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_fill(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * dst = op;

    यदि (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_roll(ध्रुव संरचना ggml_hexagon_session * sess, ध्रुव संरचना ggml_tensor * op) {
    GGML_UNUSED(sess);

    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * dst  = op;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_are_same_shape(src0, dst)) {
        प्रत्यागम असत्य;
    }

    यदि (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
स्थिर तार्किक ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    यदि (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        प्रत्यागम असत्य;
    }

    तार्किक supp = असत्य;
    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
            supp = सत्य;
            भङ्ग;

        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_ADD:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_DIV:
            supp = ggml_hexagon_supported_द्विमान(sess, op);
            भङ्ग;

        पक्ष GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            भङ्ग;

        पक्ष GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            भङ्ग;

        पक्ष GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            भङ्ग;

        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_SCALE:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            भङ्ग;

        पक्ष GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            भङ्ग;

        पक्ष GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            भङ्ग;

        पक्ष GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            भङ्ग;

        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(op)) {
                पक्ष GGML_UNARY_OP_NEG:
                पक्ष GGML_UNARY_OP_EXP:
                पक्ष GGML_UNARY_OP_SIGMOID:
                पक्ष GGML_UNARY_OP_SOFTPLUS:
                पक्ष GGML_UNARY_OP_TANH:
                पक्ष GGML_UNARY_OP_ABS:
                पक्ष GGML_UNARY_OP_SILU:
                पक्ष GGML_UNARY_OP_GELU:
                पक्ष GGML_UNARY_OP_GELU_QUICK:
                पक्ष GGML_UNARY_OP_GELU_ERF:
                पक्ष GGML_UNARY_OP_RELU:
                पक्ष GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    भङ्ग;
                स्वभाव:
                    supp = असत्य;
                    भङ्ग;
            }
            भङ्ग;

        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(op)) {
                पक्ष GGML_GLU_OP_SWIGLU:
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                पक्ष GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU_QUICK:
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    भङ्ग;
                स्वभाव:
                    supp = असत्य;
                    भङ्ग;
            }
            भङ्ग;

        पक्ष GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            भङ्ग;

        पक्ष GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            भङ्ग;

        पक्ष GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            भङ्ग;

        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            भङ्ग;

        पक्ष GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            भङ्ग;

        पक्ष GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            भङ्ग;

        पक्ष GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            भङ्ग;

        पक्ष GGML_OP_शीर्ष_K:
            supp = ggml_hexagon_supported_शीर्ष_k(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            भङ्ग;

        पक्ष GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            भङ्ग;

        पक्ष GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            भङ्ग;

        पक्ष GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            भङ्ग;

        पक्ष GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            भङ्ग;

        पक्ष GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            भङ्ग;

        पक्ष GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            भङ्ग;

        पक्ष GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            भङ्ग;

        पक्ष GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            भङ्ग;

        पक्ष GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            भङ्ग;

        पक्ष GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            भङ्ग;

        पक्ष GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            भङ्ग;

        स्वभाव:
            भङ्ग;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    प्रत्यागम supp;
}

स्थिर तार्किक ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    तार्किक supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    प्रत्यागम supp;
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ रिक्त,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ रिक्त,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    प्रति (size_t i = 0; i < opt_ndev; i++) {
        ध्रुव auto & cfg = opt_device_configs[i];
        यदि (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } अन्यथा {
            std::string peers_str;
            प्रति (ध्रुव auto & p : cfg.mdev_group) {
                यदि (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = नूतन ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    प्रति (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

स्थिर ध्रुव वर्ण * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    प्रत्यागम "HTP";
    GGML_UNUSED(reg);
}

स्थिर size_t ggml_backend_hexagon_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    प्रत्यागम opt_ndev;
    GGML_UNUSED(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    यदि (index >= opt_ndev || !hreg->devices[index].context) {
        प्रत्यागम nullptr;
    }

    प्रत्यागम &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

स्थिर शून्य * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    यदि (n_backends < 2 || n_backends > 4) {
        प्रत्यागम nullptr;
    }

    प्रति (size_t i = 0; i < n_backends; ++i) {
        यदि (!ggml_backend_is_hexagon(backends[i])) {
            प्रत्यागम nullptr;
        }
    }

    प्रति (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        प्रति (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            यदि (sess_i->phys_idx == sess_j->phys_idx) {
                प्रत्यागम nullptr;
            }
        }
    }

    auto * ctx = नूतन ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    प्रति (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (चञ्चल uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (शून्य *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = परिमाण(int32_t);
        ctx->fence_tensors[i].nb[1]  = परिमाण(int32_t);
        ctx->fence_tensors[i].nb[2]  = परिमाण(int32_t);
        ctx->fence_tensors[i].nb[3]  = परिमाण(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    प्रत्यागम ctx;
}

स्थिर शून्य ggml_backend_hexagon_comm_free(शून्य * comm_ctx_v) {
    यदि (!comm_ctx_v) प्रत्यागम;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    प्रति (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((शून्य *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

स्थिर तार्किक ggml_backend_hexagon_comm_allreduce_tensor(शून्य * comm_ctx_v, संरचना ggml_tensor ** tensors) {
    यदि (opt_ar_select == 0 || !comm_ctx_v) प्रत्यागम असत्य;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ध्रुव size_t n_backends = comm_ctx->n_backends;

    यदि (n_backends < 2 || n_backends > 4) प्रत्यागम असत्य;

    प्रति (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        प्रति (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            यदि (sess_i->phys_idx == sess_j->phys_idx) {
                प्रत्यागम असत्य;
            }
        }
    }

    प्रति (size_t i = 0; i < n_backends; i++) {
        यदि (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            प्रत्यागम असत्य;
        }
        यदि (tensors[i]->type != tensors[0]->type) {
            प्रत्यागम असत्य;
        }
        यदि (!ggml_is_contiguous(tensors[i])) {
            प्रत्यागम असत्य;
        }
        यदि (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            प्रत्यागम असत्य;
        }
    }

    यदि (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    प्रति (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        संरचना htp_allreduce_kernel_params kparams;
        यदि (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, असत्य, असत्य, /*is_shard_ok=*/ असत्य, &kparams)) {
            प्रत्यागम असत्य;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    प्रति (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        यदि ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    यदि (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    यदि (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    प्रति (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ध्रुव ggml_tensor *> data_tensors(n_backends);
    std::vector<ध्रुव ggml_tensor *> sync_tensors(n_backends);
    प्रति (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    प्रति (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        प्रति (size_t j = 0; j < n_backends; j++) {
            यदि (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    प्रत्यागम सत्य;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(पूर्णाङ्क main_device, ध्रुव भिन्न * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    यदि (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    यदि (!dev) प्रत्यागम nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    प्रत्यागम &dev_ctx->buffer_type;
}

स्थिर शून्य * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    GGML_UNUSED(reg);
    यदि (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_hexagon_split_buffer_type;
    }
    यदि (strcmp(name, "ggml_backend_comm_init") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_hexagon_comm_init;
    }
    यदि (strcmp(name, "ggml_backend_comm_free") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_hexagon_comm_free;
    }
    यदि (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    प्रत्यागम रिक्त;
}

template<typename T> std::vector<T> str_to_vec(ध्रुव वर्ण* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    यावत् (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    प्रत्यागम v;
}

template<typename T, पूर्णाङ्क BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    प्रति (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    प्रत्यागम str;
}

स्थिर शून्य ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, तार्किक discovery_supported, ध्रुव std::unordered_map<पूर्णाङ्क, fastrpc_domain> & cdsp_map) {
    यदि (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        यदि (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } अन्यथा {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } अन्यथा {
        विकल्प (cfg.physical_idx) {
            पक्ष 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                भङ्ग;
            पक्ष 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                भङ्ग;
            स्वभाव:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                भङ्ग;
        }
    }
    प्रति (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
स्थिर शून्य ggml_hexagon_discover_devices() {
    std::unordered_map<पूर्णाङ्क, fastrpc_domain> cdsp_map;
    तार्किक discovery_supported = असत्य;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    पूर्णाङ्क err = remote_system_request(&domain_info);
    यदि (err == AEE_SUCCESS && domain_info.sys.अंक_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.अंक_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (पूर्णाङ्क) domains.size();

        err = remote_system_request(&domain_info);
        यदि (err == AEE_SUCCESS) {
            discovery_supported = सत्य;
            ध्रुव पूर्णाङ्क n_domains = std::min(domain_info.sys.अंक_domains, (पूर्णाङ्क) domains.size());
            प्रति (पूर्णाङ्क i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (पूर्णाङ्क) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                यदि (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (पूर्णाङ्क) domains[i].type);
                    अनुवर्त;
                }
                यदि (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    अनुवर्त;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } अन्यथा यदि (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    प्रति (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

स्थिर शून्य ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q4_0 == (unsigned पूर्णाङ्क) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q4_1 == (unsigned पूर्णाङ्क) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q8_0 == (unsigned पूर्णाङ्क) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_MXFP4 == (unsigned पूर्णाङ्क) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_IQ4_NL == (unsigned पूर्णाङ्क) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q4_K == (unsigned पूर्णाङ्क) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q5_K == (unsigned पूर्णाङ्क) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q6_K == (unsigned पूर्णाङ्क) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q3_K == (unsigned पूर्णाङ्क) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q2_K == (unsigned पूर्णाङ्क) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q1_0 == (unsigned पूर्णाङ्क) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_Q2_0 == (unsigned पूर्णाङ्क) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned पूर्णाङ्क) HTP_TYPE_BF16 == (unsigned पूर्णाङ्क) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ध्रुव वर्ण * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ध्रुव वर्ण * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ध्रुव वर्ण * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ध्रुव वर्ण * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ध्रुव वर्ण * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ध्रुव वर्ण * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ध्रुव वर्ण * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ध्रुव वर्ण * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ध्रुव वर्ण * str_etm      = getenv("GGML_HEXAGON_ETM");
    ध्रुव वर्ण * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ध्रुव वर्ण * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ध्रुव वर्ण * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ध्रुव वर्ण * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ध्रुव वर्ण * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ध्रुव वर्ण * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ध्रुव वर्ण * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ध्रुव वर्ण * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ध्रुव वर्ण * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ध्रुव वर्ण * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ध्रुव वर्ण * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ध्रुव वर्ण * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ध्रुव वर्ण * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ध्रुव वर्ण * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ध्रुव वर्ण * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ध्रुव वर्ण * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    यदि (!str_arch) {
        पूर्णाङ्क err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        यदि (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } अन्यथा {
            यदि (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } अन्यथा यदि (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } अन्यथा {
        यदि (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, रिक्त, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? नूतन std::regex(str_opfilter, RE_ICASE) : रिक्त;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, रिक्त, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, रिक्त, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, रिक्त, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  रिक्त, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    यदि (str_power_mode) {
        पूर्णाङ्क m = htp_power_mode_from_str(str_power_mode);
        यदि (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } अन्यथा {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, रिक्त, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    यदि (str_mbuf) {
        ध्रुव वर्ण * p = str_mbuf;
        प्रति (पूर्णाङ्क idx = 0; idx < 3 && p && *p; idx++) {
            यावत् (*p == ' ') p++;
            ध्रुव वर्ण * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            यावत् (len > 0 && p[len - 1] == ' ') len--;
            यदि (len > 0) {
                std::string token(p, len);
                यदि (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                यदि (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                यदि (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            यदि (!comma) भङ्ग;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ध्रुव वर्ण * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    यदि (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    यदि (str_devices && str_devices[0] != '\0') {
        तार्किक is_single_संख्या = सत्य;
        प्रति (पूर्णाङ्क i = 0; str_devices[i] != '\0'; i++) {
            यदि (!isdigit((unsigned वर्ण)str_devices[i])) {
                is_single_संख्या = असत्य;
                भङ्ग;
            }
        }
        यदि (is_single_संख्या) {
            पूर्णाङ्क n = atoi(str_devices);
            यदि (n < 1) n = 1;
            यदि (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            प्रति (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (पूर्णाङ्क)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } अन्यथा {
            std::string s_devices(str_devices);
            std::vector<std::string> वस्तुसंच;
            std::string curr_item;
            पूर्णाङ्क bracket_depth = 0;
            प्रति (वर्ण ch : s_devices) {
                यदि (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } अन्यथा यदि (ch == ']') {
                    यदि (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } अन्यथा यदि (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    यदि (s != std::string::npos) {
                        वस्तुसंच.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } अन्यथा {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            यदि (s != std::string::npos) {
                वस्तुसंच.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            प्रति (ध्रुव auto & item : वस्तुसंच) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                यदि (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    पूर्णाङ्क virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    यदि (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        यत्न {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } गृहाण (...) {
                            virt = 0;
                        }
                    } अन्यथा {
                        size_t dev_colon = dev_name.find(':');
                        यदि (dev_colon != std::string::npos) {
                            यत्न {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } गृहाण (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<पूर्णाङ्क> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    यावत् (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        यदि (ps == std::string::npos) अनुवर्त;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        यदि (dash_pos != std::string::npos) {
                            यत्न {
                                पूर्णाङ्क p_start = std::stoi(p_part.substr(0, dash_pos));
                                पूर्णाङ्क p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                प्रति (पूर्णाङ्क p = p_start; p <= p_end; p++) {
                                    यदि (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } गृहाण (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } अन्यथा {
                            यत्न {
                                पूर्णाङ्क p = std::stoi(p_part);
                                यदि (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } गृहाण (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    यदि (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    यदि (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        प्रति (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } अन्यथा {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } अन्यथा यदि (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    पूर्णाङ्क phys = 0;
                    पूर्णाङ्क virt = 0;
                    यत्न {
                        यदि (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } अन्यथा {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } गृहाण (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        अनुवर्त;
                    }

                    यदि (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } अन्यथा {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } अन्यथा {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } अन्यथा {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#यदि defined(__ANDROID__)
    यदि (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    यदि (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            विकल्प (v.size()) {
                पक्ष 1:  opt_profile = v[0]; प्रत्यागम opt_pmu_evt; // mode with default pmu events
                पक्ष 8:  opt_profile = 2;    प्रत्यागम v;           // mode with custom  pmu events
                स्वभाव: opt_profile = 0;    प्रत्यागम {};          // garbage input
            }}();
        यदि (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = नूतन ggml_hexagon_registry(reg);
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_मोजणी,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(शून्य) {
    स्थिर तार्किक initialized = असत्य;

    स्थिर ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ रिक्त };

    {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        यदि (!initialized) {
            auto nErr = htpdrv_init();
            यदि (nErr != AEE_SUCCESS) {
                प्रत्यागम रिक्त;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = सत्य;
    }

    प्रत्यागम &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
