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

using intvec  = std::vector<पूर्णांक>;
using uintvec = std::vector<अचिन्हित पूर्णांक>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

संरचना ggml_hexagon_device_config {
    पूर्णांक         physical_idx = 0;
    पूर्णांक         virtual_idx  = 0;
    पूर्णांक         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

स्थिर ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

स्थिर पूर्णांक    opt_arch    = 0; // autodetect
स्थिर माप_प्रकार opt_ndev    = 1;
स्थिर माप_प्रकार opt_nhvx    = 0; // use all
स्थिर पूर्णांक    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
स्थिर माप_प्रकार opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
स्थिर पूर्णांक    opt_etm     = 0;
स्थिर पूर्णांक    opt_verbose = 0;
स्थिर पूर्णांक    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
स्थिर तार्किक   opt_hostbuf = असत्य;
स्थिर तार्किक   opt_dma64   = असत्य;
स्थिर पूर्णांक    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

स्थिर स्थिरांक std::pair<स्थिरांक अक्षर *, पूर्णांक> htp_power_mode_names[] = {
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

स्थिर पूर्णांक htp_power_mode_from_str(स्थिरांक अक्षर * s) {
    साठी (स्वतः & e : htp_power_mode_names) {
#जर defined(_MSC_VER)
        जर (_stricmp(s, e.first) == 0) {
#अन्यथा
        जर (strcasecmp(s, e.first) == 0) {
#endif
            परत e.second;
        }
    }
    परत -1;
}

स्थिर स्थिरांक अक्षर * htp_power_mode_to_str(पूर्णांक mode) {
    साठी (स्वतः & e : htp_power_mode_names) {
        जर (e.second == mode) {
            परत e.first;
        }
    }
    परत "unknown";
}

स्थिर माप_प्रकार opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
स्थिर माप_प्रकार opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
स्थिर माप_प्रकार opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

स्थिर पूर्णांक    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
स्थिर पूर्णांक    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
स्थिर पूर्णांक    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
स्थिर पूर्णांक    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
स्थिर पूर्णांक    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
स्थिर पूर्णांक    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
स्थिर u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

स्थिर पूर्णांक opt_opbatch  = 1280; // max number of ops in a batch
स्थिर पूर्णांक opt_opqueue  = 8;   // max number of pending batches
स्थिर पूर्णांक opt_optrace  = 0;    // trace buffer size per thread (0 means default)
स्थिर पूर्णांक opt_oppoll   = 0;    // polling for batch completions
स्थिर पूर्णांक opt_opfusion  = 1;    // enable/disable op fusion
स्थिर पूर्णांक opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

गणना ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

स्थिर ओळीत तार्किक ggml_hexagon_is_fusion_enabled(पूर्णांक flag) {
    जर (opt_opfusion <= 0) परत असत्य;
    जर (opt_opfusion == 1) परत सत्य; // 1 enables all
    परत (opt_opfusion & flag) != 0;
}

स्थिर std::regex* opt_opfilter = रिक्त; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    जर (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

स्थिर स्थिरांक अक्षर * status_to_str(uint32_t status) {
    निवड (status) {
        स्थिती HTP_STATUS_OK:
            परत "OK";
        स्थिती HTP_STATUS_NO_SUPPORT:
            परत "NO-SUPPORT";
        स्थिती HTP_STATUS_INVAL_PARAMS:
            परत "INVAL-PARAMS";
        स्थिती HTP_STATUS_VTCM_TOO_SMALL:
            परत "VTCM-TOO-SMALL";
        स्थिती HTP_STATUS_INTERNAL_ERR:
            परत "INTERNAL-ERROR";
        पूर्वनिर्धारित:
            परत "UNKNOWN";
    }
}

// ** debug helpers

स्थिर शून्य ggml_hexagon_dump_op_exec(स्थिरांक std::string &sess_name, स्थिरांक htp_opnode & गाठ, स्थिरांक uint32_t req_flags) {
    जर (!opt_verbose) परत;

    htp_opformat fmt(गाठ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                गाठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

स्थिर शून्य ggml_hexagon_dump_op_supp(स्थिरांक std::string &sess_name, स्थिरांक संरचना ggml_tensor * op, तार्किक supp) {
    जर (!opt_verbose) परत;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

स्थिर स्थिरांक अक्षर * htp_event_name(uint16_t id) {
    निवड (id) {
        स्थिती HTP_TRACE_EVT_DMA:            परत "DMA";
        स्थिती HTP_TRACE_EVT_HVX_COMP:       परत "HVX_COMP";
        स्थिती HTP_TRACE_EVT_HVX_A_QUANT:    परत "HVX_A_QUANT";
        स्थिती HTP_TRACE_EVT_HVX_A_PREP:     परत "HVX_A_PREP";
        स्थिती HTP_TRACE_EVT_HVX_W_DEQUANT:  परत "HVX_W_DEQUANT";
        स्थिती HTP_TRACE_EVT_HVX_W_PREP:     परत "HVX_W_PREP";
        स्थिती HTP_TRACE_EVT_HVX_O_PROC:     परत "HVX_O_PROC";
        स्थिती HTP_TRACE_EVT_HVX_FA_QK:      परत "HVX_QK_FA";
        स्थिती HTP_TRACE_EVT_HVX_FA_SFM:     परत "HVX_SFM_FA";
        स्थिती HTP_TRACE_EVT_HVX_FA_Q_PREP:  परत "HVX_Q_PREP";
        स्थिती HTP_TRACE_EVT_HVX_FA_K_PREP:  परत "HVX_K_PREP";
        स्थिती HTP_TRACE_EVT_HVX_FA_V_PREP:  परत "HVX_V_PREP";
        स्थिती HTP_TRACE_EVT_HVX_GDN_PREP:   परत "HVX_GDN_PREP";
        स्थिती HTP_TRACE_EVT_HVX_GDN_SOLVE:  परत "HVX_GDN_SOLVE";
        स्थिती HTP_TRACE_EVT_HVX_GDN_V_PREP: परत "HVX_GDN_V_PREP";
        स्थिती HTP_TRACE_EVT_HVX_GDN_D_PREP: परत "HVX_GDN_D_PREP";
        स्थिती HTP_TRACE_EVT_HVX_GDN_OUT:    परत "HVX_GDN_OUT";
        स्थिती HTP_TRACE_EVT_HVX_GDN_STATE:  परत "HVX_GDN_STATE";
        स्थिती HTP_TRACE_EVT_HVX_GDN_REM:    परत "HVX_GDN_REM";
        स्थिती HTP_TRACE_EVT_HMX_COMP:       परत "HMX_COMP";
        स्थिती HTP_TRACE_EVT_L2FLUSH:        परत "L2FLUSH";
        स्थिती HTP_TRACE_EVT_INIT:           परत "INIT";
        स्थिती HTP_TRACE_EVT_BUFF:           परत "BUFF";
        स्थिती HTP_TRACE_EVT_FENCE:          परत "FENCE";
        पूर्वनिर्धारित:                           परत "UNKNOWN";
    }
}

स्थिर शून्य ggml_hexagon_dump_op_prof(स्थिरांक std::string &sess_name, स्थिरांक htp_opnode & गाठ, स्थिरांक htp_prof_desc & pd) {
    जर (!opt_profile) परत;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    स्थिरांक uint32_t * pmu = pd.pmu;

    अक्षर pmu_str[256] = "";
    जर (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, आकार(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(गाठ);
    दशांश mhz = op_usec > 0 ? (दशांश) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            गाठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

स्थिर शून्य ggml_hexagon_dump_batch_prof(स्थिरांक std::string & sess_name, स्थिरांक htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    दशांश batch_mhz = rsp.usecs > 0 ? (दशांश) batch_cycles / rsp.usecs : 0.0f;

    अक्षर evt_str[256] = "----";
    जर (opt_profile == 3) {
        snprintf(evt_str, आकार(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (अचिन्हित दीर्घ दीर्घ) batch_cycles, (अचिन्हित दीर्घ दीर्घ) rsp.cycles_start, batch_mhz);
}

स्थिर शून्य ggml_hexagon_dump_trace_events(स्थिरांक std::string & sess_name, स्थिरांक htp_opbatch_rsp & rsp,
                                           स्थिरांक htp_trace_desc * trace_events, uint32_t n_traces) {
    जर (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        साठी (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        साठी (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            साठी (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                स्थिरांक स्वतः & e = trace_events[t * n_traces + idx];
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
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_matmul_add_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * src2,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_unary_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    uint32_t op,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_unary_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_द्विमान_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    uint32_t op,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_द्विमान_kernel_params * kparams,
    स्थिरांक संरचना ggml_tensor * src2 = nullptr
);

स्थिर शून्य ggml_hexagon_precompute_get_rows_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_get_rows_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_set_rows_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_set_rows_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_softmax_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_softmax_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_rope_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_rope_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_ssm_conv_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_ssm_conv_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_gated_delta_net_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_gdn_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_sort_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    तार्किक is_शीर्ष_k,
    संरचना htp_sort_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_pool_2d_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_pool_2d_kernel_params * kparams,
    तार्किक is_pool_1d
);

स्थिर तार्किक ggml_hexagon_precompute_concat_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_concat_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_cpy_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_copy_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_im2col_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_im2col_3d_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_mmnx_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * act,
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर शून्य ggml_hexagon_precompute_fused_mmidnx_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * act,
    स्थिरांक संरचना ggml_tensor * dst,
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_precompute_allreduce_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    तार्किक has_add,
    तार्किक is_row_bcast,
    तार्किक is_shard_ok,
    संरचना htp_allreduce_kernel_params * kparams
);

स्थिर तार्किक ggml_hexagon_rows_stride(स्थिरांक int64_t * ne, स्थिरांक माप_प्रकार * nb, माप_प्रकार * stride);
स्थिर तार्किक ggml_hexagon_matmul_can_collapse(स्थिरांक संरचना ggml_tensor * src0, स्थिरांक संरचना ggml_tensor * src1, स्थिरांक संरचना ggml_tensor * dst);
स्थिर ggml_tensor ggml_hexagon_tensor_collapse_rows(स्थिरांक संरचना ggml_tensor * t);

स्थिर तार्किक mm_is_hmx_eligible(स्थिरांक ggml_tensor * t);
स्थिर htp_op_code op_remap_to_htp(स्थिरांक ggml_tensor * t);
स्थिर तार्किक is_supported_mul_mat_nx_kernel(स्थिरांक ggml_tensor * src0, स्थिरांक संरचना htp_mm_kernel_params * kparams);
स्थिर तार्किक is_supported_mul_mat_id_nx_kernel(स्थिरांक ggml_tensor * src0, स्थिरांक संरचना htp_mm_kernel_params * kparams);
स्थिर तार्किक is_mergeable_mul_mat(स्थिरांक ggml_tensor * t);
स्थिर तार्किक is_mergeable_mul_mat_pair(स्थिरांक ggml_tensor * n1, स्थिरांक ggml_tensor * n2);
स्थिर तार्किक is_mergeable_mul_mat_id(स्थिरांक ggml_tensor * t);
स्थिर तार्किक is_mergeable_mul_mat_id_pair(स्थिरांक ggml_tensor * n1, स्थिरांक ggml_tensor * n2);

// ** backend sessions

संरचना ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    माप_प्रकार               shadow_size { 0 };
    uint32_t             flags { 0 };
};

स्थिर ओळीत तार्किक ggml_hexagon_tensor_is_fuseable(स्थिरांक संरचना ggml_tensor * t) {
    जर (!t->extra) परत असत्य;
    स्वतः extra = (स्थिरांक संरचना ggml_hexagon_tensor_extra *) t->extra;
    परत (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

स्थिर ओळीत तार्किक ggml_hexagon_can_rt_repack(स्थिरांक संरचना ggml_tensor * t) {
    जर (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        परत असत्य;
    }
    जर (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        परत असत्य;
    }
    परत opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

स्थिर ओळीत तार्किक ggml_hexagon_tensor_is_tiled(स्थिरांक संरचना ggml_tensor * t) {
    स्थिरांक स्वतः * extra = (स्थिरांक ggml_hexagon_tensor_extra *) t->extra;
    जर (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        परत सत्य;
    }
    परत ggml_hexagon_can_rt_repack(t);
}

स्थिर ओळीत तार्किक ggml_hexagon_needs_repack(स्थिरांक संरचना ggml_tensor * t) {
    स्थिरांक स्वतः * extra = (स्थिरांक ggml_hexagon_tensor_extra *) t->extra;
    जर (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        परत असत्य;
    }
    परत ggml_hexagon_can_rt_repack(t);
}

स्थिर ओळीत तार्किक ggml_hexagon_tensors_overlap(स्थिरांक संरचना ggml_tensor * a, स्थिरांक संरचना ggml_tensor * b) {
    स्थिरांक uintptr_t a0 = (uintptr_t) a->data;
    स्थिरांक uintptr_t b0 = (uintptr_t) b->data;
    स्थिरांक uintptr_t a1 = a0 + ggml_nbytes(a);
    स्थिरांक uintptr_t b1 = b0 + ggml_nbytes(b);

    परत a0 < b1 && b0 < a1;
}

स्थिर ओळीत तार्किक ggml_hexagon_can_row_partition(स्थिरांक संरचना ggml_tensor * t) {
    जर (t->ne[1] > 1 && (t->nb[1] & 127) != 0) परत असत्य;
    जर (t->ne[2] > 1 && (t->nb[2] & 127) != 0) परत असत्य;
    जर (t->ne[3] > 1 && (t->nb[3] & 127) != 0) परत असत्य;
    परत सत्य;
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
    माप_प्रकार                      n_backends = 0;
    अस्थिर uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

संरचना ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    अस्थिर uint32_t *    fence_slot   = nullptr;
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
    पूर्णांक              phys_idx;
    पूर्णांक              virt_idx;
    तार्किक             valid_session;
    तार्किक             valid_handle;
    तार्किक             valid_queue;
    तार्किक             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<पूर्णांक, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_size   = 0;
    माप_प्रकार   max_vmem    = 0;
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

    ggml_hexagon_session(स्थिरांक ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_मोजणी = 0) noexcept(असत्य);
    ~ggml_hexagon_session() noexcept(सत्य);

    स्थिरांक अक्षर* c_name() स्थिरांक { परत name.c_str(); }

    शून्य allocate(स्थिरांक ggml_hexagon_device_config & config) noexcept(असत्य);
    शून्य release() noexcept(सत्य);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    शून्य      free_fence(शून्य * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, अस्थिर uint32_t *> cpy_fence_slots;

    शून्य enqueue_mdev_group();
    शून्य enqueue_op(स्थिरांक htp_opnode & गाठ);
    शून्य enqueue_cpy(स्थिरांक ggml_tensor * src, ggml_tensor * dst, स्थिरांक ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    शून्य enqueue_fence(स्थिरांक ggml_tensor * sync_tensor, uint32_t fence_seq = 0, तार्किक रुको = सत्य);
    शून्य enqueue_allreduce(स्थिरांक ggml_tensor * dst, स्थिरांक std::vector<स्थिरांक ggml_tensor *> & src_tensors,
                           स्थिरांक std::vector<स्थिरांक ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    शून्य start_batch();
    शून्य flush_sync(तार्किक all = सत्य);
    शून्य flush_async();
    शून्य flush_batch(माप_प्रकार min_ops = 1);
    शून्य flush_peers();
    शून्य flush_pending(तार्किक all = सत्य);

    ggml_hexagon_shared_buffer * mmap_tensor(स्थिरांक ggml_tensor * t);
    तार्किक clone_buffer(स्थिरांक ggml_hexagon_shared_buffer*);
    शून्य release_buffer(स्थिरांक ggml_hexagon_shared_buffer*);
    शून्य unclone_buffer(स्थिरांक ggml_hexagon_shared_buffer*);

    शून्य add_peer(ggml_hexagon_session * peer) {
        जर (हे->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } अन्यथा {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

संरचना ggml_backend_hexagon_device_context {
    पूर्णांक                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(पूर्णांक dev_id, स्थिरांक ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    स्थिरांक अक्षर * c_name() स्थिरांक { परत config.name.c_str(); }

    ggml_hexagon_session * session() {
        जर (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        परत sess.get();
    }
};

संरचना ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(स्थिरांक std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        हे->dev_ctx = dev_ctx;
        हे->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

संरचना ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    पूर्णांक       fd   = -1;
    माप_प्रकार    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(माप_प्रकार size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        जर (!base) {
            फेका std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        जर (fd < 0) {
            rpcmem_free(base);
            फेका std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        हे->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        जर (base) {
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

    स्थिरांक अक्षर * c_name() स्थिरांक { परत sess->c_name(); }
    uint8_t *    base()   स्थिरांक { परत mem ? mem->base : nullptr; }
    माप_प्रकार       size()   स्थिरांक { परत mem ? mem->size : 0;  }
    पूर्णांक          fd()     स्थिरांक { परत mem ? mem->fd   : -1; }

    शून्य mmap(तार्किक extended = असत्य) {
        जर (!हे->mem)   परत;
        जर (हे->mapped) परत;

        GGML_ASSERT(!हे->pinned || !extended);

        हे->extended = extended;
        fastrpc_map_flags flags = हे->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        पूर्णांक err = fastrpc_mmap(sess->domain_id, fd(), (शून्य *) base(), 0, size(), flags);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (अचिन्हित) err);
            फेका std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (शून्य *) base(), size(), fd(), pinned, extended);

        हे->mapped = सत्य;
    }

    शून्य unmap() {
        जर (!हे->mapped) परत;

        जर (!हे->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        जर (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (शून्य *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (शून्य *) base(), size(), fd());

        हे->mapped = असत्य;
    }

    शून्य alloc(माप_प्रकार size) {
        जर (हे->mem) परत;

        हे->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (शून्य *) base(), हे->size(), fd(), (पूर्णांक) pinned);
        जर (हे->pinned) {
            mmap();
        }
    }

    शून्य मुक्त() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (शून्य *) base(), size(), fd());
        हे->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, माप_प्रकार size, तार्किक pinned = असत्य) {
        हे->sess     = sess;
        हे->mapped   = असत्य;
        हे->pinned   = pinned;
        हे->extended = असत्य;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        माप_प्रकार guard_offset = (size + 4095) & ~4095;
        माप_प्रकार total_size   = guard_offset + 4096;
        जर (!pinned && opt_dma64) {
            constexpr माप_प्रकार extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, स्थिरांक ggml_hexagon_shared_buffer & other) {
        हे->sess   = sess;
        हे->mem    = other.mem;
        हे->mapped = असत्य;
        हे->pinned = other.pinned;
        हे->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        मुक्त();
        साठी (स्वतः * extra : tensor_extra) {
            delete extra;
        }
    }
};

संरचना ggml_hexagon_fence_buffer : सार्वजनिक ggml_hexagon_shared_buffer {
    uint32_t              slot_मोजणी = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, माप_प्रकार size)
        : ggml_hexagon_shared_buffer(sess, size, असत्य /* pinned */),
          slot_मोजणी(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(हे);
        backend_buffer.size    = size;
        mmap(असत्य);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        जर (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (माप_प्रकार) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } अन्यथा जर (slot_head + n_slots <= slot_मोजणी) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (माप_प्रकार) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        जर (ptr) {
            स_रखो(ptr, 0, (माप_प्रकार) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        परत ptr;
    }

    शून्य free_slot(शून्य * ptr, uint32_t n_slots = 1) {
        जर (!ptr) परत;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        साठी (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

ओळीत uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    परत ptr;
}

ओळीत शून्य ggml_hexagon_session::free_fence(शून्य * ptr, uint32_t n_slots) {
    जर (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

स्थिर शून्य * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    परत sbuf->base();
}

स्थिर गणना ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    स्वतः sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (शून्य *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    स्वतः extra = नवीन ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   स्थिरांक शून्य *          data,
                                                   माप_प्रकार                offset,
                                                   माप_प्रकार                size) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    स्वतः sess  = sbuf->sess;

    जर (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        जर (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (पूर्णांक) buffer->usage, extra->flags);

    जर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        स_नकल((अक्षर *) tensor->data + offset, data, size);
        परत;
    }

    जर (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        परत;
    }

    जर (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    स_नकल(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    जर (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   स्थिरांक ggml_tensor *   tensor,
                                                   शून्य *                data,
                                                   माप_प्रकार                offset,
                                                   माप_प्रकार                size) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    स्वतः sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (पूर्णांक) buffer->usage, extra->flags);

    जर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        स_नकल(data, (स्थिरांक अक्षर *) tensor->data + offset, size);
        परत;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

स्थिर तार्किक ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   स्थिरांक संरचना ggml_tensor * src,
                                                   संरचना ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    परत असत्य;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

स्थिर शून्य ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          स्थिरांक शून्य *          data,
                                                          माप_प्रकार                offset,
                                                          माप_प्रकार                size,
                                                          माप_प्रकार                n_copies,
                                                          माप_प्रकार                stride_tensor,
                                                          माप_प्रकार                stride_data) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    स्वतः sess  = sbuf->sess;

    जर (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        जर (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (पूर्णांक) buffer->usage, extra->flags);

    जर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        साठी (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((uint8_t *) tensor->data + offset + i * stride_tensor, (स्थिरांक uint8_t *) data + i * stride_data, size);
        }
        परत;
    }

    जर (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    साठी (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल(extra->shadow_buf.data() + offset + i * stride_tensor, (स्थिरांक uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    जर (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      स्थिरांक ggml_tensor *   tensor,
                                                      शून्य *                data,
                                                      माप_प्रकार                offset,
                                                      माप_प्रकार                size,
                                                      माप_प्रकार                n_copies,
                                                      माप_प्रकार                stride_tensor,
                                                      माप_प्रकार                stride_data) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    स्वतः sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (पूर्णांक) buffer->usage);

    जर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        साठी (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((uint8_t *)data + i * stride_data, (स्थिरांक uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        परत;
    }

    माप_प्रकार temp_size      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    माप_प्रकार slice_size     = tensor->ne[1] * ggml_row_size(tensor->type, tensor->ne[0]);
    माप_प्रकार slice_offset   = offset % slice_size;
    माप_प्रकार row_size_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_size_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((temp_size % row_size_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((offset + temp_size) <= ggml_nbytes(tensor));

    std::vector<uint8_t> temp_buf(temp_size);
    htp_unpack_tensor_tiled(temp_buf.data(), tensor, offset, temp_size);

    साठी (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

स्थिर शून्य ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    स्वतः sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    स्वतः sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (शून्य *) sbuf->base(), sbuf->size());
    स_रखो(sbuf->base(), value, sbuf->size());
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
                                                        स्थिरांक शून्य *          data,
                                                        माप_प्रकार                offset,
                                                        माप_प्रकार                size) {
    स_नकल((अक्षर *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        स्थिरांक ggml_tensor *   tensor,
                                                        शून्य *                data,
                                                        माप_प्रकार                offset,
                                                        माप_प्रकार                size) {
    स_नकल(data, (स्थिरांक अक्षर *) tensor->data + offset, size);
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

स्थिर स्थिरांक अक्षर * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    परत static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, माप_प्रकार size) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    स्वतः sess    = dev_ctx->session();
    जर (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        परत nullptr;
    }
    प्रयत्न {
        ggml_hexagon_shared_buffer * sbuf = नवीन ggml_hexagon_shared_buffer(sess, size, असत्य);
        परत ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } पकडा (स्थिरांक std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        परत nullptr;
    }
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, माप_प्रकार size) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    स्वतः sess    = dev_ctx->session();
    जर (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        परत nullptr;
    }
    प्रयत्न {
        ggml_hexagon_shared_buffer * sbuf = नवीन ggml_hexagon_shared_buffer(sess, size, असत्य);
        परत ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } पकडा (स्थिरांक std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        परत nullptr;
    }
}

स्थिर माप_प्रकार ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    परत 128;  // HVX alignment
    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक संरचना ggml_tensor * t) {
    जर (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        माप_प्रकार tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        परत std::max(tiled_size, ggml_nbytes(t));
    }
    परत ggml_nbytes(t);
}

स्थिर माप_प्रकार parse_size(स्थिरांक अक्षर * str, माप_प्रकार default_unit = 1024 * 1024) {
    जर (!str || str[0] == '\0') {
        परत 0;
    }
    अक्षर * end = रिक्त;
    द्विगुण val = म_से_भग्न(str, &end);
    जर (val < 0) {
        परत 0;
    }
    जर (end && *end) {
        पर्यंत (*end == ' ') end++;
        जर (*end == 'k' || *end == 'K') {
            परत (माप_प्रकार) (val * 1024);
        } अन्यथा जर (*end == 'm' || *end == 'M') {
            परत (माप_प्रकार) (val * 1024 * 1024);
        } अन्यथा जर (*end == 'g' || *end == 'G') {
            परत (माप_प्रकार) (val * 1024 * 1024 * 1024);
        }
    }
    परत (माप_प्रकार) (val * default_unit);
}

स्थिर माप_प्रकार ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    परत opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    परत असत्य;
    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    परत सत्य;
    GGML_UNUSED(buft);
}

संरचना ggml_backend_hexagon_alloc_buffer_n_plan_item {
    माप_प्रकार size;
    पूर्णांक    first;
    पूर्णांक    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

स्थिर स्थिरांक अक्षर * ggml_hexagon_kv_layer_suffix(स्थिरांक संरचना ggml_tensor * t) {
    जर (म_भेदन(t->name, "cache_", 6) != 0) {
        परत रिक्त;
    }
    स्थिरांक अक्षर * p = म_माला(t->name, "_l");
    जर (!p || !है_अंक((अचिन्हित अक्षर)p[2])) {
        परत रिक्त;
    }
    परत p;
}

संरचना ggml_backend_hexagon_alloc_unit {
    माप_प्रकार size;
    पूर्णांक    first;
    पूर्णांक    last;
};

स्थिर ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णांक n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    स्थिरांक माप_प्रकार alignment = ggml_backend_buft_get_alignment(buft);
    स्थिरांक माप_प्रकार max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    पूर्णांक i = 0;
    पर्यंत (i < n_tensors) {
        संरचना ggml_tensor * t = tensors[i];
        माप_प्रकार unit_size = 0;
        पूर्णांक unit_first = i;
        पूर्णांक unit_last = i + 1;

        जर (t->data == रिक्त && t->view_src == रिक्त) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        स्थिरांक अक्षर * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        पर्यंत (unit_last < n_tensors) {
            संरचना ggml_tensor * next = tensors[unit_last];

            जर (next->view_src != रिक्त) {
                unit_last++;
                सुरूठेवा;
            }

            जर (layer_suffix != रिक्त) {
                स्थिरांक अक्षर * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                जर (next_suffix != रिक्त && म_भेद(layer_suffix, next_suffix) == 0) {
                    जर (next->data == रिक्त) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    सुरूठेवा;
                }
            }

            थांबा;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    माप_प्रकार cur_buf_size  = 0;
    पूर्णांक    cur_buf_first = 0;

    साठी (स्थिरांक स्वतः & unit : units) {
        जर (unit.size == 0) {
            सुरूठेवा;
        }

        जर (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    जर (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    परत plan;
}

स्थिर ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिरांक ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    साठी (स्थिरांक स्वतः & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        जर (buffer == रिक्त) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            साठी (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            परत रिक्त;
        }

        संरचना ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        संरचना ggml_tensor * t_failed = रिक्त;
        साठी (पूर्णांक j = item.first; j < item.last; j++) {
            संरचना ggml_tensor * t = tensors[j];
            जर (t->data == रिक्त) {
                जर (t->view_src == रिक्त) {
                    जर (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        थांबा;
                    }
                } अन्यथा जर (t->buffer == रिक्त) {
                    जर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        थांबा;
                    }
                }
            } अन्यथा {
                जर (t->view_src != रिक्त && t->buffer == रिक्त) {
                    जर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        थांबा;
                    }
                }
            }
        }
        जर (t_failed != रिक्त) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            साठी (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            परत रिक्त;
        }

        buffers.push_back(buffer);
    }

    जर (buffers.empty()) {
        परत रिक्त;
    }

    जर (buffers.size() == 1) {
        परत buffers[0];
    }

    परत ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

स्थिर माप_प्रकार ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिरांक ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    माप_प्रकार total = 0;
    साठी (स्थिरांक स्वतः & item : plan) {
        total += item.size;
    }
    परत total;
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

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(पूर्णांक dev_id, स्थिरांक ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = नवीन ggml_backend_hexagon_buffer_type_context(config.name, हे);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = नवीन ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", हे);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = नवीन ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", हे);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

स्थिर तार्किक ggml_backend_buffer_is_hexagon(स्थिरांक संरचना ggml_backend_buffer * b) {
    परत b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

संरचना ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<पूर्णांक, पूर्णांक>                b_map; // buffer fd   to index
    std::unordered_map<स्थिरांक ggml_tensor*, पूर्णांक> t_map; // tensor ptr  to index
    std::unordered_multimap<शून्य*, पूर्णांक>         d_map; // tensor data to index

    अचिन्हित पूर्णांक n_bufs;     // num buffers in the batch
    अचिन्हित पूर्णांक n_tens;     // num tensors ...
    अचिन्हित पूर्णांक n_ops;      // num ops ...
    माप_प्रकार       b_vmem;     // sum of non-extended buffer sizes

    अचिन्हित पूर्णांक n_bufs_max;
    अचिन्हित पूर्णांक n_tens_max;
    अचिन्हित पूर्णांक n_ops_max;
    माप_प्रकार       b_vmem_max;

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

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, माप_प्रकार batch_size, माप_प्रकार max_vmem) {
        हे->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<माप_प्रकार>(n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS, HTP_OP_MAX_TENSORS);

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

    तार्किक empty() स्थिरांक { परत n_ops == 0; }

    // add buffer and return its index
    पूर्णांक add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        स्वतः it = b_map.find(sbuf->fd());
        जर (it != b_map.end()) { परत it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        पूर्णांक bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        जर (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (शून्य*) sbuf->base(), (माप_प्रकार) b.size, b_vmem);

        परत bi;
    }

    तार्किक same_shape(स्थिरांक htp_tensor * h, स्थिरांक ggml_tensor * t) स्थिरांक {
        स्वतः extra = (ggml_hexagon_tensor_extra *) t->extra;

        स्थिरांक तार्किक is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        स्थिरांक int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        स्थिरांक int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? आकार(ggml_fp16_t) : t->nb[0];
        गणना ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        परत (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    पूर्णांक add_tensor(स्थिरांक ggml_tensor * t) {
        स्वतः extra = (ggml_hexagon_tensor_extra *) t->extra;
        स्वतः sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        स्वतः range = d_map.equal_range(t->data);
        साठी (स्वतः it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            जर (same_shape(h, t)) { परत it->second; }
        }

        // Lookup by tensor ptr
        स्वतः it = t_map.find(t);
        जर (it != t_map.end()) { परत it->second; }

        // Add new tensor to the batch
        पूर्णांक ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base();
        माप_प्रकार   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        स्थिरांक तार्किक is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        जर (is_tiled) {
            जर (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = आकार(ggml_fp16_t);
            } अन्यथा {
                h.nb[0] = t->nb[0];
            }
            स्थिरांक int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            स्थिरांक int64_t ne1_padded = hex_round_up(t->ne[1], 32);
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
        जर ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        जर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        जर ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (शून्य*) t->data, (माप_प्रकार) t_offset, t_size, h.flags,
                (माप_प्रकार) h.ne[0], (माप_प्रकार) h.ne[1], (माप_प्रकार) h.ne[2], (माप_प्रकार) h.ne[3]);

        परत ti;
    }

    तार्किक fit_op(स्थिरांक htp_opnode & गाठ) स्थिरांक {
        जर (n_ops >= n_ops_max) परत असत्य;

        // check how much extras we will need
        माप_प्रकार extra_bufs = 0;
        माप_प्रकार extra_vmem = 0;
        माप_प्रकार extra_tens = 0;

        पूर्णांक seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णांक n_seen_bufs = 0;

        स्वतः fit_tensor = [&](स्थिरांक ggml_tensor *t) {
            जर (!t) परत;
            जर (!t_map.count(t)) {
                extra_tens++;

                स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                पूर्णांक fd = sbuf->fd();
                जर (!b_map.count(fd)) {
                    साठी (पूर्णांक i = 0; i < n_seen_bufs; i++) {
                        जर (seen_bufs[i] == fd) परत;
                    }
                    जर (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    जर (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        साठी (स्थिरांक स्वतः * src : गाठ.get_inputs()) {
            fit_tensor(src);
        }
        साठी (स्थिरांक स्वतः * output : गाठ.get_outputs()) {
            fit_tensor(output);
        }

        जर ((extra_bufs + n_bufs) > n_bufs_max) परत असत्य;
        जर ((extra_tens + n_tens) > n_tens_max) परत असत्य;
        जर ((extra_vmem + b_vmem) > b_vmem_max) परत असत्य;

        परत सत्य;
    }

    // assumes that fit_op() was called first and returned true
    शून्य add_op(स्थिरांक htp_opnode & गाठ) {
        // Add new op

        अचिन्हित पूर्णांक n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = गाठ;

        htp_op_desc &o = h_ops[n];
        स_नकल(o.params,        गाठ.गाठ->op_params, आकार(गाठ.गाठ->op_params));
        स_नकल(o.kernel_params, गाठ.kernel_params,   आकार(o.kernel_params));
        o.opcode = गाठ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        स्वतः inputs = गाठ.get_inputs();
        साठी (अचिन्हित पूर्णांक i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        स्वतः outputs = गाठ.get_outputs();
        साठी (अचिन्हित पूर्णांक i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    शून्य sort_buffers() {
        जर (n_bufs <= 1) परत;

        std::vector<पूर्णांक> order(n_bufs);
        साठी (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) { order[i] = (पूर्णांक) i; }

        std::stable_sort(order.begin(), order.end(), [&](पूर्णांक a, पूर्णांक b) {
            परत h_bufs[a].size > h_bufs[b].size;
        });

        तार्किक already_sorted = सत्य;
        साठी (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) {
            जर (order[i] != (पूर्णांक) i) {
                already_sorted = असत्य;
                थांबा;
            }
        }
        जर (already_sorted) परत;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        साठी (अचिन्हित पूर्णांक new_bi = 0; new_bi < n_bufs; new_bi++) {
            पूर्णांक old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        साठी (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        साठी (अचिन्हित पूर्णांक i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    शून्य update_mdev_group(uint32_t mdev_idx) {
        जर (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    तार्किक try_fuse_common(std::initializer_list<स्थिरांक ggml_tensor *> tensors) स्थिरांक {
        माप_प्रकार extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        पूर्णांक seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णांक n_seen_bufs = 0;

        साठी (स्थिरांक स्वतः * t : tensors) {
            जर (!t || t_map.count(t)) {
                सुरूठेवा;
            }
            extra_tens++;
            स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            पूर्णांक fd = sbuf->fd();
            जर (!b_map.count(fd)) {
                तार्किक found = असत्य;
                साठी (पूर्णांक i = 0; i < n_seen_bufs; i++) {
                    जर (seen_bufs[i] == fd) {
                        found = सत्य;
                        थांबा;
                    }
                }
                जर (!found) {
                    जर (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    जर (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        जर ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            परत असत्य;
        }

        परत सत्य;
    }

    तार्किक try_fuse_common(स्थिरांक ggml_tensor * t1, स्थिरांक ggml_tensor * t2) स्थिरांक {
        परत try_fuse_common({t1, t2});
    }

    तार्किक try_fuse_allreduce_add(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0 || opt_ar_select != 2) परत असत्य;
        जर (गाठ.opcode != HTP_OP_ADD) परत असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        जर (last_गाठ.opcode != HTP_OP_ALLREDUCE) परत असत्य;

        स्वतः * ar_kparams = (संरचना htp_allreduce_kernel_params *) last_गाठ.kernel_params;
        स्थिरांक uint32_t rank    = (uint32_t) ar_kparams->rank;
        स्थिरांक uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        स्थिरांक ggml_tensor * ar_local = last_गाठ.inputs[rank];
        स्थिरांक ggml_tensor * add_src0 = गाठ.src0();
        स्थिरांक ggml_tensor * add_src1 = गाठ.src1();
        स्थिरांक ggml_tensor * add_dst  = गाठ.dst();

        जर (!ggml_hexagon_tensor_is_fuseable(ar_local)) परत असत्य;

        स्थिरांक ggml_tensor * res_tensor;
        जर (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } अन्यथा जर (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } अन्यथा {
            परत असत्य;
        }

        जर (ar_local->type != res_tensor->type) परत असत्य;

        स्थिरांक तार्किक is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        स्थिरांक तार्किक is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        जर (!is_same_shape && !is_row_bcast) परत असत्य;

        जर (is_same_shape) {
            जर (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                परत असत्य;
            }
            जर (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                परत असत्य;
            }
        }
        जर (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            परत असत्य;
        }

        // scatter is only valid for in-place add within max outputs
        स्थिरांक तार्किक is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        संरचना htp_allreduce_kernel_params new_kparams;
        जर (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, सत्य, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            परत असत्य;
        }

        स्थिरांक तार्किक scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        साठी (uint32_t r = 0; r < n_ranks; r++) {
            स्थिरांक ggml_tensor * ar_src = last_गाठ.inputs[r];
            जर (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                सुरूठेवा;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            जर (scatter_ok && r == rank) {
                सुरूठेवा;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            परत असत्य;
        }

        जर (!try_fuse_common(res_tensor, add_dst)) {
            परत असत्य;
        }

        last_गाठ.opcode = HTP_OP_ALLREDUCE_ADD;
        last_गाठ.name   = "ALLREDUCE+ADD";
        last_गाठ.inputs.push_back(res_tensor);
        last_गाठ.outputs.clear();
        last_गाठ.outputs.push_back(add_dst);
        last_गाठ.fused.push_back(गाठ.गाठ);
        स_नकल(last_गाठ.kernel_params, &new_kparams, आकार(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        स_नकल(o.kernel_params, &new_kparams, आकार(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        जर (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            साठी (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            साठी (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } अन्यथा {
            o.dst[0] = add_tensor(add_dst);
            साठी (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (पूर्णांक) new_kparams.mode, (पूर्णांक) new_kparams.n_dsts);
        परत सत्य;
    }

    तार्किक try_fuse_rms_norm_mul(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0) परत असत्य;
        जर (गाठ.opcode != HTP_OP_MUL) परत असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        जर (last_गाठ.opcode != HTP_OP_RMS_NORM) परत असत्य;

        स्थिरांक ggml_tensor * mul_src0 = गाठ.src0();
        स्थिरांक ggml_tensor * mul_src1 = गाठ.src1();
        स्थिरांक ggml_tensor * rms_out  = last_गाठ.dst();

        जर (!ggml_hexagon_tensor_is_fuseable(rms_out)) परत असत्य;

        स्थिरांक ggml_tensor * weight;
        जर (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } अन्यथा जर (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } अन्यथा {
            परत असत्य;
        }

        स्थिरांक ggml_tensor * src0 = last_गाठ.src0();

        जर (src0->ne[0] != weight->ne[0] || src0->ne[0] != गाठ.dst()->ne[0]) {
            परत असत्य;
        }

        स्थिरांक तार्किक is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        स्थिरांक तार्किक is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        जर (!is_row_bcast && !is_same_shape) परत असत्य;

        जर (!ggml_are_same_shape(src0, गाठ.dst())) {
            परत असत्य;
        }
        जर (ggml_is_contiguous(src0) != ggml_is_contiguous(गाठ.dst())) {
            परत असत्य;
        }

        संरचना htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, गाठ.dst(), &new_kparams
        );

        जर ((माप_प्रकार) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            परत असत्य;
        }

        जर (!try_fuse_common(weight, गाठ.dst())) {
            परत असत्य;
        }

        last_गाठ.opcode = HTP_OP_RMS_NORM_MUL;
        last_गाठ.name   = "RMS_NORM+MUL";
        last_गाठ.inputs.clear();
        last_गाठ.inputs.push_back(src0);
        last_गाठ.inputs.push_back(weight);
        last_गाठ.outputs.clear();
        last_गाठ.outputs.push_back(गाठ.dst());
        last_गाठ.fused.push_back(गाठ.गाठ);
        स_नकल(last_गाठ.kernel_params, &new_kparams, आकार(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        स_नकल(o.kernel_params, &new_kparams, आकार(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        साठी (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाठ.dst());
        साठी (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        परत सत्य;
    }

    तार्किक try_fuse_mul_mat_add(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0) परत असत्य;
        जर (गाठ.opcode != HTP_OP_ADD) परत असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        जर (last_गाठ.opcode != HTP_OP_MUL_MAT) परत असत्य;

        स्थिरांक ggml_tensor * add_src0 = गाठ.src0();
        स्थिरांक ggml_tensor * add_src1 = गाठ.src1();
        स्थिरांक ggml_tensor * mm_out   = last_गाठ.dst();

        जर (!ggml_hexagon_tensor_is_fuseable(mm_out)) परत असत्य;

        स्थिरांक ggml_tensor * src2;
        जर (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } अन्यथा जर (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } अन्यथा {
            परत असत्य;
        }

        स्थिरांक ggml_tensor * src0 = last_गाठ.src0();
        स्थिरांक ggml_tensor * src1 = last_गाठ.src1();

        जर (src2->type != GGML_TYPE_F32) परत असत्य;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        जर (!ggml_are_same_shape(mm_out, गाठ.dst())) परत असत्य;

        स्थिरांक संरचना htp_mm_kernel_params * orig_kparams = (स्थिरांक संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
        संरचना htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, गाठ.dst(), &kparams);
        जर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            परत असत्य;
        }

        जर (orig_kparams->collapse != kparams.collapse) {
            परत असत्य;
        }

        स्थिरांक पूर्णांक src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        स्थिरांक तार्किक can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        जर (!can_fuse) परत असत्य;

        जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            परत असत्य;
        }

        जर (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            जर (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                परत असत्य;
            }
        }

        जर (!try_fuse_common(src2, गाठ.dst())) {
            परत असत्य;
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
        स_नकल(last_गाठ.kernel_params, &kparams, आकार(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        स_नकल(o.kernel_params, &kparams, आकार(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        साठी (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाठ.dst());
        साठी (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        परत सत्य;
    }

    तार्किक try_fuse_mul_mat_nx(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0 || गाठ.opcode != HTP_OP_MUL_MAT) परत असत्य;
        जर (!is_mergeable_mul_mat(गाठ.गाठ)) परत असत्य;

        स्थिरांक ggml_tensor * w_in = गाठ.src0();
        स्थिरांक ggml_tensor * x_in = गाठ.src1();
        स्थिरांक ggml_tensor * d_in = गाठ.dst();

        htp_opnode & last_गाठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        जर (last_गाठ.opcode == HTP_OP_MUL_MAT_NX) {
            स्थिरांक uint32_t curr_n = (uint32_t) last_गाठ.outputs.size();
            जर (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                परत असत्य;
            }

            स्थिरांक ggml_tensor * w0 = last_गाठ.inputs[0];
            स्थिरांक ggml_tensor * x  = last_गाठ.inputs[curr_n];

            जर (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                परत असत्य;
            }
            जर (!last_गाठ.fused.empty() && (mm_is_hmx_eligible(last_गाठ.fused[0]) != mm_is_hmx_eligible(गाठ.गाठ))) {
                परत असत्य;
            }

            स्थिरांक संरचना htp_mm_kernel_params * orig_kparams = (स्थिरांक संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
            स्थिरांक तार्किक collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            जर (orig_kparams->collapse && !collapse) {
                परत असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            जर (collapse) {
                स्थिरांक ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            जर (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                परत असत्य;
            }
            जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                परत असत्य;
            }

            जर (!try_fuse_common(w_in, d_in)) {
                परत असत्य;
            }

            last_गाठ.inputs[curr_n] = w_in;
            last_गाठ.inputs.push_back(x);
            last_गाठ.outputs.push_back(d_in);
            last_गाठ.fused.push_back(गाठ.गाठ);
            स_नकल(last_गाठ.kernel_params, &kparams, आकार(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            स_नकल(o.kernel_params, &kparams, आकार(kparams));

            साठी (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_गाठ.inputs[s]);
            }
            साठी (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            साठी (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाठ.outputs[d]);
            }
            साठी (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            परत सत्य;
        }

        // Case 2: last_node is single MUL_MAT
        जर (last_गाठ.opcode == HTP_OP_MUL_MAT) {
            जर (!is_mergeable_mul_mat_pair(last_गाठ.गाठ, गाठ.गाठ)) {
                परत असत्य;
            }

            स्थिरांक ggml_tensor * w0 = last_गाठ.src0();
            स्थिरांक ggml_tensor * x  = last_गाठ.src1();
            स्थिरांक ggml_tensor * w1 = गाठ.src0();
            स्थिरांक ggml_tensor * dst_0 = last_गाठ.dst();
            स्थिरांक ggml_tensor * dst_1 = गाठ.dst();

            स्थिरांक संरचना htp_mm_kernel_params * orig_kparams = (स्थिरांक संरचना htp_mm_kernel_params *) last_गाठ.kernel_params;
            स्थिरांक तार्किक collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            जर (orig_kparams->collapse && !collapse) {
                परत असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            जर (collapse) {
                स्थिरांक ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            जर (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                परत असत्य;
            }
            जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                परत असत्य;
            }

            जर (!try_fuse_common(w1, गाठ.dst())) {
                परत असत्य;
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
            स_नकल(last_गाठ.kernel_params, &kparams, आकार(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            स_नकल(o.kernel_params, &kparams, आकार(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            साठी (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            साठी (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            परत सत्य;
        }

        परत असत्य;
    }

    तार्किक try_fuse_mul_mat_id_nx(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0 || गाठ.opcode != HTP_OP_MUL_MAT_ID) परत असत्य;
        जर (!is_mergeable_mul_mat_id(गाठ.गाठ)) परत असत्य;

        स्थिरांक ggml_tensor * w_in   = गाठ.src0();
        स्थिरांक ggml_tensor * x_in   = गाठ.src1();
        स्थिरांक ggml_tensor * ids_in = गाठ.गाठ->src[2];
        स्थिरांक ggml_tensor * d_in   = गाठ.dst();

        htp_opnode & last_गाठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        जर (last_गाठ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            स्थिरांक uint32_t curr_n = (uint32_t) last_गाठ.outputs.size();
            जर (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                परत असत्य;
            }

            स्थिरांक ggml_tensor * w0  = last_गाठ.inputs[0];
            स्थिरांक ggml_tensor * x   = last_गाठ.inputs[curr_n];
            स्थिरांक ggml_tensor * ids = last_गाठ.inputs[curr_n + 1];

            जर (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                परत असत्य;
            }
            जर (!last_गाठ.fused.empty() && (mm_is_hmx_eligible(last_गाठ.fused[0]) != mm_is_hmx_eligible(गाठ.गाठ))) {
                परत असत्य;
            }

            संरचना htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            जर (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                परत असत्य;
            }
            जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                परत असत्य;
            }

            जर (!try_fuse_common(w_in, d_in)) {
                परत असत्य;
            }

            last_गाठ.inputs[curr_n] = w_in;
            last_गाठ.inputs[curr_n + 1] = x;
            last_गाठ.inputs.push_back(ids);
            last_गाठ.outputs.push_back(d_in);
            last_गाठ.fused.push_back(गाठ.गाठ);
            स_नकल(last_गाठ.kernel_params, &kparams, आकार(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            स_नकल(o.kernel_params, &kparams, आकार(kparams));

            साठी (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_गाठ.inputs[s]);
            }
            साठी (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            साठी (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाठ.outputs[d]);
            }
            साठी (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            परत सत्य;
        }

        // Case 2: last_node is single MUL_MAT_ID
        जर (last_गाठ.opcode == HTP_OP_MUL_MAT_ID) {
            जर (!is_mergeable_mul_mat_id_pair(last_गाठ.गाठ, गाठ.गाठ)) {
                परत असत्य;
            }

            स्थिरांक ggml_tensor * w0  = last_गाठ.src0();
            स्थिरांक ggml_tensor * x   = last_गाठ.src1();
            स्थिरांक ggml_tensor * ids = last_गाठ.गाठ->src[2];
            स्थिरांक ggml_tensor * w1  = गाठ.src0();

            संरचना htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, गाठ.dst(), 2, &kparams);
            जर (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                परत असत्य;
            }
            जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                परत असत्य;
            }

            जर (!try_fuse_common(w1, गाठ.dst())) {
                परत असत्य;
            }

            स्थिरांक ggml_tensor * dst_0 = last_गाठ.dst();
            स्थिरांक ggml_tensor * dst_1 = गाठ.dst();

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
            स_नकल(last_गाठ.kernel_params, &kparams, आकार(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            स_नकल(o.kernel_params, &kparams, आकार(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            साठी (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            साठी (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            परत सत्य;
        }

        परत असत्य;
    }

    तार्किक try_fuse_gdn_cpy(स्थिरांक htp_opnode & गाठ) {
        जर (n_ops == 0 || गाठ.opcode != HTP_OP_CPY) परत असत्य;

        htp_opnode & last_गाठ = ops[n_ops - 1];
        जर (last_गाठ.opcode != HTP_OP_GATED_DELTA_NET) परत असत्य;
        जर (last_गाठ.outputs.size() != 1) परत असत्य;

        स्थिरांक ggml_tensor * gdn_out  = last_गाठ.dst();
        स्थिरांक ggml_tensor * cpy_गाठ = गाठ.गाठ;
        स्थिरांक ggml_tensor * cpy_src  = गाठ.src0();
        स्थिरांक ggml_tensor * cpy_dst  = गाठ.dst();

        जर (!cpy_src || !cpy_dst || !cpy_dst->data) परत असत्य;
        जर (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) परत असत्य;
        जर ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_गाठ->flags & GGML_TENSOR_FLAG_OUTPUT)) परत असत्य;

        स्थिरांक ggml_tensor * v = last_गाठ.गाठ->src[2];
        जर (!v) परत असत्य;

        स्थिरांक int64_t S_v      = v->ne[0];
        स्थिरांक int64_t H        = v->ne[1];
        स्थिरांक int64_t n_tokens = v->ne[2];
        स्थिरांक int64_t n_seqs   = v->ne[3];
        स्थिरांक int64_t K        = ggml_get_op_params_i32(last_गाठ.गाठ, 0);
        स्थिरांक माप_प्रकार  tail_off = (माप_प्रकार) S_v * H * n_tokens * n_seqs * आकार(दशांश);

        स्थिरांक int64_t D         = S_v * S_v * H;
        स्थिरांक int64_t n_written = std::min<int64_t>(n_tokens, K);

        जर (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            परत असत्य;
        }

        जर (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != आकार(दशांश)) {
            परत असत्य;
        }
        जर (n_seqs > 1 && cpy_dst->nb[1] != (माप_प्रकार) D * आकार(दशांश)) {
            परत असत्य;
        }
        जर (n_written > 1) {
            जर (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (माप_प्रकार) D * n_seqs * आकार(दशांश)) {
                परत असत्य;
            }
        }

        जर (!try_fuse_common({cpy_dst})) {
            परत असत्य;
        }

        last_गाठ.name += "+CPY";
        last_गाठ.outputs.push_back(cpy_dst);
        last_गाठ.fused.push_back(गाठ.गाठ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        साठी (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        परत सत्य;
    }

    तार्किक try_fuse(स्थिरांक htp_opnode & गाठ) {
        जर (!opt_opfusion) परत असत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(गाठ)) परत सत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(गाठ))  परत सत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(गाठ))   परत सत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(गाठ))    परत सत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(गाठ)) परत सत्य;
        जर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(गाठ))       परत सत्य;
        परत असत्य;
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
    माप_प्रकार                      shm_blk_size;
    माप_प्रकार                      depth;

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, माप_प्रकार batch_size, माप_प्रकार depth) : depth(depth) {
        माप_प्रकार n_bufs    = HTP_OP_MAX_BUFS;
        माप_प्रकार n_ops     = batch_size;
        माप_प्रकार n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        माप_प्रकार tr_size = 0;
        जर (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * आकार(htp_trace_desc);
        }

        shm_blk_size = आकार(htp_buf_desc)  * n_bufs    +
                       आकार(htp_tensor)    * n_tensors +
                       आकार(htp_op_desc)   * n_ops     +
                       आकार(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = नवीन ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, सत्य /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        जर (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    माप_प्रकार shm_size() स्थिरांक { परत shm_buf ? shm_buf->size() : 0; }

    // push new batch
    तार्किक ढकला(htp_opbatch_req& req, dspqueue_buffer& dbuf, स्थिरांक ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(आकार(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(आकार(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(आकार(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(आकार(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(आकार(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(आकार(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        जर (seq - shm_buf->sess->batch_rsp_seq > depth) { परत असत्य; }

        स्थिरांक uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        स्थिरांक माप_प्रकार b_size = आकार(htp_buf_desc)  * req.n_bufs;
        स्थिरांक माप_प्रकार t_size = आकार(htp_tensor)    * req.n_tensors;
        स्थिरांक माप_प्रकार o_size = आकार(htp_op_desc)   * req.n_ops;
        स्थिरांक माप_प्रकार p_size = आकार(htp_prof_desc) * req.n_ops;

        माप_प्रकार tr_size = 0;
        जर (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * आकार(htp_trace_desc);
        } अन्यथा {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((माप_प्रकार) slot * shm_blk_size);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        स_नकल(b_ptr, (शून्य *) op_batch->h_bufs.data(), b_size);
        स_नकल(t_ptr, (शून्य *) op_batch->h_tens.data(), t_size);
        स_नकल(o_ptr, (शून्य *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (अचिन्हित दीर्घ दीर्घ) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (माप_प्रकार) dbuf.size);

        जर (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            साठी (अचिन्हित पूर्णांक i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (शून्य *) b[i].base, (माप_प्रकार) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            साठी (अचिन्हित पूर्णांक i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (अचिन्हित दीर्घ दीर्घ) t[i].data, t[i].size,
                            (माप_प्रकार) t[i].ne[0], (माप_प्रकार) t[i].ne[1], (माप_प्रकार) t[i].ne[2], (माप_प्रकार) t[i].ne[3]);
            }
        }

        परत सत्य;
    }

    शून्य काढा(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        स्थिरांक uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        स्थिरांक माप_प्रकार b_size = आकार(htp_buf_desc)  * rsp.n_bufs;
        स्थिरांक माप_प्रकार t_size = आकार(htp_tensor)    * rsp.n_tensors;
        स्थिरांक माप_प्रकार o_size = आकार(htp_op_desc)   * rsp.n_ops;
        स्थिरांक माप_प्रकार p_size = आकार(htp_prof_desc) * rsp.n_ops;

        माप_प्रकार tr_size = 0;
        uint32_t n_traces = 0;
        जर (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * आकार(htp_trace_desc);
        }

        स्थिरांक माप_प्रकार m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (अचिन्हित दीर्घ दीर्घ) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (माप_प्रकार) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        जर (rsp.n_ops > 0) {
            स्वतः & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            स्थिरांक htp_prof_desc * pd = (स्थिरांक htp_prof_desc *) p_ptr;
            स्थिरांक htp_trace_desc * trace_events = nullptr;
            जर (opt_profile == 3) {
                trace_events = (स्थिरांक htp_trace_desc *) (p_ptr + p_size);
            }

            जर (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            साठी (uint32_t i = 0; i < rsp.n_ops; i++) {
                जर (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            जर (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

शून्य ggml_hexagon_session::flush_peers() {
    स्वतः vpeers = std::move(virt_peers);
    virt_peers.clear();
    साठी (स्वतः * peer : vpeers) {
        peer->flush_sync();
    }

    स्वतः ppeers = std::move(phys_peers);
    phys_peers.clear();
    साठी (स्वतः * peer : ppeers) {
        peer->flush_async();
    }

    साठी (स्वतः & sub : हे->mdev.sessions) {
        sub->flush_peers();
    }
}

शून्य ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

शून्य ggml_hexagon_session::flush_pending(तार्किक all) {
    साठी (स्वतः & sub : हे->mdev.sessions) {
        sub->flush_pending(all);
        जर (sub->last_error > HTP_STATUS_OK) {
            हे->last_error = sub->last_error.load();
        }
    }

    पर्यंत (हे->batch_rsp_seq < हे->batch_req_seq) {
        संरचना htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        संरचना dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        स्थिरांक uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        पूर्णांक err = dspqueue_read(हे->queue, &flags, 1, &n_dbufs, &dbuf, आकार(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        जर (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            सुरूठेवा;
        }

        जर (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (अचिन्हित) err);
        }

        // Basic sanity checks
        जर (rsp_size != आकार(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", हे->c_name(), rsp_size, n_dbufs);
        }

        जर (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", हे->c_name(), status_to_str(rsp.status));
            हे->last_error = rsp.status;
            साठी (स्वतः & sub : हे->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->काढा(rsp, dbuf);

        GGML_ASSERT(rsp.seq == हे->batch_rsp_seq + 1);
        हे->batch_rsp_seq = rsp.seq;

        जर (!all) थांबा;
    }
}

शून्य ggml_hexagon_session::flush_sync(तार्किक all) {
    flush_async();
    flush_pending(all);
}

शून्य ggml_hexagon_session::start_batch() {
    जर (हे->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

शून्य ggml_hexagon_session::flush_batch(माप_प्रकार min_ops) {
    जर (op_batch->n_ops < min_ops) { परत; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    स्थिरांक uint64_t seq = ++हे->batch_req_seq;

    op_batch->update_mdev_group(हे->mdev.idx);

    जर (!op_queue->ढकला(req, dbuf, op_batch, seq)) {
        flush_pending(असत्य);
        op_queue->ढकला(req, dbuf, op_batch, seq);
    }

    साठी (स्वतः & sub : हे->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        जर (!sub->op_queue->ढकला(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(असत्य);
            sub->op_queue->ढकला(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        पूर्णांक err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, आकार(sub_req), (स्थिरांक uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        जर (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (अचिन्हित) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", हे->c_name(), dbuf.ptr, dbuf.size);

    पूर्णांक err = dspqueue_write(हे->queue, 0, 1, &dbuf, आकार(req), (स्थिरांक uint8_t*) &req, DSPQUEUE_TIMEOUT);
    जर (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", हे->c_name(), (अचिन्हित) err);
    }

    op_batch->reset();
}

शून्य ggml_hexagon_session::enqueue_op(स्थिरांक htp_opnode & गाठ) {
    स्वतः clone_tensor_buffer = [हे](स्थिरांक ggml_tensor * t) {
        स्वतः sbuf = हे->mmap_tensor(t);
        जर (!sbuf) परत;
        जर (sbuf->sess != हे) {
            हे->clone_buffer(sbuf);
        }
        साठी (स्वतः & sub : हे->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    साठी (स्वतः t : गाठ.get_inputs()) {
        clone_tensor_buffer(t);
    }
    साठी (स्वतः t : गाठ.get_outputs()) {
        clone_tensor_buffer(t);
    }

    जर (opt_opfusion && op_batch->try_fuse(गाठ)) {
        परत;
    }

    जर (!op_batch->fit_op(गाठ)) {
        flush_async();
    }

    जर (op_batch->empty()) {
        start_batch();
    }

    जर (!op_batch->fit_op(गाठ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(गाठ);
}

शून्य ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_गाठ(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = हे->mdev_fence_slot;

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &हे->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (शून्य *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) हे->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) हे->mdev.idx;

    ggml_tensor * गाठ = group_गाठ.add_dummy(dummy_t);
    गाठ->src[0] = गाठ;
    group_गाठ.init(गाठ);
    group_गाठ.outputs.clear();
    group_गाठ.name = "MDEV_GROUP";

    साठी (स्वतः & sub : हे->mdev.sessions) {
        sub->clone_buffer(हे->fence_buf);
    }

    op_batch->add_op(group_गाठ);
}

शून्य ggml_hexagon_session::enqueue_cpy(स्थिरांक ggml_tensor * src, ggml_tensor * dst, स्थिरांक ggml_tensor * sync_tensor, uint32_t fence_seq) {
    स्थिरांक तार्किक with_fence = sync_tensor != nullptr;
    htp_opnode cpy_गाठ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* गाठ = cpy_गाठ.add_dummy(*dst);
    गाठ->op     = GGML_OP_CPY;
    गाठ->src[0] = const_cast<ggml_tensor *>(src);
    गाठ->src[1] = with_fence ? cpy_गाठ.add_dummy(*sync_tensor) : nullptr;
    जर (with_fence) {
        गाठ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_गाठ.init(गाठ);
    जर (with_fence) {
        cpy_गाठ.name = "CPY+FENCE";
    }
    स्थिरांक तार्किक ok = ggml_hexagon_precompute_cpy_params(हे, गाठ, (संरचना htp_copy_kernel_params *) cpy_गाठ.kernel_params);
    स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_copy_kernel_params *) cpy_गाठ.kernel_params;
    जर (ok && !with_fence && kparams->total_elems == 0) {
        परत;
    }
    हे->enqueue_op(cpy_गाठ);
}

शून्य ggml_hexagon_session::enqueue_fence(स्थिरांक ggml_tensor * sync_tensor, uint32_t fence_seq, तार्किक रुको) {
    htp_opnode sync_गाठ(HTP_OP_FENCE);

    ggml_tensor* गाठ = sync_गाठ.add_dummy(*sync_tensor);
    गाठ->op           = GGML_OP_NONE;
    गाठ->src[0]       = गाठ;
    गाठ->op_params[0] = (int32_t) fence_seq;
    गाठ->op_params[1] = रुको ? 0 : 1;

    sync_गाठ.init(गाठ);
    sync_गाठ.name = रुको ? "FENCE_WAIT" : "FENCE_SIGNAL";
    हे->enqueue_op(sync_गाठ);
}

स्थिर तार्किक ggml_hexagon_precompute_allreduce_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    तार्किक has_add,
    तार्किक is_row_bcast,
    तार्किक is_shard_ok,
    संरचना htp_allreduce_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    स्थिरांक uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    स्थिरांक uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? आकार(ggml_fp16_t) : आकार(दशांश);
    स्थिरांक तार्किक is_contiguous = ggml_is_contiguous(dst);

    स्थिरांक uint32_t ne0 = (uint32_t) dst->ne[0];
    स्थिरांक uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    स्थिरांक तार्किक use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    जर (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        जर (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            स्थिरांक uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            स्थिरांक uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            स्थिरांक uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } अन्यथा जर (use_1d) {
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
        जर (use_1d) {
            स्थिरांक uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            स्थिरांक uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            स्थिरांक uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            स्थिरांक uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } अन्यथा {
            स्थिरांक uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            स्थिरांक uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            स्थिरांक uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            स्थिरांक uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    जर (use_1d) {
        स्थिरांक uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        स्थिरांक माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_buffer_मोजणी(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        जर (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        पर्यंत ((माप_प्रकार) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            स्थिरांक माप_प्रकार max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((माप_प्रकार) (max_bytes_per_buf / elem_size), 128);
            जर (block_elems < 128) थांबा;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        जर (sess->vtcm_size < (माप_प्रकार) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            परत असत्य;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        परत सत्य;
    } अन्यथा {
        स्थिरांक uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        स्थिरांक माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_buffer_मोजणी(n_ranks, n_threads, has_add, is_row_bcast);

        स्थिरांक uint32_t row_bytes = ne0 * elem_size;
        स्थिरांक uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        स्थिरांक uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        पर्यंत ((माप_प्रकार) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            स्थिरांक माप_प्रकार max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            जर (max_rows_per_buf == 0) थांबा;
        }

        जर (sess->vtcm_size < (माप_प्रकार) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            परत असत्य;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        परत सत्य;
    }
}

शून्य ggml_hexagon_session::enqueue_allreduce(
    स्थिरांक ggml_tensor * dst,
    स्थिरांक std::vector<स्थिरांक ggml_tensor *> & src_tensors,
    स्थिरांक std::vector<स्थिरांक ggml_tensor *> & sync_tensors,
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
    साठी (माप_प्रकार i = 0; i < src_tensors.size(); i++) {
        ar_गाठ.inputs.push_back(src_tensors[i]);
    }
    साठी (माप_प्रकार i = 0; i < sync_tensors.size(); i++) {
        ar_गाठ.inputs.push_back(ar_गाठ.add_dummy(*sync_tensors[i]));
    }

    ar_गाठ.outputs.clear();
    साठी (माप_प्रकार i = 0; i < src_tensors.size(); i++) {
        ar_गाठ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        हे, dst, rank, n_ranks, असत्य, असत्य, /*is_shard_ok=*/ असत्य,
        (संरचना htp_allreduce_kernel_params *) ar_गाठ.kernel_params);

    ar_गाठ.name = "ALLREDUCE";
    हे->enqueue_op(ar_गाठ);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(स्थिरांक ggml_tensor * t) {
    जर (!t) परत nullptr;

    स्वतः sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    जर (!sbuf->mapped) {
        स्थिरांक तार्किक is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        स्थिरांक तार्किक extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    परत sbuf;
}

तार्किक ggml_hexagon_session::clone_buffer(स्थिरांक ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    जर (sbuf->sess == हे) परत सत्य;

    स्वतः mem = sbuf->mem;
    पूर्णांक   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    जर (हे->cloned_buffers.find(fd) != हे->cloned_buffers.end()) परत सत्य;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", हे->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    स्वतः clone = std::make_unique<ggml_hexagon_shared_buffer>(हे, *sbuf);
    प्रयत्न {
        clone->mmap(sbuf->extended);
    } पकडा (स्थिरांक std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", हे->c_name(), exc.what());
        परत असत्य;
    }

    हे->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(हे);
    परत सत्य;
}

शून्य ggml_hexagon_session::release_buffer(स्थिरांक ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    स्वतः mem = sbuf->mem;
    पूर्णांक   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    स्वतः it = हे->cloned_buffers.find(fd);
    जर (it != हे->cloned_buffers.end()) {
        स्वतः clone = std::move(it->second);
        हे->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(हे);
}

शून्य ggml_hexagon_session::unclone_buffer(स्थिरांक ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    स्वतः mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    साठी (स्वतः * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

स्थिर माप_प्रकार ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    स्थिरांक माप_प्रकार MiB = 1024 * 1024;
    स्थिरांक माप_प्रकार GiB = MiB  * 1024;

    माप_प्रकार vmem = 0;
    माप_प्रकार step = 256u * MiB;

    प्रयत्न {
        sbufs.push_back(नवीन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;
        sbufs.push_back(नवीन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;
        sbufs.push_back(नवीन ggml_hexagon_shared_buffer(sess, GiB, सत्य)); vmem += GiB;

        पर्यंत (1) {
            sbufs.push_back(नवीन ggml_hexagon_shared_buffer(sess, step, सत्य));
            vmem += step;
        }
    } पकडा (...) { }

    साठी (स्वतः b : sbufs) { delete b; }

    परत vmem - step; // backoff to account for overhead from internal mappings
}

शून्य ggml_hexagon_session::allocate(स्थिरांक ggml_hexagon_device_config & config) noexcept(असत्य) {
    पूर्णांक phys_idx = config.physical_idx;
    पूर्णांक virt_idx = config.virtual_idx;

    हे->valid_session = असत्य;
    हे->valid_handle  = असत्य;
    हे->valid_queue   = असत्य;
    हे->valid_iface   = असत्य;

    हे->name          = config.name;
    हे->phys_idx      = phys_idx;
    हे->virt_idx      = virt_idx;
    हे->domain_id     = config.domain_id;
    हे->session_id    = 0;
    हे->batch_req_seq = 0;
    हे->batch_rsp_seq = 0;
    हे->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", हे->name.c_str(), हे->domain_id, phys_idx, virt_idx);

    जर (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        फेका std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    स्थिरांक std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    जर (virt_idx > 0) {
        संरचना remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<अक्षर *>(dom_name.c_str());
        n.session_name     = const_cast<अक्षर *>(हे->name.c_str());
        n.session_name_len = हे->name.size();
        n.session_id       = virt_idx;

        पूर्णांक err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (शून्य *) &n, आकार(n));
        जर (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           हे->c_name(), phys_idx, virt_idx, err);
            फेका std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        हे->session_id    = n.session_id;
        हे->domain_id     = n.effective_domain_id;
        हे->valid_session = सत्य;
    } अन्यथा {
        संरचना remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<अक्षर *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        पूर्णांक err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (शून्य *) &eff, आकार(eff));
        जर (err == AEE_SUCCESS) {
            हे->domain_id = eff.effective_domain_id;
        } अन्यथा {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           हे->name.c_str(), err, हे->domain_id);
        }
    }

    // Enable unsigned modules
    {
        संरचना remote_rpc_control_unsigned_module u;
        u.domain = हे->domain_id;
        u.enable = 1;
        पूर्णांक err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (शून्य *) &u, आकार(u));
        जर (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", हे->c_name(), err);
            फेका std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    अक्षर session_uri[256];
    {
        अक्षर htp_uri[256];
        snprintf(htp_uri, आकार(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        संरचना remote_rpc_get_uri u = {};
        u.session_id      = हे->session_id;
        u.domain_name     = const_cast<अक्षर *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<अक्षर *>(htp_uri);
        u.module_uri_len  = म_माप(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = आकार(session_uri);

        पूर्णांक err = remote_session_control(FASTRPC_GET_URI, (शून्य *) &u, आकार(u));
        जर (err != AEE_SUCCESS) {
            snprintf(session_uri, आकार(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), हे->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          हे->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    पूर्णांक err = htp_iface_open(session_uri, &हे->handle);
    जर (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", हे->c_name(), session_uri, err);
        फेका std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    हे->valid_handle = सत्य;

    // Query HW info and resolve session options
    {
        अचिन्हित पूर्णांक hw_n_threads = 0;
        अचिन्हित पूर्णांक hw_n_hvx     = 0;
        अचिन्हित पूर्णांक hw_n_hmx     = 0;
        अचिन्हित दीर्घ दीर्घ hw_vtcm_size = 0;
        पूर्णांक hw_err = htp_iface_hwinfo(हे->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        जर (hw_err == 0) {
            स्थिरांक uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            हे->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (माप_प्रकार) max_n_threads) : max_n_threads;
            हे->n_hvx     = हे->n_threads;
            हे->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            हे->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          हे->c_name(), हे->n_threads, हे->n_hvx, हे->n_hmx,
                          (अचिन्हित दीर्घ दीर्घ)(हे->vtcm_size / (1024 * 1024)));
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", हे->c_name(), hw_err);
            स्थिरांक uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            हे->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (माप_प्रकार) HTP_MAX_NTHREADS) : default_n_threads;
            हे->n_hvx     = हे->n_threads;
            हे->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            हे->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        संरचना remote_rpc_control_latency l;
        l.enable = 1;

        पूर्णांक err = remote_handle64_control(हे->handle, DSPRPC_CONTROL_LATENCY, (शून्य *) &l, आकार(l));
        जर (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (अचिन्हित) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", हे->c_name(),
                  हे->session_id, हे->domain_id, session_uri, (अचिन्हित दीर्घ) हे->handle);

    स्थिरांक माप_प्रकार req_q_size = (आकार(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    स्थिरांक माप_प्रकार rsp_q_size = (आकार(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(हे->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (शून्य *) हे,  // Callback context
                          &queue);
    जर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", हे->name.c_str(), (अचिन्हित) err);
        फेका std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    हे->valid_queue = सत्य;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &हे->queue_id);
    जर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (अचिन्हित) err);
        फेका std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    जर (opt_etm) {
        err = htp_iface_etm(हे->handle, 1);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (अचिन्हित) err);
        }
    }

    // Allocate buffers and state for op batching
    हे->op_queue = नवीन ggml_hexagon_opqueue(हे, opt_opbatch, opt_opqueue);

    हे->fence_buf = नवीन ggml_hexagon_fence_buffer(हे, &dev_ctx->fence_buffer_type, 64 * 1024);
    जर (हे->mdev.count > 1) {
        हे->mdev_fence_slot = हे->alloc_fence(हे->mdev.count);
    }

    जर (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(हे);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", हे->c_name(), opt_vmem);
    }
    स्थिरांक माप_प्रकार shm_size = हे->op_queue->shm_size();
    हे->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    हे->op_batch = नवीन ggml_hexagon_opbatch(हे, opt_opbatch, हे->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(हे->handle, हे->session_id, हे->queue_id, हे->n_threads, opt_nhmx, हे->max_vmem, (uint32_t) opt_power_mode);
    जर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", हे->c_name(), (अचिन्हित) err);
        फेका std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    हे->valid_iface = सत्य;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", हे->c_name(), htp_power_mode_to_str(opt_power_mode));

    जर (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(हे->handle, opt_profile, &pmu_conf);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (अचिन्हित) err);
        }
    }
}

शून्य ggml_hexagon_session::release() noexcept(सत्य) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", हे->name.c_str());

    हे->mdev.sessions.clear();

    पूर्णांक err;

    जर (हे->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(हे->handle);
        जर (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (अचिन्हित) err);
        }
    }

    delete हे->op_batch;
    delete हे->op_queue;
    साठी (स्वतः & it : हे->cpy_fence_slots) {
        free_fence((शून्य *) it.second, 1);
    }
    हे->cpy_fence_slots.clear();

    जर (हे->fence_buf) {
        unclone_buffer(हे->fence_buf);
        delete हे->fence_buf;
        हे->fence_buf = nullptr;
    }
    पर्यंत (!हे->cloned_buffers.empty()) {
        release_buffer(हे->cloned_buffers.begin()->second.get());
    }

    जर (opt_etm) {
        err = htp_iface_etm(हे->handle, 0);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (अचिन्हित) err);
        }
    }

    जर (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(हे->handle, 0, &pmu_conf);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (अचिन्हित) err);
        }
    }

    जर (हे->valid_queue) {
        err = dspqueue_close(queue);
        जर (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (अचिन्हित) err);
        }
    }

    जर (हे->valid_handle) {
        htp_iface_close(हे->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(स्थिरांक ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_मोजणी) noexcept(असत्य) {
    हे->dev        = dev;
    हे->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    हे->mdev.idx   = mdev_idx;
    हे->mdev.count = mdev_मोजणी > 0 ? mdev_मोजणी : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)हे) & 0xFFFF;

    प्रयत्न {
        allocate(config);
        जर (हे->mdev.idx == 0 && !config.mdev_group.empty()) {
            साठी (माप_प्रकार i = 0; i < config.mdev_group.size(); i++) {
                हे->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], हे->dev, (uint32_t) (i + 1), हे->mdev.count));
            }
        }
    } पकडा (स्थिरांक std::exception & exc) {
        release();
        फेका;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(सत्य) {
    release();
}

// ** backend interface

स्थिर तार्किक ggml_hexagon_flash_attn_is_hmx_eligible(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * q,
    स्थिरांक संरचना ggml_tensor * k,
    स्थिरांक संरचना ggml_tensor * v,
    स्थिरांक संरचना ggml_tensor * sinks
) {
    जर (sess->n_hmx == 0) {
        परत असत्य;
    }

    जर (opt_fa_select < 2) {
        परत असत्य;
    }

    जर ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        परत असत्य;
    }

    स्थिरांक uint32_t DK = q->ne[0];
    स्थिरांक uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    जर (DK % 8 != 0 || DV % 8 != 0) {
        परत असत्य;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    जर (opt_fa_select > 2) {
        परत सत्य;
    }

    स्थिरांक uint32_t M = q->ne[1];

    // Prefill or batched decode
    जर (M > 1) {
        परत सत्य;
    }

    स्थिरांक uint32_t n_head = q->ne[2];
    स्थिरांक uint32_t n_kv_heads = k->ne[2];
    स्थिरांक uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    स्थिरांक uint32_t S = k->ne[1];
    स्थिरांक uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    जर (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        परत सत्य;
    }

    // Tile alignment for 32-row HMX tiles
    स्थिरांक तार्किक is_tile_aligned = (G > 0 && (32 % G == 0));
    जर (!is_tile_aligned) {
        जर (sess->n_threads >= 6) {
            परत असत्य;
        }
        परत S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    जर (DK == 256) {
        s_cross = 4096;
    } अन्यथा जर (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } अन्यथा {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    परत S >= s_cross;
}

स्थिर तार्किक ggml_hexagon_precompute_flash_attn_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_fa_kernel_params * kparams
) {
    जर (opt_fa_select < 1) {
        परत असत्य;
    }

    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक संरचना ggml_tensor * q    = op->src[0];
    स्थिरांक संरचना ggml_tensor * k    = op->src[1];
    स्थिरांक संरचना ggml_tensor * v    = op->src[2];
    स्थिरांक संरचना ggml_tensor * mask = op->src[3];

    स्थिरांक uint32_t neq0 = q->ne[0];  // head_dim (DK)
    स्थिरांक uint32_t neq1 = q->ne[1];  // n_tokens
    स्थिरांक uint32_t neq2 = q->ne[2];  // n_heads

    स्थिरांक uint32_t nek1 = k->ne[1];  // kv_len

    स्थिरांक uint32_t nev0 = v->ne[0];  // head_dim (DV)

    स्थिरांक uint32_t DK = neq0;
    स्थिरांक uint32_t DV = nev0;

    स्थिरांक uint32_t n_kv_heads = k->ne[2];
    स्थिरांक uint32_t G          = neq2 / n_kv_heads;

    दशांश scale         = 1.0f;
    दशांश max_bias      = 0.0f;
    दशांश logit_softcap = 0.0f;
    स_नकल(&scale,         &op->op_params[0], आकार(दशांश));
    स_नकल(&max_bias,      &op->op_params[1], आकार(दशांश));
    स_नकल(&logit_softcap, &op->op_params[2], आकार(दशांश));

    जर (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    स्थिरांक uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::न्यूनमान(std::log2(n_head));
    kparams->m0 = std::घात(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::घात(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    स्थिरांक संरचना ggml_tensor * sinks = op->src[4];
    जर (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        स्थिरांक uint32_t DK_pad = hex_round_up(DK, 64);
        स्थिरांक uint32_t DV_pad = hex_round_up(DV, 64);
        माप_प्रकार Br = 0, Bc = 0;
        पूर्णांक ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        जर (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            जर (logit_softcap == 0.0f) {
                kparams->scale = scale * (दशांश) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } अन्यथा {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (दशांश) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            स्थिरांक माप_प्रकार row_vec_bytes = hex_align_up(Bc * आकार(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            स्थिरांक माप_प्रकार m_ओळ_bytes = hex_align_up(Bc * आकार(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_ओळ_bytes / आकार(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_मूल्ये(G);
            जर (mask) {
                kparams->src3_div2 = init_fastdiv_मूल्ये(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_मूल्ये(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            परत सत्य;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    स्थिरांक माप_प्रकार size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    स्थिरांक माप_प्रकार size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    स्थिरांक माप_प्रकार size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

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
    जर (mask) {
        kparams->src3_div2 = init_fastdiv_मूल्ये(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_मूल्ये(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_flash_attn_ext(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * src3 = op->src[3];
    स्थिरांक संरचना ggml_tensor * src4 = op->src[4];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    जर ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        परत असत्य;
    }

    जर (src3 && src3->type != GGML_TYPE_F16) {  // mask
        परत असत्य;
    }

    जर (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        परत असत्य;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    जर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        परत असत्य;
    }

    संरचना htp_fa_kernel_params kparams;
    जर (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        परत असत्य;
    }

    जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_gated_delta_net(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    जर (opt_gdn_select < 1) {
        परत असत्य;
    }

    स्थिरांक संरचना ggml_tensor * q     = op->src[0];
    स्थिरांक संरचना ggml_tensor * k     = op->src[1];
    स्थिरांक संरचना ggml_tensor * v     = op->src[2];
    स्थिरांक संरचना ggml_tensor * g     = op->src[3];
    स्थिरांक संरचना ggml_tensor * beta  = op->src[4];
    स्थिरांक संरचना ggml_tensor * state = op->src[5];
    स्थिरांक संरचना ggml_tensor * dst   = op;

    जर (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    स्थिरांक int64_t S_v      = v->ne[0];
    स्थिरांक int64_t H        = v->ne[1];
    स्थिरांक int64_t n_tokens = v->ne[2];
    स्थिरांक int64_t n_seqs   = v->ne[3];
    स्थिरांक int64_t K        = ggml_get_op_params_i32(op, 0);

    जर (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        परत असत्य;
    }
    जर (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        परत असत्य;
    }
    जर ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        परत असत्य;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    जर (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        परत असत्य;
    }
    जर (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        परत असत्य;
    }

    स्थिरांक uint32_t total_rows = (uint32_t) (H * n_seqs);
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    स्थिरांक तार्किक can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    जर (can_use_hmx) {
        संरचना htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        जर (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, सत्य, &n_heads_batch)) {
            परत असत्य;
        }
    } अन्यथा {
        संरचना htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        जर (layout.total_bytes > sess->vtcm_size) {
            परत असत्य;
        }
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_matmul_is_hmx_eligible(
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    तार्किक is_matmul_id,
    तार्किक is_batched
) {
    GGML_UNUSED(dst);

    जर (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        परत असत्य;
    }

    स्थिरांक पूर्णांक ne00  = src0->ne[0];
    स्थिरांक पूर्णांक ne01  = src0->ne[1];
    स्थिरांक पूर्णांक ne11  = src1->ne[1];
    स्थिरांक पूर्णांक ne12  = src1->ne[2];
    स्थिरांक पूर्णांक wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    जर (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        परत असत्य;
    }

    // HMX supports F16, F32, and repack quantized types.
    जर (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        परत असत्य;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    जर (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        परत असत्य;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    स्थिरांक तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    जर (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        परत असत्य;
    }
    जर (!is_matmul_id && is_batched && is_tiled) {
        परत असत्य;
    }

    // HMX assumes contiguous row-major layout.
    जर (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        परत असत्य;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    स्थिरांक पूर्णांक m = is_matmul_id ? ne12 : ne11;
    जर (m <= HTP_MM_HMX_MIN_NROWS) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(dst);
}

स्थिर तार्किक ggml_hexagon_precompute_hmx_mm_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * act,
    स्थिरांक संरचना ggml_tensor * dst,
    पूर्णांक wtype,
    पूर्णांक ne00_padded,
    पूर्णांक ne01_padded,
    पूर्णांक ne02,
    पूर्णांक ne11,
    पूर्णांक ne12,
    पूर्णांक ne11_padded,
    तार्किक is_matmul_id,
    तार्किक is_batched,
    माप_प्रकार src2_size,
    माप_प्रकार vtcm_budget,
    संरचना htp_mm_kernel_params * kparams
) {
    स्थिरांक तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिरांक पूर्णांक aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    स्थिरांक पूर्णांक n_threads = (पूर्णांक)sess->n_threads;
    स्थिरांक पूर्णांक ne10 = act->ne[0];

    पूर्णांक m_for_solver = ne11;
    पूर्णांक m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    जर (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        स्थिरांक तार्किक dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : सत्य;
        स्थिरांक तार्किक act_row_split = ggml_hexagon_can_row_partition(act);
        जर (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (पूर्णांक) sess->mdev.count - 1) / (पूर्णांक) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    स्थिरांक तार्किक pipeline = is_matmul_id ? असत्य : htp_mm_hmx_pipeline(m_for_solver);

    स्थिरांक तार्किक is_batched_val = is_matmul_id ? असत्य : is_batched;
    स्थिरांक पूर्णांक group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    माप_प्रकार m_chunk = 0;
    माप_प्रकार n_chunk = 0;
    माप_प्रकार vtcm_size = 0;
    तार्किक use_grouped = असत्य;
    पूर्णांक act_threads_selected = 0;

    जर (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        जर (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = सत्य;
        }
    }

    जर (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        पूर्णांक m_id_rows = 0;
        जर (dst && is_matmul_id) {
            स्थिरांक पूर्णांक n_experts = ne02 > 0 ? ne02 : 1;
            स्थिरांक माप_प्रकार total_expert_rows = (माप_प्रकार) dst->ne[1] * dst->ne[2];
            पूर्णांक m_per_expert = (पूर्णांक) ((total_expert_rows + n_experts - 1) / n_experts);
            जर (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (पूर्णांक) sess->mdev.count - 1) / (पूर्णांक) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        स्थिरांक uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        जर (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            परत असत्य;
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

    जर (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    परत सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_hvx_mm_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    पूर्णांक wtype,
    तार्किक is_tiled,
    पूर्णांक ne02,
    पूर्णांक ne03,
    पूर्णांक ne10,
    पूर्णांक ne11,
    पूर्णांक ne12,
    पूर्णांक ne13,
    तार्किक is_matmul_id,
    स्थिरांक माप_प्रकार src2_row_size,
    माप_प्रकार vtcm_budget,
    संरचना htp_mm_kernel_params * kparams
) {
    जर (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    स्थिरांक पूर्णांक src1_nrows = ne11 * ne12 * ne13;

    जर (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        स्थिरांक तार्किक k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        जर (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (पूर्णांक) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            संरचना htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            साठी (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, सत्य, असत्य
                );
                जर (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    थांबा;
                }
            }
            जर (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                परत;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            परत;
        } अन्यथा {
            तार्किक try_tiled = (k_align && opt_mm_select >= 1);
            जर (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                जर (src1_nrows < (पूर्णांक) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } अन्यथा {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                संरचना htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                साठी (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, असत्य, असत्य
                    );
                    जर (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        थांबा;
                    }
                }

                uint32_t m_chunk = 0;
                जर (htp_mm_hvx_solve_vtcm_params(
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
                    परत;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            परत;
        }
    } अन्यथा जर (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    } अन्यथा जर (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        जर (htp_mm_hvx_solve_vtcm_params(
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
            परत;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    } अन्यथा जर (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        जर (htp_mm_hvx_solve_vtcm_params(
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
            परत;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    } अन्यथा जर (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        संरचना htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        जर (htp_mm_hvx_solve_vtcm_params(
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
            परत;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    }
}

स्थिर शून्य ggml_hexagon_precompute_matmul_params_impl(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    स्थिरांक माप_प्रकार src2_row_size,
    स्थिरांक माप_प्रकार src2_size,
    संरचना htp_mm_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक पूर्णांक ne00 = src0->ne[0];
    स्थिरांक पूर्णांक ne01 = src0->ne[1];
    स्थिरांक पूर्णांक ne02 = src0->ne[2];
    स्थिरांक पूर्णांक ne03 = src0->ne[3];

    स्थिरांक पूर्णांक ne10 = src1->ne[0];
    स्थिरांक पूर्णांक ne11 = src1->ne[1];
    स्थिरांक पूर्णांक ne12 = src1->ne[2];
    स्थिरांक पूर्णांक ne13 = src1->ne[3];

    स्थिरांक तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिरांक पूर्णांक wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    स्थिरांक पूर्णांक ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    स्थिरांक पूर्णांक ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    स्थिरांक पूर्णांक ne01_tiled  = hex_round_up(ne01, 32);

    स्थिरांक तार्किक is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    स्थिरांक तार्किक is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    स्थिरांक माप_प्रकार vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    तार्किक hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    जर (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        जर (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            जा finalize;
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
स्थिर तार्किक ggml_hexagon_rows_stride(स्थिरांक int64_t * ne, स्थिरांक माप_प्रकार * nb, माप_प्रकार * stride) {
    माप_प्रकार s = 0, next = 0;
    साठी (पूर्णांक i = 1; i < GGML_MAX_DIMS; i++) {
        जर (ne[i] == 1) सुरूठेवा;
        जर (s == 0) { s = nb[i]; next = s * ne[i]; सुरूठेवा; }
        जर (nb[i] != next) परत असत्य;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    परत सत्य;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
स्थिर तार्किक ggml_hexagon_matmul_can_collapse(स्थिरांक संरचना ggml_tensor * src0, स्थिरांक संरचना ggml_tensor * src1, स्थिरांक संरचना ggml_tensor * dst) {
    माप_प्रकार s1, sd;
    परत (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

स्थिर तार्किक ggml_hexagon_matmul_add_can_collapse(
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * src2,
    स्थिरांक संरचना ggml_tensor * dst
) {
    जर (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        परत असत्य;
    }
    जर (!src2) {
        परत सत्य;
    }
    स्थिरांक int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    जर (src2_nrows == 1) {
        परत सत्य;
    }
    माप_प्रकार s2;
    परत src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

स्थिर ggml_tensor ggml_hexagon_tensor_collapse_rows(स्थिरांक संरचना ggml_tensor * t) {
    माप_प्रकार stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    परत c;
}

स्थिर शून्य ggml_hexagon_precompute_matmul_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
) {
    जर (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        स्थिरांक ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        स्थिरांक ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        परत;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

स्थिर शून्य ggml_hexagon_precompute_fused_matmul_add_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * src2,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_mm_kernel_params * kparams
) {
    जर (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        स्थिरांक ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        स्थिरांक ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        स्थिरांक ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        स्थिरांक संरचना ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        स्थिरांक माप_प्रकार src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        परत;
    }
    स्थिरांक माप_प्रकार src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

स्थिर तार्किक ggml_hexagon_precompute_द्विमान_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    uint32_t op,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_द्विमान_kernel_params * kparams,
    स्थिरांक संरचना ggml_tensor * src2
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक माप_प्रकार elem_size = ggml_type_size(src0->type);
    स्थिरांक माप_प्रकार src0_row_size = src0->ne[0] * elem_size;
    स्थिरांक माप_प्रकार src1_row_size = src1->ne[0] * elem_size;
    स्थिरांक माप_प्रकार dst_row_size  = dst->ne[0]  * elem_size;

    स्थिरांक माप_प्रकार src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    स्थिरांक माप_प्रकार src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    स्थिरांक माप_प्रकार dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    जर (!htp_द्विमान_strides_safe(src0, src1, dst, elem_size)) {
        परत असत्य;
    }

    स्थिरांक तार्किक is_add_id = op == HTP_OP_ADD_ID;
    स्थिरांक तार्किक is_scalar = !is_add_id && src1->ne[0] == 1;
    स्थिरांक तार्किक is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    स्थिरांक तार्किक is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    स्थिरांक तार्किक is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    स्थिरांक तार्किक is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    स्थिरांक तार्किक is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    जर (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        स्थिरांक uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        स्थिरांक uint32_t n_threads = sess->n_threads;
        स्थिरांक uint32_t max_chunk_elems = 32768 / elem_size;
        स्थिरांक uint32_t min_chunk_elems = 256;
        स्थिरांक uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        स्थिरांक uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        स्थिरांक uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

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
        जर (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            परत असत्य;
        }

        kparams->vtcm_size = L.total_bytes;
        परत सत्य;
    }

    गणना htp_द्विमान_kernel_type kernel_type;
    माप_प्रकार src1_size = 0;
    माप_प्रकार src2_size = 0;

    जर (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_द्विमान_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * आकार(int32_t), 128);
        जर (src1_size + src2_size > sess->vtcm_size) {
            परत असत्य;
        }
    } अन्यथा जर (is_row_bcast) {
        kernel_type = HTP_द्विमान_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } अन्यथा जर (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        जर (src1_size > sess->vtcm_size) {
            परत असत्य;
        }
        kernel_type = HTP_द्विमान_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा जर (is_same_shape) {
        kernel_type = HTP_द्विमान_KERNEL_SAME_SHAPE;
    } अन्यथा जर (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        जर (src1_size > sess->vtcm_size) {
            परत असत्य;
        }
        kernel_type = HTP_द्विमान_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        जर (src1_size > sess->vtcm_size) {
            परत असत्य;
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

    स्थिरांक तार्किक src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    स्थिरांक तार्किक src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    स्थिरांक तार्किक dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    स्थिरांक तार्किक src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    स्थिरांक तार्किक src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    स्थिरांक तार्किक dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    तार्किक split_at_ne01 = असत्य;
    जर (src0->ne[2] > 1) {
        जर (kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } अन्यथा जर (kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } अन्यथा {
            split_at_ne01 = सत्य;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    तार्किक split_at_ne02 = असत्य;
    जर (src0->ne[3] > 1) {
        जर (kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } अन्यथा जर (kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } अन्यथा {
            split_at_ne02 = सत्य;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    संरचना htp_द्विमान_vtcm_layout L;
    htp_द्विमान_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    जर (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        जर (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            स्थिरांक uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            स्थिरांक uint32_t n_threads = sess->n_threads;
            स्थिरांक uint32_t max_chunk_elems = 32768 / elem_size;
            स्थिरांक uint32_t min_chunk_elems = 256;
            स्थिरांक uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            स्थिरांक uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            स्थिरांक uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_द्विमान_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_द्विमान_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            जर (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                परत असत्य;
            }

            kparams->vtcm_size = L.total_bytes;
            परत सत्य;
        }
        परत असत्य;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    परत सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_unary_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    uint32_t op,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_unary_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    स्थिरांक माप_प्रकार elem_size = ggml_type_size(src0->type);

    स्थिरांक माप_प्रकार src0_data_row_size = src0->ne[0] * elem_size;
    स्थिरांक माप_प्रकार dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    स्थिरांक माप_प्रकार src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    स्थिरांक माप_प्रकार dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    माप_प्रकार src1_data_row_size = 0;
    माप_प्रकार src1_row_size_aligned = 0;
    तार्किक broadcast_weight = असत्य;

    जर (op == HTP_OP_RMS_NORM_MUL) {
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

    स्थिरांक uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_मूल्ये(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_मूल्ये(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_मूल्ये(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_मूल्ये(tiles_per_row);
}

स्थिर शून्य ggml_hexagon_precompute_get_rows_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_get_rows_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक uint32_t ne00 = src0->ne[0];
    स्थिरांक uint32_t ne02 = src0->ne[2];
    स्थिरांक uint32_t ne03 = src0->ne[3];

    स्थिरांक uint32_t ne10 = src1->ne[0];
    स्थिरांक uint32_t ne11 = src1->ne[1];
    स्थिरांक uint32_t ne12 = src1->ne[2];
    स्थिरांक uint32_t nr = ne10 * ne11 * ne12;

    स्थिरांक तार्किक tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिरांक पूर्णांक wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    जर (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } अन्यथा जर (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } अन्यथा {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    स्थिरांक uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    संरचना htp_get_rows_vtcm_layout vtcm_layout = {};
    पर्यंत (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        जर (vtcm_layout.total_bytes <= sess->vtcm_size) {
            थांबा;
        }
        --kparams->n_threads;
    }

    जर (kparams->n_threads == 0 && total_tasks > 0) {
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
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0, //मूल्ये
    स्थिरांक संरचना ggml_tensor * src1, // indices
    स्थिरांक संरचना ggml_tensor * dst,  // destination
    संरचना htp_set_rows_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक uint32_t nr = src0->ne[1];

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
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_softmax_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    दशांश scale = 1.0f;
    दशांश max_bias = 0.0f;
    स_नकल(&scale,    &op->op_params[0], आकार(दशांश));
    स_नकल(&max_bias, &op->op_params[1], आकार(दशांश));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    स्थिरांक uint32_t n_head = src0->ne[2];
    स्थिरांक uint32_t n_head_log2 = 1u << (uint32_t) न्यूनमान(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    जर (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } अन्यथा {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    स्थिरांक uint32_t ne00 = src0->ne[0];
    स्थिरांक uint32_t ne10 = src1 ? src1->ne[0] : 1;

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
    जर (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } अन्यथा जर (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } अन्यथा {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    जर (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_मूल्ये(src0->ne[1]);
    जर (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_मूल्ये(src0->ne[2]);
    स्थिरांक uint32_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिरांक uint32_t ne13 = src1 ? src1->ne[3] : 1;
    जर (ne12 > 0) kparams->div_ne12 = init_fastdiv_मूल्ये(ne12);
    जर (ne13 > 0) kparams->div_ne13 = init_fastdiv_मूल्ये(ne13);
}

स्थिर शून्य ggml_hexagon_precompute_rope_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_rope_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    स्थिरांक uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

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

    जर (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_मूल्ये(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_मूल्ये(dst->ne[1]);
    }
}

स्थिर शून्य ggml_hexagon_precompute_ssm_conv_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_ssm_conv_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक uint32_t d_conv  = (uint32_t) src1->ne[0];
    स्थिरांक uint32_t d_inner = (uint32_t) src0->ne[1];
    स्थिरांक uint32_t n_t     = (uint32_t) dst->ne[1];
    स्थिरांक uint32_t n_s     = (uint32_t) dst->ne[2];
    स्थिरांक uint32_t ncs     = (uint32_t) src0->ne[0];

    स्थिरांक uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    स्थिरांक uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    स्थिरांक uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    स्थिरांक uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * आकार(दशांश), 128) + 128;
    स्थिरांक uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * आकार(दशांश), 128);
    स्थिरांक uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    जर (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        स्थिरांक uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * आकार(दशांश), 128);
        स्थिरांक uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * आकार(दशांश), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        स्थिरांक uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * आकार(दशांश), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } अन्यथा {
        स्थिरांक माप_प्रकार vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        स्थिरांक uint32_t src0_block_T    = hex_round_up(ncs * 32 * आकार(दशांश), 128);
        स्थिरांक माप_प्रकार   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        स्थिरांक माप_प्रकार   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * आकार(दशांश)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        जर (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        जर (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        जर (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        स्थिरांक uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * आकार(दशांश), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * आकार(दशांश), 128);
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
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_gdn_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक संरचना ggml_tensor * q     = op->src[0];
    स्थिरांक संरचना ggml_tensor * k     = op->src[1];
    स्थिरांक संरचना ggml_tensor * v     = op->src[2];
    स्थिरांक संरचना ggml_tensor * g     = op->src[3];
    स्थिरांक संरचना ggml_tensor * state = op->src[5];

    स्थिरांक uint32_t S_v      = (uint32_t) v->ne[0];
    स्थिरांक uint32_t H        = (uint32_t) v->ne[1];
    स्थिरांक uint32_t n_tokens = (uint32_t) v->ne[2];
    स्थिरांक uint32_t n_seqs   = (uint32_t) v->ne[3];
    स्थिरांक uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    स्थिरांक uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    स्थिरांक uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    स्थिरांक uint32_t total_rows = H * n_seqs;
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    स्थिरांक तार्किक can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    संरचना htp_gdn_hmx_vtcm_layout hmx_layout;
    संरचना htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    जर (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, सत्य, &n_heads_batch)) {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / आकार(दशांश));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((दशांश) S_v);

    जर (H > 0)                  kparams->div_H         = init_fastdiv_मूल्ये(H);
    जर (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_मूल्ये((uint32_t) q->ne[1]);
    जर (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_मूल्ये((uint32_t) k->ne[1]);
    जर (rq3 > 0)                kparams->div_rq3       = init_fastdiv_मूल्ये(rq3);
    जर (rk3 > 0)                kparams->div_rk3       = init_fastdiv_मूल्ये(rk3);
    जर (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_मूल्ये(kparams->n_threads);
}

स्थिर शून्य ggml_hexagon_precompute_sort_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    तार्किक is_शीर्ष_k,
    संरचना htp_sort_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    स्थिरांक uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t ne00       = src0->ne[0];
    स्थिरांक uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    जर (!is_शीर्ष_k) {
        order = ((स्थिरांक int32_t *) op->op_params)[0];
    }

    स्थिरांक uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिरांक माप_प्रकार vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

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
    स्थिरांक uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    स्थिरांक int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    जर (numer_hi < 0) {
        *ox_hi = 0;
    } अन्यथा {
        स्थिरांक uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    जर (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

स्थिर तार्किक ggml_hexagon_pool_shape_fits(
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * dst
) {
    परत ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

स्थिर शून्य ggml_hexagon_precompute_pool_2d_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * dst,
    संरचना htp_pool_2d_kernel_params * kparams,
    तार्किक is_pool_1d
) {
    स_रखो(kparams, 0, आकार(*kparams));

    स्थिरांक uint32_t src_x = (uint32_t) src0->ne[0];
    स्थिरांक uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    स्थिरांक uint32_t dst_x = (uint32_t) dst->ne[0];
    स्थिरांक uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    स्थिरांक uint32_t src_plane_bytes = src_x * src_y * आकार(दशांश);
    स्थिरांक uint32_t dst_plane_bytes = dst_x * dst_y * आकार(दशांश);
    स्थिरांक uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    स्थिरांक uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    स्थिरांक तार्किक ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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

    स्थिरांक तार्किक narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (दशांश) (kparams->kernel_x * kparams->kernel_y);
}

स्थिर तार्किक ggml_hexagon_precompute_concat_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_concat_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!src0 || !src1 || !dst) {
        परत असत्य;
    }

    पूर्णांक dim = ((स्थिरांक int32_t *) op->op_params)[0];
    जर (dim < 0 || dim >= GGML_MAX_DIMS) {
        परत असत्य;
    }
    kparams->dim = dim;

    जर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        परत असत्य;
    }
    जर (src0->type != dst->type || src1->type != dst->type) {
        परत असत्य;
    }

    स्थिरांक uint32_t type_size = ggml_type_size(dst->type);

    साठी (पूर्णांक d = 0; d < GGML_MAX_DIMS; d++) {
        स्थिरांक int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        जर (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            परत असत्य;
        }
    }

    स्थिरांक uint32_t max_gather_ne0 = 128 / type_size;
    स्थिरांक uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    स्थिरांक तार्किक b_dense = (src1->nb[0] == type_size);
    स्थिरांक तार्किक gather_strides_ok =
        src0->nb[1] >= (माप_प्रकार) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (माप_प्रकार) src1->ne[0] * type_size
                 : src1->nb[0] >= (माप_प्रकार) src1->ne[1] * type_size);

    जर (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (माप_प्रकार) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        जर (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            परत सत्य;
        }
    }

    स्थिरांक तार्किक dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    जर (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        परत सत्य;
    }

    स्थिरांक तार्किक is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    स्थिरांक तार्किक is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    स्थिरांक तार्किक transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    जर (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        स्थिरांक uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        संरचना htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        जर (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            परत असत्य;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        परत सत्य;
    }

    परत असत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_cpy_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_copy_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!src0 || !dst) {
        परत असत्य;
    }

    जर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        परत असत्य;
    }
    जर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    स्थिरांक int64_t nelem_src = ggml_nelements(src0);
    स्थिरांक int64_t nelem_dst = ggml_nelements(dst);
    जर (nelem_src != nelem_dst || nelem_src < 0) {
        परत असत्य;
    }

    स्थिरांक uint32_t src_type_size = ggml_type_size(src0->type);
    स्थिरांक uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    जर (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        परत सत्य;
    }

    जर (nelem_src == 1) {
        जर (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            परत सत्य;
        }
        जर ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            परत सत्य;
        }
        परत असत्य;
    }

    स्थिरांक तार्किक sametype = (src0->type == dst->type);

    तार्किक same_extents = सत्य;
    साठी (पूर्णांक d = 0; d < GGML_MAX_DIMS; d++) {
        जर (src0->ne[d] != dst->ne[d]) {
            same_extents = असत्य;
            थांबा;
        }
    }

    स्थिरांक तार्किक transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (माप_प्रकार) src0->ne[0] * src_type_size) || (dst->nb[1] < (माप_प्रकार) dst->ne[0] * dst_type_size);
    स्थिरांक तार्किक sameshape  = same_extents && !transposed;

    स्थिरांक तार्किक src_is_contiguous = ggml_is_contiguous(src0);
    स्थिरांक तार्किक dst_is_contiguous = ggml_is_contiguous(dst);

    जर (sametype) {
        जर (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            परत सत्य;
        }

        स्थिरांक uint32_t max_gather_ne0 = 128 / src_type_size;
        स्थिरांक uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        जर ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (माप_प्रकार) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            जर (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, असत्य,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                परत सत्य;
            }
        }

        जर (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            परत सत्य;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_मूल्ये((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_मूल्ये((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_मूल्ये((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_मूल्ये((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_मूल्ये((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_मूल्ये((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        परत सत्य;
    }

    जर (!sameshape) {
        परत असत्य;
    }

    स्थिरांक तार्किक valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    जर (!valid_conversion) {
        परत असत्य;
    }

    स्थिरांक uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    संरचना htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    जर (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        परत असत्य;
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

    परत सत्य;
}

स्थिर शून्य ggml_hexagon_precompute_fused_mmnx_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0, // W0
    स्थिरांक संरचना ggml_tensor * act,  // x
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->n_threads = sess->n_threads;

    स्थिरांक पूर्णांक ne00 = src0->ne[0];
    स्थिरांक पूर्णांक ne01 = src0->ne[1];
    स्थिरांक पूर्णांक ne02 = src0->ne[2];
    स्थिरांक पूर्णांक ne03 = src0->ne[3];

    स्थिरांक पूर्णांक ne10 = act->ne[0];
    स्थिरांक पूर्णांक ne11 = act->ne[1];
    स्थिरांक पूर्णांक ne12 = act->ne[2];
    स्थिरांक पूर्णांक ne13 = act->ne[3];

    स्थिरांक तार्किक is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिरांक पूर्णांक wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    स्थिरांक पूर्णांक ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    स्थिरांक पूर्णांक ne11_padded = hex_round_up(ne11, 32);
    स्थिरांक पूर्णांक ne01_tiled  = hex_round_up(ne01, 32);

    स्थिरांक माप_प्रकार vtcm_budget = sess->vtcm_size;
    स्थिरांक तार्किक is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    तार्किक hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    जर (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, असत्य, is_batched)) {
        जर (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, असत्य, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            जा finalize;
        }
    }

    जर (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        परत;
    }

    {
        स्थिरांक पूर्णांक act_nrows = ne11 * ne12 * ne13;
        स्थिरांक माप_प्रकार act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        स्थिरांक माप_प्रकार src0_row_size = src0->nb[1];

        स्थिरांक uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        साठी (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            संरचना htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, असत्य, सत्य
            );
            जर (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                थांबा;
            }
        }

        संरचना htp_mm_hvx_vtcm_layout L;
        तार्किक try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, असत्य, सत्य
        );

        जर (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
            परत;
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
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * src0, // W0
    स्थिरांक संरचना ggml_tensor * act,  // x
    स्थिरांक संरचना ggml_tensor * dst,  // dst0
    int32_t n_weights,
    संरचना htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

स्थिर तार्किक ggml_hexagon_op_mul_mat_use_fwht(स्थिरांक संरचना ggml_tensor * dst) {
    जर (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        परत असत्य;
    }

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    स्थिरांक int64_t n = src1->ne[0];
    जर (n < 32 || (n & (n - 1)) != 0) {
        परत असत्य;
    }

    परत src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

स्थिर तार्किक ggml_hexagon_supported_mul_mat(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * dst) {
    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    जर (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        जर (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            परत असत्य;
        }
        संरचना htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        परत ukparams.vtcm_row_per_thread > 0 && (माप_प्रकार) ukparams.vtcm_size <= sess->vtcm_size;
    }

    जर (dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        परत असत्य;
    }

    निवड (src0->type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q2_K: {
            जर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                परत असत्य;
            }

            स्थिरांक int64_t qk = ggml_blck_size(src0->type);
            जर (src0->ne[0] % qk) {
                परत असत्य;
            }

            जर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                परत असत्य;
            }
            जर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                परत असत्य;
            }
            थांबा;
        }

        स्थिती GGML_TYPE_F16:
            जर (src0->nb[1] < src0->nb[0]) {
                परत असत्य;
            }
            जर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                परत असत्य;
            }
            जर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                परत असत्य;
            }
            थांबा;

        स्थिती GGML_TYPE_F32:
            जर (src1->type != GGML_TYPE_F32) {
                परत असत्य;
            }
            जर (src0->nb[1] < src0->nb[0]) {
                परत असत्य;
            }
            जर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                परत असत्य;
            }
            जर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                परत असत्य;
            }
            थांबा;

        स्थिती GGML_TYPE_BF16:
            जर (src0->nb[1] < src0->nb[0]) {
                परत असत्य;
            }
            जर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                परत असत्य;
            }
            जर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                परत असत्य;
            }
            थांबा;

        पूर्वनिर्धारित:
            परत असत्य;
    }

    संरचना htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    जर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_mul_mat_id(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    जर (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        परत असत्य;
    }

    निवड (src0->type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q2_K:
            जर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                परत असत्य;
            }

            जर (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                परत असत्य;
            }
            थांबा;

        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_BF16:
            जर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                परत असत्य;
            }
            जर (src0->ne[0] % 32 != 0) {
                परत असत्य;
            }
            थांबा;

        पूर्वनिर्धारित:
            परत असत्य;
    }

    संरचना htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    जर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_द्विमान(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type == GGML_TYPE_F32) {
        जर (src1->type != GGML_TYPE_F32) {
            परत असत्य;
        }
        जर (dst->type != GGML_TYPE_F32) {
            परत असत्य;
        }
    }
    अन्यथा जर (src0->type == GGML_TYPE_F16) {
        जर (src1->type != GGML_TYPE_F16) {
            परत असत्य;
        }
        जर (dst->type != GGML_TYPE_F16) {
            परत असत्य;
        }
    }
    अन्यथा {
        परत असत्य;
    }

    जर (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        परत असत्य;
    }
    जर (!ggml_are_same_shape(src0, dst)) {
        परत असत्य;
    }
    जर (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        परत असत्य;
    }

    संरचना htp_द्विमान_kernel_params kparams;
    परत ggml_hexagon_precompute_द्विमान_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_add_id(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!src2) {
        परत असत्य;
    }
    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        परत असत्य;
    }
    जर (!ggml_are_same_shape(src0, dst)) {
        परत असत्य;
    }
    जर (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        परत असत्य;
    }
    जर (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        परत असत्य;
    }
    जर (src0->nb[0] != आकार(दशांश) || src1->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश) || src2->nb[0] != आकार(int32_t)) {
        परत असत्य;
    }

    // REVISIT: add support for non-contiguous tensors
    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }
    जर (src2->nb[1] < src2->ne[0] * आकार(int32_t)) {
        परत असत्य;
    }

    संरचना htp_द्विमान_kernel_params kparams;
    परत ggml_hexagon_precompute_द्विमान_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

स्थिर तार्किक ggml_hexagon_supported_unary(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        परत असत्य;
    }
    जर (dst->type != src0->type) {
        परत असत्य;
    }
    जर (!ggml_is_contiguous_rows(src0)) {
        परत असत्य;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    जर (src0->type == GGML_TYPE_F16) {
        निवड (op->op) {
            स्थिती GGML_OP_NORM:
            स्थिती GGML_OP_RMS_NORM:
            स्थिती GGML_OP_L2_NORM:
            स्थिती GGML_OP_SCALE:
            स्थिती GGML_OP_CLAMP:
            स्थिती GGML_OP_SQR:
            स्थिती GGML_OP_SQRT:
            स्थिती GGML_OP_LOG:
                थांबा;
            स्थिती GGML_OP_UNARY:
                जर (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    परत असत्य;
                }
                थांबा;
            पूर्वनिर्धारित:
                परत असत्य;
        }
    }

    जर (!ggml_are_same_shape(src0, dst)) {
        परत असत्य;
    }

    // dst must be contiguous; src0 may be non-contiguous
    जर (!ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_sum(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_sum_rows(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    // TODO: add support for non-contigiuos tensors
    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_argmax(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (dst->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_pool_2d(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक int32_t * params = op->op_params;

    जर (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        परत असत्य;
    }
    जर (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        परत असत्य;
    }
    जर (!ggml_hexagon_pool_shape_fits(src0, op)) {
        परत असत्य;
    }

    स्थिरांक int32_t kernel_x = params[1];
    स्थिरांक int32_t kernel_y = params[2];
    स्थिरांक int32_t stride_x = params[3];
    स्थिरांक int32_t stride_y = params[4];
    स्थिरांक int32_t pad_x    = params[5];
    स्थिरांक int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    जर (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        परत असत्य;
    }

    स्थिरांक uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    स्थिरांक uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    परत htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

स्थिर तार्किक ggml_hexagon_supported_pool_1d(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक int32_t * params = op->op_params;

    जर (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        परत असत्य;
    }
    जर (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        परत असत्य;
    }
    जर (!ggml_hexagon_pool_shape_fits(src0, op)) {
        परत असत्य;
    }

    स्थिरांक int32_t kernel = params[1];
    स्थिरांक int32_t stride = params[2];
    स्थिरांक int32_t pad    = params[3];
    जर (kernel <= 0 || stride <= 0 || pad < 0) {
        परत असत्य;
    }

    स्थिरांक uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    स्थिरांक uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    संरचना htp_pool_vtcm_layout layout;
    परत htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

स्थिर तार्किक ggml_hexagon_supported_activations(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        परत असत्य;
    }
    जर (dst->type != src0->type) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous_1(src0)) {
        परत असत्य;
    }
    जर (!ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    जर (src1) {
        जर (src1->type != src0->type) {
            परत असत्य;
        }
        जर (!ggml_are_same_shape(src0, src1)) {
            परत असत्य;
        }
        जर (!ggml_is_contiguous_1(src1)) {
            परत असत्य;
        }
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_softmax(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src2) {
        परत असत्य;  // FIXME: add support for sinks
    }

    जर (src0->type != GGML_TYPE_F32) {
        परत असत्य;
    }
    जर (dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (src0->ne[2] > 512) {
        परत असत्य;
    }

    जर (src1) {
        जर (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            परत असत्य;
        }
        जर (src0->ne[0] != src1->ne[0]) {
            परत असत्य;
        }
        जर (src1->ne[1] < src0->ne[1]) {
            परत असत्य;
        }
        जर (src0->ne[2] % src1->ne[2] != 0) {
            परत असत्य;
        }
        जर (src0->ne[3] % src1->ne[3] != 0) {
            परत असत्य;
        }
    }

    जर (src1) {
        जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            परत असत्य;
        }
    } अन्यथा {
        जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            परत असत्य;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    स्थिरांक int64_t ne0 = src0->ne[0];
    जर (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        परत असत्य;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    जर (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        परत असत्य;
    }

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    संरचना htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    जर (layout.total_bytes > sess->vtcm_size) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_set_rows(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1]; // indices
    स्थिरांक संरचना ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    जर (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        परत असत्य;
    }

    जर (src0->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        परत असत्य;
    }

    जर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_get_rows(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1]; // indices
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        परत असत्य;
    }

    जर (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        परत असत्य;
    }

    जर (src0->type == dst->type) {
        जर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            परत असत्य;
        }
    } अन्यथा जर (src0->type == GGML_TYPE_I32) {
        परत असत्य;
    } अन्यथा जर (dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    जर (ggml_is_empty(op)) {
        परत सत्य;
    }

    संरचना htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    जर (kparams.n_threads == 0 || (माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_argsort(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    स्थिरांक संरचना ggml_tensor * dst  = op;         // indices

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    स्थिरांक uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिरांक माप_प्रकार vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    संरचना htp_sort_vtcm_layout layout;
    जर (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, असत्य)) {
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_शीर्ष_k(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0]; //मूल्ये
    स्थिरांक संरचना ggml_tensor * dst  = op;         // indices

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    स्थिरांक uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिरांक माप_प्रकार vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    संरचना htp_sort_vtcm_layout layout;
    जर (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, सत्य)) {
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_rope(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * src2 = op->src[2];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!ggml_are_same_shape(src0, dst)) {
        परत असत्य;
    }

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    जर (src0->ne[0] <= 0) {
        परत असत्य;
    }

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    जर (src0_nrows == 0) {
        परत असत्य;
    }

    स्थिरांक int32_t * op_params = &op->op_params[0];
    स्थिरांक पूर्णांक n_dims = op_params[1];
    स्थिरांक पूर्णांक mode   = op_params[2];
    स्थिरांक पूर्णांक n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    जर (n_dims < 0 || n_dims % 2 != 0) {
        परत असत्य;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    जर (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        परत असत्य;
    }

    दशांश freq_base;
    स_नकल(&freq_base, op_params + 5, आकार(दशांश));
    जर (freq_base < 0.0f) {
        परत असत्य;
    }

    जर (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        परत असत्य;
    }

    स्थिरांक तार्किक is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    जर (mode == GGML_ROPE_TYPE_VISION) {
        जर (n_dims != (पूर्णांक) (src0->ne[0] / 2) || n_offs != 0) {
            परत असत्य;
        }
    }

    जर (is_mrope) {
        स्थिरांक int32_t * sections = op_params + 11;
        जर (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            परत असत्य;
        }
    }

    स्थिरांक int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    जर (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        परत असत्य;
    }

    जर (src2) {
        जर (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            परत असत्य;
        }
        जर (src2->ne[0] < (n_dims / 2)) {
            परत असत्य;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    जर (src0->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश)) {
        परत असत्य;
    }
    जर (src0->nb[1] < src0->ne[0] * आकार(दशांश) || dst->nb[1] < dst->ne[0] * आकार(दशांश)) {
        परत असत्य;
    }

    स्थिरांक uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    स्थिरांक uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    संरचना htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    जर (layout.total_bytes > sess->vtcm_size) {
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_ssm_conv(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    // Only support FP32 for now
    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    // Check IO tensor shapes and dims
    जर (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        परत असत्य; // src0 should be effectively 3D
    }

    स्थिरांक पूर्णांक d_conv  = src1->ne[0];
    स्थिरांक पूर्णांक d_inner = src0->ne[1];
    स्थिरांक पूर्णांक n_t = dst->ne[1];
    स्थिरांक पूर्णांक n_s = dst->ne[2];

    जर (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        परत असत्य;
    }
    जर (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        परत असत्य;
    }
    जर (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        परत असत्य;
    }
    जर (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        परत असत्य;
    }
    जर (src0->nb[0] != आकार(दशांश) || src1->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश)) {
        परत असत्य;
    }
    जर (src0->nb[1] != src0->ne[0] * आकार(दशांश) || src1->nb[1] != src1->ne[0] * आकार(दशांश)) {
        परत असत्य;
    }

    संरचना htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    जर ((माप_प्रकार) kparams.vtcm_size > sess->vtcm_size) {
        परत असत्य;
    }

    परत सत्य;
}

स्थिर तार्किक im2col_3d_calc_output_size(uint32_t आदान, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    जर (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        परत असत्य;
    }

    स्थिरांक int64_t numerator = (int64_t) आदान + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    जर (numerator < 0) {
        परत असत्य;
    }

    स्थिरांक uint64_t size = (uint64_t) (numerator / stride) + 1;
    जर (size == 0 || size > UINT32_MAX) {
        परत असत्य;
    }

    *output = (uint32_t) size;
    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_im2col_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!src0 || !src1 || !dst) {
        परत असत्य;
    }

    जर (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        परत असत्य;
    }

    जर (src1->nb[0] != आकार(दशांश) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    स्थिरांक int32_t * op_params = (स्थिरांक int32_t *) op->op_params;
    स्थिरांक int32_t s0 = op_params[0];
    स्थिरांक int32_t s1 = op_params[1];
    स्थिरांक int32_t p0 = op_params[2];
    स्थिरांक int32_t p1 = op_params[3];
    स्थिरांक int32_t d0 = op_params[4];
    स्थिरांक int32_t d1 = op_params[5];
    स्थिरांक तार्किक is_2D = (op_params[6] == 1);

    जर (s0 <= 0 || d0 <= 0 || p0 < 0) {
        परत असत्य;
    }
    जर (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        परत असत्य;
    }

    स्थिरांक uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    स्थिरांक uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    स्थिरांक uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    स्थिरांक uint32_t IW = (uint32_t) src1->ne[0];
    स्थिरांक uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    स्थिरांक uint32_t KW = (uint32_t) src0->ne[0];
    स्थिरांक uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    स्थिरांक uint32_t OW = (uint32_t) dst->ne[1];

    जर (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        परत असत्य;
    }

    स्थिरांक uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    स्थिरांक uint64_t total_rows_64   = (uint64_t) N * OH;
    जर (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        परत असत्य;
    }
    स्थिरांक uint32_t patch_stride = (uint32_t) patch_stride_64;
    स्थिरांक uint32_t total_rows   = (uint32_t) total_rows_64;

    जर (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        परत असत्य;
    }
    जर (is_2D && dst->ne[2] != OH) {
        परत असत्य;
    }

    स्थिरांक uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? आकार(ggml_fp16_t) : आकार(दशांश);

    तार्किक is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    जर (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    स्थिरांक uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    स्थिरांक uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    स्थिरांक uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    जर (is_patchembed) {
        स्थिरांक uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * आकार(दशांश);
        स्थिरांक uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        जर (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            स्थिरांक uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            स्थिरांक uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            संरचना htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            जर (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                परत सत्य;
            }
        }
    }

    साठी (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        स्थिरांक uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        जर (Wb > UINT32_MAX) {
            जर (owb == 1) थांबा;
            सुरूठेवा;
        }

        स्थिरांक uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * आकार(दशांश);
        स्थिरांक uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        जर (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            जर (owb == 1) थांबा;
            सुरूठेवा;
        }

        स्थिरांक uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        स्थिरांक uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        संरचना htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        जर (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            परत सत्य;
        }
        जर (owb == 1) थांबा;
    }

    परत असत्य;
}

स्थिर तार्किक ggml_hexagon_precompute_im2col_3d_params(
    स्थिरांक संरचना ggml_hexagon_session * sess,
    स्थिरांक संरचना ggml_tensor * op,
    संरचना htp_im2col_kernel_params * kparams
) {
    स_रखो(kparams, 0, आकार(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (!src0 || !src1 || !dst) {
        परत असत्य;
    }

    जर (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        परत असत्य;
    }

    जर (src1->nb[0] != आकार(दशांश) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    स्थिरांक int32_t * params = (स्थिरांक int32_t *) op->op_params;
    स्थिरांक int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    स्थिरांक int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    स्थिरांक int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    स्थिरांक int32_t ic = params[9];

    जर (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        परत असत्य;
    }

    जर (src1->ne[3] % ic != 0) {
        परत असत्य;
    }

    स्थिरांक uint32_t IC = (uint32_t) ic;
    स्थिरांक uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    जर (N == 0) {
        परत असत्य;
    }

    uint32_t OW, OH, OD;
    जर (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        परत असत्य;
    }

    स्थिरांक uint32_t KD = (uint32_t) src0->ne[2];
    स्थिरांक uint32_t KH = (uint32_t) src0->ne[1];
    स्थिरांक uint32_t KW = (uint32_t) src0->ne[0];
    स्थिरांक uint32_t ID = (uint32_t) src1->ne[2];
    स्थिरांक uint32_t IH = (uint32_t) src1->ne[1];
    स्थिरांक uint32_t IW = (uint32_t) src1->ne[0];

    स्थिरांक uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    स्थिरांक uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    स्थिरांक uint64_t dst_ne3_64      = (uint64_t) OD * N;

    जर (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        परत असत्य;
    }
    स्थिरांक uint32_t patch_stride = (uint32_t) patch_stride_64;
    स्थिरांक uint32_t total_rows   = (uint32_t) total_rows_64;
    स्थिरांक uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    जर (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        परत असत्य;
    }

    स्थिरांक uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? आकार(ggml_fp16_t) : आकार(दशांश);

    स्थिरांक तार्किक is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    स्थिरांक uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    स्थिरांक uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    स्थिरांक uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    जर (is_patchembed) {
        स्थिरांक uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * आकार(दशांश);
        स्थिरांक uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        जर (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            स्थिरांक uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            स्थिरांक uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            संरचना htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            जर (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                परत सत्य;
            }
        }
    }

    साठी (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        स्थिरांक uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        जर (Wb > UINT32_MAX) {
            जर (owb == 1) थांबा;
            सुरूठेवा;
        }

        स्थिरांक uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * आकार(दशांश);
        स्थिरांक uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        जर (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            जर (owb == 1) थांबा;
            सुरूठेवा;
        }

        स्थिरांक uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        स्थिरांक uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        संरचना htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        जर (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            परत सत्य;
        }
        जर (owb == 1) थांबा;
    }

    परत असत्य;
}

स्थिर तार्किक ggml_hexagon_supported_im2col(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    संरचना htp_im2col_kernel_params kparams;
    परत ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_im2col_3d(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    संरचना htp_im2col_kernel_params kparams;
    परत ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_pad(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    स्थिरांक int32_t lp0 = ((स्थिरांक int32_t *) op->op_params)[0];
    स्थिरांक int32_t rp0 = ((स्थिरांक int32_t *) op->op_params)[1];
    स्थिरांक int32_t circular = ((स्थिरांक int32_t *) op->op_params)[8];

    जर (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_cumsum(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_diag(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    // diag only supports F32 currently
    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    // Input must have ne[1] == 1 (vector input)
    जर (src0->ne[1] != 1) {
        परत असत्य;
    }

    // Output must be square in first two dimensions
    जर (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_solve_tri(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0]; // A
    स्थिरांक संरचना ggml_tensor * src1 = op->src[1]; // B
    स्थिरांक संरचना ggml_tensor * dst  = op;         // X

    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (src0->ne[0] != src0->ne[1]) {
        परत असत्य;
    }

    जर (src0->ne[1] != src1->ne[1]) {
        परत असत्य;
    }

    जर (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        परत असत्य;
    }

    जर (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        परत असत्य;
    }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_tri(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32) { परत असत्य; }
    जर (dst->type  != GGML_TYPE_F32) { परत असत्य; }
    जर (!ggml_are_same_shape(src0, dst)) { परत असत्य; }
    जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { परत असत्य; }

    परत सत्य;

    GGML_UNUSED(sess);
}

स्थिर स्थिरांक अक्षर * ggml_backend_hexagon_name(ggml_backend_t backend) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    परत sess->c_name();
}

स्थिर शून्य ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

स्थिर htp_op_code op_remap_to_htp(स्थिरांक ggml_tensor * t) {
    निवड (t->op) {
        स्थिती GGML_OP_FLASH_ATTN_EXT:  परत HTP_OP_FLASH_ATTN_EXT;
        स्थिती GGML_OP_MUL_MAT:
            जर (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                परत HTP_OP_HADAMARD;
            }
            परत HTP_OP_MUL_MAT;
        स्थिती GGML_OP_MUL_MAT_ID:      परत HTP_OP_MUL_MAT_ID;
        स्थिती GGML_OP_MUL:             परत HTP_OP_MUL;
        स्थिती GGML_OP_ADD:             परत HTP_OP_ADD;
        स्थिती GGML_OP_ADD_ID:          परत HTP_OP_ADD_ID;
        स्थिती GGML_OP_SUB:             परत HTP_OP_SUB;
        स्थिती GGML_OP_DIV:             परत HTP_OP_DIV;
        स्थिती GGML_OP_DUP:
        स्थिती GGML_OP_CPY:             परत HTP_OP_CPY;
        स्थिती GGML_OP_CONT:            परत HTP_OP_CPY;
        स्थिती GGML_OP_GET_ROWS:        परत HTP_OP_GET_ROWS;
        स्थिती GGML_OP_SET_ROWS:        परत HTP_OP_SET_ROWS;
        स्थिती GGML_OP_SUM:             परत HTP_OP_SUM;
        स्थिती GGML_OP_SUM_ROWS:        परत HTP_OP_SUM_ROWS;
        स्थिती GGML_OP_POOL_2D:         परत HTP_OP_POOL_2D;
        स्थिती GGML_OP_POOL_1D:         परत HTP_OP_POOL_1D;
        स्थिती GGML_OP_ARGSORT:         परत HTP_OP_ARGSORT;
        स्थिती GGML_OP_शीर्ष_K:           परत HTP_OP_शीर्ष_K;
        स्थिती GGML_OP_ARGMAX:          परत HTP_OP_ARGMAX;
        स्थिती GGML_OP_NORM:            परत HTP_OP_NORM;
        स्थिती GGML_OP_L2_NORM:         परत HTP_OP_L2_NORM;
        स्थिती GGML_OP_RMS_NORM:        परत HTP_OP_RMS_NORM;
        स्थिती GGML_OP_CONCAT:          परत HTP_OP_CONCAT;
        स्थिती GGML_OP_SCALE:           परत HTP_OP_SCALE;
        स्थिती GGML_OP_CLAMP:           परत HTP_OP_CLAMP;
        स्थिती GGML_OP_LEAKY_RELU:      परत HTP_OP_LEAKY_RELU;
        स्थिती GGML_OP_SQR:             परत HTP_OP_SQR;
        स्थिती GGML_OP_SQRT:            परत HTP_OP_SQRT;
        स्थिती GGML_OP_LOG:             परत HTP_OP_UNARY_LOG;
        स्थिती GGML_OP_SOFT_MAX:        परत HTP_OP_SOFTMAX;
        स्थिती GGML_OP_SSM_CONV:        परत HTP_OP_SSM_CONV;
        स्थिती GGML_OP_GATED_DELTA_NET: परत HTP_OP_GATED_DELTA_NET;
        स्थिती GGML_OP_ROPE:            परत HTP_OP_ROPE;
        स्थिती GGML_OP_REPEAT:          परत HTP_OP_REPEAT;
        स्थिती GGML_OP_CUMSUM:          परत HTP_OP_CUMSUM;
        स्थिती GGML_OP_FILL:            परत HTP_OP_FILL;
        स्थिती GGML_OP_DIAG:            परत HTP_OP_DIAG;
        स्थिती GGML_OP_SOLVE_TRI:       परत HTP_OP_SOLVE_TRI;
        स्थिती GGML_OP_TRI:             परत HTP_OP_TRI;
        स्थिती GGML_OP_PAD:             परत HTP_OP_PAD;
        स्थिती GGML_OP_IM2COL:          परत HTP_OP_IM2COL;
        स्थिती GGML_OP_IM2COL_3D:       परत HTP_OP_IM2COL_3D;
        स्थिती GGML_OP_ROLL:            परत HTP_OP_ROLL;

        स्थिती GGML_OP_UNARY:
            निवड (ggml_get_unary_op(t)) {
                स्थिती GGML_UNARY_OP_SILU:       परत HTP_OP_UNARY_SILU;
                स्थिती GGML_UNARY_OP_GELU:       परत HTP_OP_UNARY_GELU;
                स्थिती GGML_UNARY_OP_GELU_QUICK: परत HTP_OP_UNARY_GELU_QUICK;
                स्थिती GGML_UNARY_OP_GELU_ERF:   परत HTP_OP_UNARY_GELU_ERF;
                स्थिती GGML_UNARY_OP_SIGMOID:    परत HTP_OP_UNARY_SIGMOID;
                स्थिती GGML_UNARY_OP_NEG:        परत HTP_OP_UNARY_NEG;
                स्थिती GGML_UNARY_OP_EXP:        परत HTP_OP_UNARY_EXP;
                स्थिती GGML_UNARY_OP_SOFTPLUS:   परत HTP_OP_UNARY_SOFTPLUS;
                स्थिती GGML_UNARY_OP_TANH:       परत HTP_OP_UNARY_TANH;
                स्थिती GGML_UNARY_OP_ABS:        परत HTP_OP_UNARY_ABS;
                स्थिती GGML_UNARY_OP_RELU:       परत HTP_OP_UNARY_RELU;
                स्थिती GGML_UNARY_OP_STEP:       परत HTP_OP_UNARY_STEP;
            पूर्वनिर्धारित:
                थांबा;
            }
            थांबा;

        स्थिती GGML_OP_GLU:
            निवड (ggml_get_glu_op(t)) {
                स्थिती GGML_GLU_OP_SWIGLU:       परत HTP_OP_GLU_SWIGLU;
                स्थिती GGML_GLU_OP_SWIGLU_OAI:   परत HTP_OP_GLU_SWIGLU_OAI;
                स्थिती GGML_GLU_OP_SWIGLU_CLAMP: परत HTP_OP_GLU_SWIGLU_CLAMP;
                स्थिती GGML_GLU_OP_GEGLU:        परत HTP_OP_GLU_GEGLU;
                स्थिती GGML_GLU_OP_GEGLU_QUICK:  परत HTP_OP_GLU_GEGLU_QUICK;
                स्थिती GGML_GLU_OP_GEGLU_ERF:    परत HTP_OP_GLU_GEGLU_ERF;
                पूर्वनिर्धारित: थांबा;
            }
            थांबा;

        पूर्वनिर्धारित:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    परत HTP_OP_INVALID;
}

स्थिर ओळीत तार्किक op_is_compute(ggml_tensor *गाठ)
{
    परत !ggml_op_is_empty(गाठ->op) && !ggml_is_empty(गाठ) && (गाठ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

स्थिर तार्किक mm_is_hmx_eligible(स्थिरांक ggml_tensor * t) {
    जर (opt_nhmx == 0) { परत असत्य; }

    स्थिरांक ggml_tensor * src0 = t->src[0];
    स्थिरांक ggml_tensor * src1 = t->src[1];

    जर (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        स्थिरांक ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        परत ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, असत्य, असत्य);
    }

    स्थिरांक तार्किक is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    स्थिरांक तार्किक is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    परत ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

स्थिर तार्किक is_supported_mul_mat_nx_kernel(स्थिरांक ggml_tensor * src0, स्थिरांक संरचना htp_mm_kernel_params * kparams) {
    जर (kparams->n_hmx) {
        परत kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    जर (!ggml_hexagon_tensor_is_tiled(src0)) {
        परत असत्य;
    }

    परत kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

स्थिर तार्किक is_supported_mul_mat_id_nx_kernel(स्थिरांक ggml_tensor * src0, स्थिरांक संरचना htp_mm_kernel_params * kparams) {
    जर (kparams->n_hmx) {
        परत kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    जर (!ggml_hexagon_tensor_is_tiled(src0)) {
        परत असत्य;
    }

    परत kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

स्थिर तार्किक is_mergeable_mul_mat(स्थिरांक ggml_tensor * t) {
    जर (t->op != GGML_OP_MUL_MAT) परत असत्य;
    जर (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) परत असत्य;

    स्थिरांक ggml_tensor * src0 = t->src[0];
    स्थिरांक ggml_tensor * src1 = t->src[1];
    जर (src0->ne[2] != 1 || src0->ne[3] != 1) परत असत्य;

    जर (mm_is_hmx_eligible(t)) {
        परत htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    जर (src1->type != GGML_TYPE_F32) परत असत्य;

    परत ggml_hexagon_tensor_is_tiled(src0);
}

स्थिर तार्किक is_mergeable_mul_mat_pair(स्थिरांक ggml_tensor * n1, स्थिरांक ggml_tensor * n2) {
    जर (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        परत असत्य;
    }
    जर (n1->src[1] != n2->src[1]) {
        परत असत्य;
    }
    जर (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        परत असत्य;
    }
    जर (n1->src[0]->type != n2->src[0]->type) {
        परत असत्य;
    }
    जर (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        परत असत्य;
    }
    जर (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        परत असत्य;
    }
    परत सत्य;
}

स्थिर तार्किक is_mergeable_mul_mat_id(स्थिरांक ggml_tensor * t) {
    जर (t->op != GGML_OP_MUL_MAT_ID) परत असत्य;

    स्थिरांक ggml_tensor * src0 = t->src[0];
    जर (mm_is_hmx_eligible(t)) {
        परत htp_is_hmx_weight_type(src0->type);
    }

    परत ggml_hexagon_tensor_is_tiled(src0);
}

स्थिर तार्किक is_mergeable_mul_mat_id_pair(स्थिरांक ggml_tensor * n1, स्थिरांक ggml_tensor * n2) {
    जर (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        परत असत्य;
    }
    जर (n1->src[1] != n2->src[1]) {
        परत असत्य;
    }
    जर (n1->src[2] != n2->src[2]) {
        परत असत्य;
    }
    जर (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        परत असत्य;
    }
    जर (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        परत असत्य;
    }
    जर (n1->src[0]->type != n2->src[0]->type) {
        परत असत्य;
    }
    जर (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        परत असत्य;
    }
    जर (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        परत असत्य;
    }
    परत सत्य;
}

स्थिर ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);

    जर (sess->last_error > HTP_STATUS_OK) {
        परत GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    स्थिरांक std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    तार्किक cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    जर (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } अन्यथा {
        // Tag fusable tensors in graph
        साठी (पूर्णांक i = 0; i < graph->n_nodes; i++) {
            स्वतः * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            जर (!extra) सुरूठेवा;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            जर (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } अन्यथा जर ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                जर ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_गाठ_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        साठी (पूर्णांक i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            जर (!op_is_compute(n)) {
                सुरूठेवा;
            }

            htp_opnode गाठ(HTP_OP_INVALID, n);
            गाठ.opcode = op_remap_to_htp(n);
            जर (गाठ.opcode == HTP_OP_MUL_MAT || गाठ.opcode == HTP_OP_MUL_MAT_ID || गाठ.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = गाठ.गाठ->src[0];
                जर (ggml_hexagon_needs_repack(src0)) {
                    जर (opt_verbose) {
                        अक्षर dims_buf[64];
                        अक्षर strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, आकार(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, आकार(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    स_नकल(raw.data(), src0->data, raw.size());
                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
                    स्वतः * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            जर (गाठ.opcode == HTP_OP_MUL_MAT || गाठ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.गाठ,
                    (संरचना htp_mm_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_MUL || गाठ.opcode == HTP_OP_ADD || गाठ.opcode == HTP_OP_ADD_ID || गाठ.opcode == HTP_OP_SUB || गाठ.opcode == HTP_OP_DIV) {
                स्थिरांक ggml_tensor * src1 = गाठ.गाठ->src[1];
                स्थिरांक ggml_tensor * src2 = (गाठ.opcode == HTP_OP_ADD_ID) ? गाठ.गाठ->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_द्विमान_params(sess,
                    गाठ.opcode, गाठ.गाठ->src[0], src1, गाठ.गाठ,
                    (संरचना htp_द्विमान_kernel_params *) गाठ.kernel_params, src2));
            } अन्यथा जर (गाठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_fa_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (htp_op_is_unary(गाठ.opcode)) {
                स्वतः inputs = गाठ.get_inputs();
                स्थिरांक संरचना ggml_tensor * src0 = inputs[0];
                स्थिरांक संरचना ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    गाठ.opcode, src0, src1, गाठ.dst(),
                    (संरचना htp_unary_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_get_rows_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_set_rows_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_rope_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    गाठ.गाठ->src[0], गाठ.गाठ->src[1], गाठ.dst(),
                    (संरचना htp_ssm_conv_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_softmax_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_gdn_kernel_params *)गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_ARGSORT || गाठ.opcode == HTP_OP_शीर्ष_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    गाठ.गाठ,
                    गाठ.opcode == HTP_OP_शीर्ष_K,
                    (संरचना htp_sort_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_POOL_2D || गाठ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, गाठ.गाठ->src[0], गाठ.dst(),
                    (संरचना htp_pool_2d_kernel_params *)गाठ.kernel_params,
                    गाठ.opcode == HTP_OP_POOL_1D);
            } अन्यथा जर (गाठ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_concat_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_CPY || गाठ.opcode == HTP_OP_CPY_FENCE) {
                स्थिरांक तार्किक ok = ggml_hexagon_precompute_cpy_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_copy_kernel_params *) गाठ.kernel_params
                );
                स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_copy_kernel_params *) गाठ.kernel_params;
                जर (ok && गाठ.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    सुरूठेवा;
                }
            } अन्यथा जर (गाठ.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_im2col_kernel_params *) गाठ.kernel_params
                );
            } अन्यथा जर (गाठ.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    गाठ.गाठ,
                    (संरचना htp_im2col_kernel_params *) गाठ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(गाठ));
        }

        जर (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } अन्यथा {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    साठी (स्थिरांक स्वतः & गाठ : *nodes_ptr) {
        sess->enqueue_op(गाठ);
    }

    जर (sess->last_error > HTP_STATUS_OK) {
        परत GGML_STATUS_FAILED;
    }

    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    जर (sess->last_error > HTP_STATUS_OK) {
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

    शून्य add(स्थिरांक ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    तार्किक check(स्थिरांक ggml_hexagon_mem_range & mr) स्थिरांक {
        साठी (स्थिरांक स्वतः & cmp : ranges) {
            जर (mr.pb != cmp.pb) {
                सुरूठेवा;
            }
            जर (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                सुरूठेवा;
            }
            जर (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                परत असत्य;
            }
        }
        परत सत्य;
    }
};

स्थिर ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(स्थिरांक ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    स्थिरांक ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    जर (tensor->buffer) {
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
    परत mr;
}

स्थिर शून्य ggml_hexagon_mem_ranges_add_गाठ(ggml_hexagon_mem_ranges & mrs, स्थिरांक htp_opnode & गाठ) {
    जर (गाठ.is_empty()) परत;

    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        जर (गाठ.गाठ->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(गाठ.गाठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            जर (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(गाठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

स्थिर तार्किक ggml_hexagon_mem_ranges_check_गाठ(स्थिरांक ggml_hexagon_mem_ranges & mrs, स्थिरांक htp_opnode & गाठ) {
    जर (गाठ.is_empty()) परत सत्य;

    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        जर (गाठ.गाठ->src[i]) {
            जर (!mrs.check(ggml_hexagon_mem_range_from_tensor(गाठ.गाठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                परत असत्य;
            }
        }
    }
    साठी (स्थिरांक स्वतः * fused : गाठ.fused) {
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            जर (fused->src[i]) {
                जर (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    परत असत्य;
                }
            }
        }
    }
    परत mrs.check(ggml_hexagon_mem_range_from_tensor(गाठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

स्थिर std::vector<पूर्णांक> ggml_hexagon_graph_optimize_reorder(स्थिरांक std::vector<htp_opnode> & nodes) {
    स्थिरांक पूर्णांक n = nodes.size();

    std::vector<पूर्णांक> res;
    res.reserve(n);

    std::vector<तार्किक> used(n, असत्य);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    साठी (पूर्णांक i0 = 0; i0 < n; i0++) {
        जर (used[i0]) {
            सुरूठेवा;
        }

        स्थिरांक स्वतः & node0 = nodes[i0];

        जर (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = सत्य;
            सुरूठेवा;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr पूर्णांक N_FORWARD = 16;

        std::vector<पूर्णांक> चळत;
        चळत.push_back(i0);

        mrs.reset();

        साठी (पूर्णांक i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            जर (used[i1]) {
                सुरूठेवा;
            }

            स्थिरांक स्वतः & node1 = nodes[i1];

            जर (node1.stackable() && node1.same_आदान(node0) && ggml_hexagon_mem_ranges_check_गाठ(mrs, node1)) {
                चळत.push_back(i1);
            } अन्यथा {
                ggml_hexagon_mem_ranges_add_गाठ(mrs, node1);
            }
        }

        साठी (पूर्णांक idx : चळत) {
            res.push_back(idx);
            used[idx] = सत्य;
        }
    }

    परत res;
}

स्थिर शून्य ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    स्थिरांक पूर्णांक n = gf->n_nodes;

    constexpr पूर्णांक MAX_FUSE = 16;

    गणना ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    साठी (पूर्णांक i = 0; i < n; i++) {
        htp_opnode गाठ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        जर (गाठ.op() == GGML_OP_ADD ||
            गाठ.op() == GGML_OP_NORM ||
            गाठ.op() == GGML_OP_RMS_NORM) {
            ops[0] = गाठ.op();

            पूर्णांक f = i + 1;
            पर्यंत (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                जर (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    थांबा;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            साठी (; f > 1; f--) {
                जर (ggml_can_fuse(gf, i, ops, f)) {
                    थांबा;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            साठी (पूर्णांक k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                गाठ.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(गाठ));
    }

    स्थिरांक स्वतः order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        पूर्णांक j = 0;
        साठी (स्थिरांक स्वतः i : order) {
            स्थिरांक स्वतः & गाठ = nodes[i];

            gf->nodes[j++] = गाठ.गाठ;

            साठी (स्वतः * fused : गाठ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

स्थिर uint64_t ggml_hexagon_session_किल्ली(स्थिरांक ggml_hexagon_session * sess) {
    परत ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

स्थिर तार्किक ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिरांक ggml_tensor * src, ggml_tensor * dst) {
    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    स्वतः sbuf_dst = sess_dst->mmap_tensor(dst);

    जर (!sess_src->clone_buffer(sbuf_dst)) { परत असत्य; }

    स्थिरांक uint64_t src_किल्ली = ggml_hexagon_session_किल्ली(sess_src);
    स्वतः & fence_slot = sess_dst->cpy_fence_slots[src_किल्ली];
    जर (!fence_slot) {
        fence_slot = (अस्थिर uint32_t *) sess_dst->alloc_fence(1);
    }

    जर (!sess_src->clone_buffer(sess_dst->fence_buf)) { परत असत्य; }

    जर (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
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
    fence_tensor.nb[0]  = आकार(int32_t);
    fence_tensor.nb[1]  = आकार(int32_t);
    fence_tensor.nb[2]  = आकार(int32_t);
    fence_tensor.nb[3]  = आकार(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ सत्य);

    sess_dst->add_peer(sess_src);

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिरांक ggml_tensor * src, ggml_tensor * dst) {
    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    स्वतः sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    जर (!sess_dst->clone_buffer(sbuf_src)) { परत असत्य; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    परत सत्य;
}

स्थिर तार्किक ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिरांक ggml_tensor * src, ggml_tensor * dst) {
    जर (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        परत असत्य;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    जर (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    स्वतः * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    स्थिरांक स्वतः * src_extra = static_cast<स्थिरांक ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    जर (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        परत सत्य;
    }

    जर (sess_src->phys_idx != sess_dst->phys_idx)
        परत ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    परत ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

स्थिर ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = नवीन ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (अस्थिर uint32_t *) sess->alloc_fence(1);

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (शून्य *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = आकार(int32_t);
    hex_event->fence_tensor.nb[1]  = आकार(int32_t);
    hex_event->fence_tensor.nb[2]  = आकार(int32_t);
    hex_event->fence_tensor.nb[3]  = आकार(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (शून्य *)hex_event, (शून्य *)hex_event->fence_slot);

    परत नवीन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

स्थिर शून्य ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    जर (hex_event->seq == 0) {
        परत;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);

    स्वतः * fence = reinterpret_cast<स्थिरांक अस्थिर std::atomic<uint32_t> *>(hex_event->fence_slot);

    जर ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    पर्यंत (सत्य) {
        जर ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            जर (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            थांबा;
        }
        std::this_thread::yield();
    }
}

स्थिर शून्य ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    स्वतः * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (शून्य *)hex_event);
    hex_event->fence_sess->free_fence((शून्य *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

स्थिर शून्य ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    स्वतः * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

स्थिर शून्य ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    स्वतः hex_event = static_cast<ggml_hexagon_event *>(event->context);

    जर (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ असत्य);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);
}

स्थिर शून्य ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    स्वतः hex_event = static_cast<ggml_hexagon_event *>(event->context);

    जर (hex_event->seq == 0) {
        परत;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (शून्य *)hex_event, hex_event->seq, (शून्य *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    जर (sess->phys_idx == hex_event->sess->phys_idx) {
        जर (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        परत;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ सत्य);
}

स्थिर शून्य ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, संरचना ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (पूर्णांक) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, स्थिरांक संरचना ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (पूर्णांक) tensor->buffer->usage : -1);
    sess->flush_sync();
    जर (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     संरचना ggml_tensor * tensor,
                                                     स्थिरांक शून्य * data,
                                                     माप_प्रकार offset,
                                                     माप_प्रकार size,
                                                     माप_प्रकार n_copies,
                                                     माप_प्रकार stride_tensor,
                                                     माप_प्रकार stride_data) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (पूर्णांक) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

स्थिर शून्य ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     स्थिरांक संरचना ggml_tensor * tensor,
                                                     शून्य * data,
                                                     माप_प्रकार offset,
                                                     माप_प्रकार size,
                                                     माप_प्रकार n_copies,
                                                     माप_प्रकार stride_tensor,
                                                     माप_प्रकार stride_data) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (पूर्णांक) tensor->buffer->usage : -1);
    sess->flush_sync();
    जर (sess->last_error > HTP_STATUS_OK) {
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
    परत &guid;
}

तार्किक ggml_backend_is_hexagon(ggml_backend_t backend) {
    परत backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

स्थिर ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, स्थिरांक अक्षर * params) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    परत नवीन ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

स्थिर स्थिरांक अक्षर * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    परत dev_ctx->c_name();

    GGML_UNUSED(dev);
}

स्थिर स्थिरांक अक्षर * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    परत "Hexagon";
    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    *मुक्त  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    परत GGML_BACKEND_DEVICE_TYPE_GPU;

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
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    परत &dev_ctx->buffer_type;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    जर (!opt_hostbuf) {
        परत रिक्त;
    }
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    परत &dev_ctx->host_buffer_type;
}

स्थिर तार्किक ggml_hexagon_supported_cpy(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    संरचना htp_copy_kernel_params kparams;
    परत ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_cont(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    संरचना htp_copy_kernel_params kparams;
    परत ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_repeat(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    GGML_UNUSED(sess);
    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    // Support f32 and f16
    जर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) परत असत्य;

    // src and dst must be the same type
    जर (src0->type != dst->type) परत असत्य;

    // dst dims must be multiples of src dims
    जर (dst->ne[0] % src0->ne[0] != 0) परत असत्य;
    जर (dst->ne[1] % src0->ne[1] != 0) परत असत्य;
    जर (dst->ne[2] % src0->ne[2] != 0) परत असत्य;
    जर (dst->ne[3] % src0->ne[3] != 0) परत असत्य;

    // require contiguous tensors (no transposition)
    जर (ggml_is_transposed(src0) || ggml_is_transposed(dst)) परत असत्य;

    परत सत्य;
}

स्थिर तार्किक ggml_hexagon_supported_concat(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    संरचना htp_concat_kernel_params kparams;
    परत ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

स्थिर तार्किक ggml_hexagon_supported_fill(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    स्थिरांक संरचना ggml_tensor * dst = op;

    जर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        परत असत्य;
    }

    परत सत्य;
    GGML_UNUSED(sess);
}

स्थिर तार्किक ggml_hexagon_supported_roll(स्थिरांक संरचना ggml_hexagon_session * sess, स्थिरांक संरचना ggml_tensor * op) {
    GGML_UNUSED(sess);

    स्थिरांक संरचना ggml_tensor * src0 = op->src[0];
    स्थिरांक संरचना ggml_tensor * dst  = op;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    जर (!ggml_are_same_shape(src0, dst)) {
        परत असत्य;
    }

    जर (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    परत सत्य;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
स्थिर तार्किक ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, स्थिरांक संरचना ggml_tensor * op) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    // reject ops that match the filter
    जर (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        परत असत्य;
    }

    तार्किक supp = असत्य;
    निवड (op->op) {
        स्थिती GGML_OP_NONE:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
        स्थिती GGML_OP_TRANSPOSE:
            supp = सत्य;
            थांबा;

        स्थिती GGML_OP_MUL:
        स्थिती GGML_OP_ADD:
        स्थिती GGML_OP_SUB:
        स्थिती GGML_OP_DIV:
            supp = ggml_hexagon_supported_द्विमान(sess, op);
            थांबा;

        स्थिती GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            थांबा;

        स्थिती GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            थांबा;

        स्थिती GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            थांबा;

        स्थिती GGML_OP_NORM:
        स्थिती GGML_OP_L2_NORM:
        स्थिती GGML_OP_RMS_NORM:
        स्थिती GGML_OP_SCALE:
        स्थिती GGML_OP_CLAMP:
        स्थिती GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            थांबा;

        स्थिती GGML_OP_SQR:
        स्थिती GGML_OP_SQRT:
        स्थिती GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            थांबा;

        स्थिती GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            थांबा;

        स्थिती GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            थांबा;

        स्थिती GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            थांबा;

        स्थिती GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            थांबा;

        स्थिती GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            थांबा;

        स्थिती GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            थांबा;

        स्थिती GGML_OP_UNARY:
            निवड (ggml_get_unary_op(op)) {
                स्थिती GGML_UNARY_OP_NEG:
                स्थिती GGML_UNARY_OP_EXP:
                स्थिती GGML_UNARY_OP_SIGMOID:
                स्थिती GGML_UNARY_OP_SOFTPLUS:
                स्थिती GGML_UNARY_OP_TANH:
                स्थिती GGML_UNARY_OP_ABS:
                स्थिती GGML_UNARY_OP_SILU:
                स्थिती GGML_UNARY_OP_GELU:
                स्थिती GGML_UNARY_OP_GELU_QUICK:
                स्थिती GGML_UNARY_OP_GELU_ERF:
                स्थिती GGML_UNARY_OP_RELU:
                स्थिती GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    थांबा;
                पूर्वनिर्धारित:
                    supp = असत्य;
                    थांबा;
            }
            थांबा;

        स्थिती GGML_OP_GLU:
            निवड (ggml_get_glu_op(op)) {
                स्थिती GGML_GLU_OP_SWIGLU:
                स्थिती GGML_GLU_OP_SWIGLU_OAI:
                स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
                स्थिती GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    थांबा;
                स्थिती GGML_GLU_OP_GEGLU_QUICK:
                स्थिती GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    थांबा;
                पूर्वनिर्धारित:
                    supp = असत्य;
                    थांबा;
            }
            थांबा;

        स्थिती GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            थांबा;

        स्थिती GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            थांबा;

        स्थिती GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            थांबा;

        स्थिती GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            थांबा;

        स्थिती GGML_OP_DUP:
        स्थिती GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            थांबा;

        स्थिती GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            थांबा;

        स्थिती GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            थांबा;

        स्थिती GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            थांबा;

        स्थिती GGML_OP_शीर्ष_K:
            supp = ggml_hexagon_supported_शीर्ष_k(sess, op);
            थांबा;

        स्थिती GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            थांबा;

        स्थिती GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            थांबा;

        स्थिती GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            थांबा;

        स्थिती GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            थांबा;

        स्थिती GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            थांबा;

        स्थिती GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            थांबा;

        स्थिती GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            थांबा;

        स्थिती GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            थांबा;

        स्थिती GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            थांबा;

        स्थिती GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            थांबा;

        स्थिती GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            थांबा;

        स्थिती GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            थांबा;

        पूर्वनिर्धारित:
            थांबा;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    परत supp;
}

स्थिर तार्किक ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    तार्किक supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    परत supp;
}

स्थिर स्थिरांक संरचना ggml_backend_device_i ggml_backend_hexagon_device_i = {
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
    साठी (माप_प्रकार i = 0; i < opt_ndev; i++) {
        स्थिरांक स्वतः & cfg = opt_device_configs[i];
        जर (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } अन्यथा {
            std::string peers_str;
            साठी (स्थिरांक स्वतः & p : cfg.mdev_group) {
                जर (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = नवीन ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    साठी (माप_प्रकार i = 0; i < opt_ndev; i++) {
        स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

स्थिर स्थिरांक अक्षर * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    परत "HTP";
    GGML_UNUSED(reg);
}

स्थिर माप_प्रकार ggml_backend_hexagon_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    परत opt_ndev;
    GGML_UNUSED(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार index) {
    स्वतः hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    जर (index >= opt_ndev || !hreg->devices[index].context) {
        परत nullptr;
    }

    परत &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

स्थिर शून्य * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) {
    जर (n_backends < 2 || n_backends > 4) {
        परत nullptr;
    }

    साठी (माप_प्रकार i = 0; i < n_backends; ++i) {
        जर (!ggml_backend_is_hexagon(backends[i])) {
            परत nullptr;
        }
    }

    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        साठी (माप_प्रकार j = i + 1; j < n_backends; j++) {
            स्वतः sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            जर (sess_i->phys_idx == sess_j->phys_idx) {
                परत nullptr;
            }
        }
    }

    स्वतः * ctx = नवीन ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    स्थिर ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (अस्थिर uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (शून्य *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = आकार(int32_t);
        ctx->fence_tensors[i].nb[1]  = आकार(int32_t);
        ctx->fence_tensors[i].nb[2]  = आकार(int32_t);
        ctx->fence_tensors[i].nb[3]  = आकार(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    परत ctx;
}

स्थिर शून्य ggml_backend_hexagon_comm_free(शून्य * comm_ctx_v) {
    जर (!comm_ctx_v) परत;
    स्वतः * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    साठी (माप_प्रकार i = 0; i < ctx->n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((शून्य *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

स्थिर तार्किक ggml_backend_hexagon_comm_allreduce_tensor(शून्य * comm_ctx_v, संरचना ggml_tensor ** tensors) {
    जर (opt_ar_select == 0 || !comm_ctx_v) परत असत्य;
    स्वतः * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    स्थिरांक माप_प्रकार n_backends = comm_ctx->n_backends;

    जर (n_backends < 2 || n_backends > 4) परत असत्य;

    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        साठी (माप_प्रकार j = i + 1; j < n_backends; j++) {
            स्वतः sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            जर (sess_i->phys_idx == sess_j->phys_idx) {
                परत असत्य;
            }
        }
    }

    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        जर (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            परत असत्य;
        }
        जर (tensors[i]->type != tensors[0]->type) {
            परत असत्य;
        }
        जर (!ggml_is_contiguous(tensors[i])) {
            परत असत्य;
        }
        जर (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            परत असत्य;
        }
    }

    जर (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    साठी (माप_प्रकार r = 0; r < n_backends; r++) {
        स्वतः sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        संरचना htp_allreduce_kernel_params kparams;
        जर (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, असत्य, असत्य, /*is_shard_ok=*/ असत्य, &kparams)) {
            परत असत्य;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    साठी (माप_प्रकार i = 1; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        जर ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    जर (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    जर (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<स्थिरांक ggml_tensor *> data_tensors(n_backends);
    std::vector<स्थिरांक ggml_tensor *> sync_tensors(n_backends);
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    साठी (माप_प्रकार r = 0; r < n_backends; r++) {
        स्वतः sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        साठी (माप_प्रकार j = 0; j < n_backends; j++) {
            जर (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    परत सत्य;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(पूर्णांक main_device, स्थिरांक दशांश * tensor_split) {
    GGML_UNUSED(tensor_split);
    स्वतः reg = ggml_backend_hexagon_reg();
    स्वतः dev = ggml_backend_reg_dev_get(reg, main_device);
    जर (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    जर (!dev) परत nullptr;
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    परत &dev_ctx->buffer_type;
}

स्थिर शून्य * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, स्थिरांक अक्षर * name) {
    GGML_UNUSED(reg);
    जर (म_भेद(name, "ggml_backend_split_buffer_type") == 0) {
        परत (शून्य *) ggml_backend_hexagon_split_buffer_type;
    }
    जर (म_भेद(name, "ggml_backend_comm_init") == 0) {
        परत (शून्य *) ggml_backend_hexagon_comm_init;
    }
    जर (म_भेद(name, "ggml_backend_comm_free") == 0) {
        परत (शून्य *) ggml_backend_hexagon_comm_free;
    }
    जर (म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        परत (शून्य *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    परत रिक्त;
}

ढाँचा<typename T> std::vector<T> str_to_vec(स्थिरांक अक्षर* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    पर्यंत (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    परत v;
}

ढाँचा<typename T, पूर्णांक BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    साठी (स्वतः i : v) { ss << i << ','; }
    स्वतः str = ss.str(); str.pop_back(); // drop last comma
    परत str;
}

स्थिर शून्य ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, तार्किक discovery_supported, स्थिरांक std::unordered_map<पूर्णांक, fastrpc_domain> & cdsp_map) {
    जर (discovery_supported) {
        स्वतः it = cdsp_map.find(cfg.physical_idx);
        जर (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } अन्यथा {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } अन्यथा {
        निवड (cfg.physical_idx) {
            स्थिती 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                थांबा;
            स्थिती 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                थांबा;
            पूर्वनिर्धारित:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                थांबा;
        }
    }
    साठी (स्वतः & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
स्थिर शून्य ggml_hexagon_discover_devices() {
    std::unordered_map<पूर्णांक, fastrpc_domain> cdsp_map;
    तार्किक discovery_supported = असत्य;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    पूर्णांक err = remote_system_request(&domain_info);
    जर (err == AEE_SUCCESS && domain_info.sys.अंक_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.अंक_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (पूर्णांक) domains.size();

        err = remote_system_request(&domain_info);
        जर (err == AEE_SUCCESS) {
            discovery_supported = सत्य;
            स्थिरांक पूर्णांक n_domains = std::min(domain_info.sys.अंक_domains, (पूर्णांक) domains.size());
            साठी (पूर्णांक i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (पूर्णांक) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                जर (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (पूर्णांक) domains[i].type);
                    सुरूठेवा;
                }
                जर (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    सुरूठेवा;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (अचिन्हित) err);
        }
    } अन्यथा जर (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (अचिन्हित) err);
    }

    // Populate domain IDs and names for all configured devices
    साठी (माप_प्रकार i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

स्थिर शून्य ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q4_0 == (अचिन्हित पूर्णांक) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q4_1 == (अचिन्हित पूर्णांक) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q8_0 == (अचिन्हित पूर्णांक) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_MXFP4 == (अचिन्हित पूर्णांक) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_IQ4_NL == (अचिन्हित पूर्णांक) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q4_K == (अचिन्हित पूर्णांक) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q5_K == (अचिन्हित पूर्णांक) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q6_K == (अचिन्हित पूर्णांक) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q3_K == (अचिन्हित पूर्णांक) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q2_K == (अचिन्हित पूर्णांक) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q1_0 == (अचिन्हित पूर्णांक) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_Q2_0 == (अचिन्हित पूर्णांक) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((अचिन्हित पूर्णांक) HTP_TYPE_BF16 == (अचिन्हित पूर्णांक) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    स्थिरांक अक्षर * str_verbose  = दो_पर्या("GGML_HEXAGON_VERBOSE");
    स्थिरांक अक्षर * str_opbatch  = दो_पर्या("GGML_HEXAGON_OPBATCH");
    स्थिरांक अक्षर * str_opqueue  = दो_पर्या("GGML_HEXAGON_OPQUEUE");
    स्थिरांक अक्षर * str_oppoll   = दो_पर्या("GGML_HEXAGON_OPPOLL");
    स्थिरांक अक्षर * str_opfusion = दो_पर्या("GGML_HEXAGON_OPFUSION");
    स्थिरांक अक्षर * str_rt_repack = दो_पर्या("GGML_HEXAGON_RT_REPACK");
    स्थिरांक अक्षर * str_opfilter = दो_पर्या("GGML_HEXAGON_OPFILTER");
    स्थिरांक अक्षर * str_profile  = दो_पर्या("GGML_HEXAGON_PROFILE");
    स्थिरांक अक्षर * str_etm      = दो_पर्या("GGML_HEXAGON_ETM");
    स्थिरांक अक्षर * str_nhvx     = दो_पर्या("GGML_HEXAGON_NHVX");
    स्थिरांक अक्षर * str_nhmx     = दो_पर्या("GGML_HEXAGON_NHMX");
    स्थिरांक अक्षर * str_mm_select = दो_पर्या("GGML_HEXAGON_MM_SELECT");
    स्थिरांक अक्षर * str_fa_select = दो_पर्या("GGML_HEXAGON_FA_SELECT");
    स्थिरांक अक्षर * str_fa_head_split = दो_पर्या("GGML_HEXAGON_FA_HEAD_SPLIT");
    स्थिरांक अक्षर * str_gdn_select = दो_पर्या("GGML_HEXAGON_GDN_SELECT");
    स्थिरांक अक्षर * str_ar_select = दो_पर्या("GGML_HEXAGON_AR_SELECT");
    स्थिरांक अक्षर * str_ar_scatter = दो_पर्या("GGML_HEXAGON_AR_SCATTER");
    स्थिरांक अक्षर * str_ndev     = दो_पर्या("GGML_HEXAGON_NDEV");
    स्थिरांक अक्षर * str_arch     = दो_पर्या("GGML_HEXAGON_ARCH");
    स्थिरांक अक्षर * str_vmem     = दो_पर्या("GGML_HEXAGON_VMEM");
    स्थिरांक अक्षर * str_mbuf     = दो_पर्या("GGML_HEXAGON_MBUF");
    स्थिरांक अक्षर * str_optrace  = दो_पर्या("GGML_HEXAGON_OPTRACE");
    स्थिरांक अक्षर * str_hostbuf  = दो_पर्या("GGML_HEXAGON_HOSTBUF");
    स्थिरांक अक्षर * str_dma64    = दो_पर्या("GGML_HEXAGON_DMA64");
    स्थिरांक अक्षर * str_power_mode = दो_पर्या("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    जर (!str_arch) {
        पूर्णांक err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        जर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } अन्यथा {
            जर (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } अन्यथा जर (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } अन्यथा {
        जर (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = म_से_अदीर्घ(str_arch, रिक्त, 0);
    }

    माप_प्रकार MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || म_से_प(str_dma64) != 0);

    स्वतः RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? नवीन std::regex(str_opfilter, RE_ICASE) : रिक्त;
    opt_verbose   = str_verbose  ? म_से_प(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? म_से_अदीर्घ(str_opbatch, रिक्त, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? म_से_अदीर्घ(str_opqueue, रिक्त, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? म_से_अदीर्घ(str_optrace, रिक्त, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? म_से_अदीर्घ(str_oppoll,  रिक्त, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? म_से_प(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? म_से_प(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? म_से_प(str_profile)                      : 0;
    opt_etm       = str_etm      ? म_से_प(str_etm)                          : 0;
    जर (str_power_mode) {
        पूर्णांक m = htp_power_mode_from_str(str_power_mode);
        जर (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } अन्यथा {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? म_से_अदीर्घ(str_nhvx, रिक्त, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? म_से_प(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? म_से_प(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? म_से_प(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? म_से_प(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? म_से_प(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? म_से_प(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? म_से_प(str_ar_scatter)          : opt_ar_scatter;

    जर (str_mbuf) {
        स्थिरांक अक्षर * p = str_mbuf;
        साठी (पूर्णांक idx = 0; idx < 3 && p && *p; idx++) {
            पर्यंत (*p == ' ') p++;
            स्थिरांक अक्षर * comma = म_अक्षर(p, ',');
            माप_प्रकार len = comma ? (माप_प्रकार)(comma - p) : म_माप(p);
            पर्यंत (len > 0 && p[len - 1] == ' ') len--;
            जर (len > 0) {
                std::string token(p, len);
                जर (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                जर (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                जर (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            जर (!comma) थांबा;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? म_से_प(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    स्थिरांक अक्षर * str_devices  = दो_पर्या("GGML_HEXAGON_DEVICES");
    जर (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    जर (str_devices && str_devices[0] != '\0') {
        तार्किक is_single_संख्या = सत्य;
        साठी (पूर्णांक i = 0; str_devices[i] != '\0'; i++) {
            जर (!है_अंक((अचिन्हित अक्षर)str_devices[i])) {
                is_single_संख्या = असत्य;
                थांबा;
            }
        }
        जर (is_single_संख्या) {
            पूर्णांक n = म_से_प(str_devices);
            जर (n < 1) n = 1;
            जर (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            साठी (माप_प्रकार i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (पूर्णांक)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } अन्यथा {
            std::string s_devices(str_devices);
            std::vector<std::string> वस्तुसंच;
            std::string curr_item;
            पूर्णांक bracket_depth = 0;
            साठी (अक्षर ch : s_devices) {
                जर (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } अन्यथा जर (ch == ']') {
                    जर (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } अन्यथा जर (ch == ',' && bracket_depth == 0) {
                    माप_प्रकार s = curr_item.find_first_not_of(" \t\r\n");
                    माप_प्रकार e = curr_item.find_last_not_of(" \t\r\n");
                    जर (s != std::string::npos) {
                        वस्तुसंच.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } अन्यथा {
                    curr_item += ch;
                }
            }
            माप_प्रकार s = curr_item.find_first_not_of(" \t\r\n");
            माप_प्रकार e = curr_item.find_last_not_of(" \t\r\n");
            जर (s != std::string::npos) {
                वस्तुसंच.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            साठी (स्थिरांक स्वतः & item : वस्तुसंच) {
                माप_प्रकार b_open  = item.find('[');
                माप_प्रकार b_close = item.rfind(']');

                जर (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    पूर्णांक virt = 0;
                    std::string phys_spec = content;
                    माप_प्रकार colon_pos = content.find(':');
                    जर (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        प्रयत्न {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } पकडा (...) {
                            virt = 0;
                        }
                    } अन्यथा {
                        माप_प्रकार dev_colon = dev_name.find(':');
                        जर (dev_colon != std::string::npos) {
                            प्रयत्न {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } पकडा (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<पूर्णांक> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    पर्यंत (std::getline(pss, p_part, ',')) {
                        माप_प्रकार ps = p_part.find_first_not_of(" \t\r\n");
                        माप_प्रकार pe = p_part.find_last_not_of(" \t\r\n");
                        जर (ps == std::string::npos) सुरूठेवा;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        माप_प्रकार dash_pos = p_part.find('-');
                        जर (dash_pos != std::string::npos) {
                            प्रयत्न {
                                पूर्णांक p_start = std::stoi(p_part.substr(0, dash_pos));
                                पूर्णांक p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                साठी (पूर्णांक p = p_start; p <= p_end; p++) {
                                    जर (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } पकडा (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } अन्यथा {
                            प्रयत्न {
                                पूर्णांक p = std::stoi(p_part);
                                जर (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } पकडा (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    जर (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    जर (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        स्वतः & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        साठी (माप_प्रकार k = 1; k < phys_list.size(); k++) {
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
                } अन्यथा जर (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    माप_प्रकार colon_pos = rest.find(':');
                    पूर्णांक phys = 0;
                    पूर्णांक virt = 0;
                    प्रयत्न {
                        जर (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } अन्यथा {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } पकडा (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        सुरूठेवा;
                    }

                    जर (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
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

#जर defined(__ANDROID__)
    जर (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    जर (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            स्वतः v  = str_to_vec<uint32_t>(str_profile);
            निवड (v.size()) {
                स्थिती 1:  opt_profile = v[0]; परत opt_pmu_evt; // mode with default pmu events
                स्थिती 8:  opt_profile = 2;    परत v;           // mode with custom  pmu events
                पूर्वनिर्धारित: opt_profile = 0;    परत {};          // garbage input
            }}();
        जर (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = नवीन ggml_hexagon_registry(reg);
}

स्थिर स्थिरांक संरचना ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
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
        जर (!initialized) {
            स्वतः nErr = htpdrv_init();
            जर (nErr != AEE_SUCCESS) {
                परत रिक्त;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = सत्य;
    }

    परत &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
