#घोषणा _USE_MATH_DEFINES

#समावेश <assert.h>
#समावेश <inttypes.h>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <string.h>
#समावेश <time.h>

#समावेश <atomic>
#समावेश <memory>
#समावेश <chrono>
#समावेश <mutex>
#समावेश <thread>
#समावेश <cstddef>
#समावेश <stdexcept>
#समावेश <string>
#समावेश <sstream>
#समावेश <iomanip>
#समावेश <unordered_set>
#समावेश <unordered_map>
#समावेश <regex>
#समावेश <queue>
#समावेश <deque>
#समावेश <algorithm>
#समावेश <cmath>
#समावेश <initializer_list>

#अगर_अघोषित M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#पूर्ण_अगर

#अगर_घोषित _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#अन्यथा
#    include <semaphore.h>
#    include <unistd.h>
#पूर्ण_अगर

#आशय clang diagnostic ignored "-Wnested-anon-types"
#आशय clang diagnostic ignored "-Wlanguage-extension-token"
#आशय clang diagnostic ignored "-Wgnu-anonymous-struct"
#आशय clang diagnostic ignored "-Wmicrosoft-enum-value"

#समावेश <AEEStdErr.h>
#समावेश <dspqueue.h>
#समावेश <rpcmem.h>

#घोषणा GGML_COMMON_IMPL_CPP
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-alloc.h"
#समावेश "ggml-common.h"
#समावेश "ggml-hexagon.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-quants.h"
#समावेश "htp-opnode.h"
#समावेश "htp-ops.h"
#समावेश "htp/matmul-ops.h"
#समावेश "htp/binary-ops.h"
#समावेश "htp/flash-attn-ops.h"
#समावेश "htp/unary-ops.h"
#समावेश "htp/get-rows-ops.h"
#समावेश "htp/set-rows-ops.h"
#समावेश "htp/softmax-ops.h"
#समावेश "htp/pool-ops.h"
#समावेश "htp/rope-ops.h"
#समावेश "htp/ssm-conv.h"
#समावेश "htp/gated-delta-net-ops.h"
#समावेश "htp/argsort-ops.h"
#समावेश "htp/concat-ops.h"
#समावेश "htp/cpy-ops.h"
#समावेश "htp/im2col-ops.h"
#समावेश "htp_iface.h"
#समावेश "htp-drv.h"
#समावेश "htp-repack.h"

using intvec  = std::vector<पूर्णांक>;
using uintvec = std::vector<अचिन्हित पूर्णांक>;
using u32vec  = std::vector<uint32_t>;

#घोषणा GGML_HEXAGON_MAX_SESSIONS          16

#घोषणा GGML_HEXAGON_FENCE_SLOT_आकार       128

काष्ठा ggml_hexagon_device_config {
    पूर्णांक         physical_idx = 0;
    पूर्णांक         virtual_idx  = 0;
    पूर्णांक         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

अटल ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

अटल पूर्णांक    opt_arch    = 0; // autodetect
अटल माप_प्रकार opt_ndev    = 1;
अटल माप_प्रकार opt_nhvx    = 0; // use all
अटल पूर्णांक    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
अटल माप_प्रकार opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
अटल पूर्णांक    opt_etm     = 0;
अटल पूर्णांक    opt_verbose = 0;
अटल पूर्णांक    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
अटल bool   opt_hostbuf = false;
अटल bool   opt_dma64   = false;
अटल पूर्णांक    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

अटल स्थिर std::pair<स्थिर अक्षर *, पूर्णांक> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_निचला_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_ऊपरी_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_निचला_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_ऊपरी_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_ऊपरी_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

अटल पूर्णांक htp_power_mode_from_str(स्थिर अक्षर * s) {
    क्रम (स्वतः & e : htp_power_mode_names) {
#अगर defined(_MSC_VER)
        अगर (_stricmp(s, e.first) == 0) {
#अन्यथा
        अगर (strcasecmp(s, e.first) == 0) {
#पूर्ण_अगर
            वापस e.second;
        }
    }
    वापस -1;
}

अटल स्थिर अक्षर * htp_power_mode_to_str(पूर्णांक mode) {
    क्रम (स्वतः & e : htp_power_mode_names) {
        अगर (e.second == mode) {
            वापस e.first;
        }
    }
    वापस "unknown";
}

अटल माप_प्रकार opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
अटल माप_प्रकार opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
अटल माप_प्रकार opt_mbuf_योग  = 0;                        // total buffer space limit (0 = unconstrained)

अटल पूर्णांक    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
अटल पूर्णांक    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
अटल पूर्णांक    opt_fa_शीर्ष_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
अटल पूर्णांक    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
अटल पूर्णांक    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
अटल पूर्णांक    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
अटल u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

अटल पूर्णांक opt_opbatch  = 1280; // max number of ops in a batch
अटल पूर्णांक opt_opqueue  = 8;   // max number of pending batches
अटल पूर्णांक opt_optrace  = 0;    // trace buffer size per thread (0 means default)
अटल पूर्णांक opt_oppoll   = 0;    // polling for batch completions
अटल पूर्णांक opt_opfusion  = 1;    // enable/disable op fusion
अटल पूर्णांक opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

क्रमागत ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

अटल अंतरभूत bool ggml_hexagon_is_fusion_enabled(पूर्णांक flag) {
    अगर (opt_opfusion <= 0) वापस false;
    अगर (opt_opfusion == 1) वापस true; // 1 enables all
    वापस (opt_opfusion & flag) != 0;
}

अटल std::regex* opt_opfilter = शून्य; // regex of ops to not claim

#घोषणा HEX_VERBOSE(...) \
    अगर (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

अटल स्थिर अक्षर * status_to_str(uint32_t status) {
    चयन (status) {
        हाल HTP_STATUS_OK:
            वापस "OK";
        हाल HTP_STATUS_NO_SUPPORT:
            वापस "NO-SUPPORT";
        हाल HTP_STATUS_INVAL_PARAMS:
            वापस "INVAL-PARAMS";
        हाल HTP_STATUS_VTCM_TOO_SMALL:
            वापस "VTCM-TOO-SMALL";
        हाल HTP_STATUS_INTERNAL_ERR:
            वापस "INTERNAL-ERROR";
        शेष:
            वापस "UNKNOWN";
    }
}

// ** debug helpers

अटल व्योम ggml_hexagon_dump_op_exec(स्थिर std::string &sess_name, स्थिर htp_opnode & गाँठ, स्थिर uint32_t req_flags) {
    अगर (!opt_verbose) वापस;

    htp_opformat fmt(गाँठ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                गाँठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

अटल व्योम ggml_hexagon_dump_op_supp(स्थिर std::string &sess_name, स्थिर काष्ठा ggml_tensor * op, bool supp) {
    अगर (!opt_verbose) वापस;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

अटल स्थिर अक्षर * htp_event_name(uint16_t id) {
    चयन (id) {
        हाल HTP_TRACE_EVT_DMA:            वापस "DMA";
        हाल HTP_TRACE_EVT_HVX_COMP:       वापस "HVX_COMP";
        हाल HTP_TRACE_EVT_HVX_A_QUANT:    वापस "HVX_A_QUANT";
        हाल HTP_TRACE_EVT_HVX_A_PREP:     वापस "HVX_A_PREP";
        हाल HTP_TRACE_EVT_HVX_W_DEQUANT:  वापस "HVX_W_DEQUANT";
        हाल HTP_TRACE_EVT_HVX_W_PREP:     वापस "HVX_W_PREP";
        हाल HTP_TRACE_EVT_HVX_O_PROC:     वापस "HVX_O_PROC";
        हाल HTP_TRACE_EVT_HVX_FA_QK:      वापस "HVX_QK_FA";
        हाल HTP_TRACE_EVT_HVX_FA_SFM:     वापस "HVX_SFM_FA";
        हाल HTP_TRACE_EVT_HVX_FA_Q_PREP:  वापस "HVX_Q_PREP";
        हाल HTP_TRACE_EVT_HVX_FA_K_PREP:  वापस "HVX_K_PREP";
        हाल HTP_TRACE_EVT_HVX_FA_V_PREP:  वापस "HVX_V_PREP";
        हाल HTP_TRACE_EVT_HVX_GDN_PREP:   वापस "HVX_GDN_PREP";
        हाल HTP_TRACE_EVT_HVX_GDN_SOLVE:  वापस "HVX_GDN_SOLVE";
        हाल HTP_TRACE_EVT_HVX_GDN_V_PREP: वापस "HVX_GDN_V_PREP";
        हाल HTP_TRACE_EVT_HVX_GDN_D_PREP: वापस "HVX_GDN_D_PREP";
        हाल HTP_TRACE_EVT_HVX_GDN_OUT:    वापस "HVX_GDN_OUT";
        हाल HTP_TRACE_EVT_HVX_GDN_STATE:  वापस "HVX_GDN_STATE";
        हाल HTP_TRACE_EVT_HVX_GDN_REM:    वापस "HVX_GDN_REM";
        हाल HTP_TRACE_EVT_HMX_COMP:       वापस "HMX_COMP";
        हाल HTP_TRACE_EVT_L2FLUSH:        वापस "L2FLUSH";
        हाल HTP_TRACE_EVT_INIT:           वापस "INIT";
        हाल HTP_TRACE_EVT_BUFF:           वापस "BUFF";
        हाल HTP_TRACE_EVT_FENCE:          वापस "FENCE";
        शेष:                           वापस "UNKNOWN";
    }
}

अटल व्योम ggml_hexagon_dump_op_prof(स्थिर std::string &sess_name, स्थिर htp_opnode & गाँठ, स्थिर htp_prof_desc & pd) {
    अगर (!opt_profile) वापस;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    स्थिर uint32_t * pmu = pd.pmu;

    अक्षर pmu_str[256] = "";
    अगर (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, माप(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(गाँठ);
    भग्न mhz = op_usec > 0 ? (भग्न) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            गाँठ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

अटल व्योम ggml_hexagon_dump_batch_prof(स्थिर std::string & sess_name, स्थिर htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    भग्न batch_mhz = rsp.usecs > 0 ? (भग्न) batch_cycles / rsp.usecs : 0.0f;

    अक्षर evt_str[256] = "----";
    अगर (opt_profile == 3) {
        snprintf(evt_str, माप(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (अचिन्हित दीर्घ दीर्घ) batch_cycles, (अचिन्हित दीर्घ दीर्घ) rsp.cycles_start, batch_mhz);
}

अटल व्योम ggml_hexagon_dump_trace_events(स्थिर std::string & sess_name, स्थिर htp_opbatch_rsp & rsp,
                                           स्थिर htp_trace_desc * trace_events, uint32_t n_traces) {
    अगर (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        क्रम (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        क्रम (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            क्रम (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                स्थिर स्वतः & e = trace_events[t * n_traces + idx];
                bool is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

क्रमागत ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

काष्ठा ggml_hexagon_session;

अटल व्योम ggml_hexagon_precompute_matmul_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_mm_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_fused_matmul_add_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * src2,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_mm_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_unary_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    uint32_t op,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_unary_kernel_params * kparams
);

अटल bool ggml_hexagon_precompute_द्विआधारी_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    uint32_t op,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_द्विआधारी_kernel_params * kparams,
    स्थिर काष्ठा ggml_tensor * src2 = nullptr
);

अटल व्योम ggml_hexagon_precompute_get_rows_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_get_rows_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_set_rows_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_set_rows_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_softmax_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_softmax_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_rope_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_rope_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_ssm_conv_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_ssm_conv_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_gated_delta_net_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_gdn_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_sort_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    bool is_शीर्ष_k,
    काष्ठा htp_sort_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_pool_2d_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_pool_2d_kernel_params * kparams,
    bool is_pool_1d
);

अटल bool ggml_hexagon_precompute_concat_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_concat_kernel_params * kparams
);

अटल bool ggml_hexagon_precompute_cpy_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_copy_kernel_params * kparams
);

अटल bool ggml_hexagon_precompute_im2col_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_im2col_kernel_params * kparams
);

अटल bool ggml_hexagon_precompute_im2col_3d_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_im2col_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_fused_mmnx_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * act,
    int32_t n_weights,
    काष्ठा htp_mm_kernel_params * kparams
);

अटल व्योम ggml_hexagon_precompute_fused_mmidnx_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * act,
    स्थिर काष्ठा ggml_tensor * dst,
    int32_t n_weights,
    काष्ठा htp_mm_kernel_params * kparams
);

अटल bool ggml_hexagon_precompute_allreduce_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    काष्ठा htp_allreduce_kernel_params * kparams
);

अटल bool ggml_hexagon_rows_stride(स्थिर int64_t * ne, स्थिर माप_प्रकार * nb, माप_प्रकार * stride);
अटल bool ggml_hexagon_matmul_can_collapse(स्थिर काष्ठा ggml_tensor * src0, स्थिर काष्ठा ggml_tensor * src1, स्थिर काष्ठा ggml_tensor * dst);
अटल ggml_tensor ggml_hexagon_tensor_collapse_rows(स्थिर काष्ठा ggml_tensor * t);

अटल bool mm_is_hmx_eligible(स्थिर ggml_tensor * t);
अटल htp_op_code op_remap_to_htp(स्थिर ggml_tensor * t);
अटल bool is_supported_mul_mat_nx_kernel(स्थिर ggml_tensor * src0, स्थिर काष्ठा htp_mm_kernel_params * kparams);
अटल bool is_supported_mul_mat_id_nx_kernel(स्थिर ggml_tensor * src0, स्थिर काष्ठा htp_mm_kernel_params * kparams);
अटल bool is_mergeable_mul_mat(स्थिर ggml_tensor * t);
अटल bool is_mergeable_mul_mat_pair(स्थिर ggml_tensor * n1, स्थिर ggml_tensor * n2);
अटल bool is_mergeable_mul_mat_id(स्थिर ggml_tensor * t);
अटल bool is_mergeable_mul_mat_id_pair(स्थिर ggml_tensor * n1, स्थिर ggml_tensor * n2);

// ** backend sessions

काष्ठा ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    माप_प्रकार               shadow_आकार { 0 };
    uint32_t             flags { 0 };
};

अटल अंतरभूत bool ggml_hexagon_tensor_is_fuseable(स्थिर काष्ठा ggml_tensor * t) {
    अगर (!t->extra) वापस false;
    स्वतः extra = (स्थिर काष्ठा ggml_hexagon_tensor_extra *) t->extra;
    वापस (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

अटल अंतरभूत bool ggml_hexagon_can_rt_repack(स्थिर काष्ठा ggml_tensor * t) {
    अगर (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        वापस false;
    }
    अगर (t->बफर && ggml_backend_buft_is_host(t->बफर->buft)) {
        वापस false;
    }
    वापस opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

अटल अंतरभूत bool ggml_hexagon_tensor_is_tiled(स्थिर काष्ठा ggml_tensor * t) {
    स्थिर स्वतः * extra = (स्थिर ggml_hexagon_tensor_extra *) t->extra;
    अगर (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        वापस true;
    }
    वापस ggml_hexagon_can_rt_repack(t);
}

अटल अंतरभूत bool ggml_hexagon_needs_repack(स्थिर काष्ठा ggml_tensor * t) {
    स्थिर स्वतः * extra = (स्थिर ggml_hexagon_tensor_extra *) t->extra;
    अगर (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        वापस false;
    }
    वापस ggml_hexagon_can_rt_repack(t);
}

अटल अंतरभूत bool ggml_hexagon_tensors_overlap(स्थिर काष्ठा ggml_tensor * a, स्थिर काष्ठा ggml_tensor * b) {
    स्थिर uintptr_t a0 = (uintptr_t) a->तथ्य;
    स्थिर uintptr_t b0 = (uintptr_t) b->तथ्य;
    स्थिर uintptr_t a1 = a0 + ggml_nbytes(a);
    स्थिर uintptr_t b1 = b0 + ggml_nbytes(b);

    वापस a0 < b1 && b0 < a1;
}

अटल अंतरभूत bool ggml_hexagon_can_row_partition(स्थिर काष्ठा ggml_tensor * t) {
    अगर (t->ne[1] > 1 && (t->nb[1] & 127) != 0) वापस false;
    अगर (t->ne[2] > 1 && (t->nb[2] & 127) != 0) वापस false;
    अगर (t->ne[3] > 1 && (t->nb[3] & 127) != 0) वापस false;
    वापस true;
}

काष्ठा htp_opnode;

काष्ठा ggml_hexagon_opbatch;
काष्ठा ggml_hexagon_opqueue;
काष्ठा ggml_hexagon_shared_बफर;
काष्ठा ggml_hexagon_fence_बफर;
काष्ठा ggml_hexagon_session;
काष्ठा ggml_backend_hexagon_device_context;

काष्ठा ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

काष्ठा ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    माप_प्रकार                      n_backends = 0;
    अस्थिर uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

काष्ठा ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    अस्थिर uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

काष्ठा ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    पूर्णांक              phys_idx;
    पूर्णांक              virt_idx;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<पूर्णांक, std::unique_ptr<ggml_hexagon_shared_बफर>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_आकार   = 0;
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
    ggml_hexagon_fence_बफर *            fence_buf = nullptr;

    ggml_hexagon_session(स्थिर ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_गिनती = 0) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    स्थिर अक्षर* c_name() स्थिर { वापस name.c_str(); }

    व्योम allocate(स्थिर ggml_hexagon_device_config & config) noexcept(false);
    व्योम release() noexcept(true);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    व्योम      free_fence(व्योम * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, अस्थिर uint32_t *> cpy_fence_slots;

    व्योम enqueue_mdev_group();
    व्योम enqueue_op(स्थिर htp_opnode & गाँठ);
    व्योम enqueue_cpy(स्थिर ggml_tensor * src, ggml_tensor * dst, स्थिर ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    व्योम enqueue_fence(स्थिर ggml_tensor * sync_tensor, uint32_t fence_seq = 0, bool रुको = true);
    व्योम enqueue_allreduce(स्थिर ggml_tensor * dst, स्थिर std::vector<स्थिर ggml_tensor *> & src_tensors,
                           स्थिर std::vector<स्थिर ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    व्योम start_batch();
    व्योम flush_sync(bool all = true);
    व्योम flush_async();
    व्योम flush_batch(माप_प्रकार min_ops = 1);
    व्योम flush_peers();
    व्योम flush_pending(bool all = true);

    ggml_hexagon_shared_बफर * mmap_tensor(स्थिर ggml_tensor * t);
    bool clone_बफर(स्थिर ggml_hexagon_shared_बफर*);
    व्योम release_बफर(स्थिर ggml_hexagon_shared_बफर*);
    व्योम unclone_बफर(स्थिर ggml_hexagon_shared_बफर*);

    व्योम add_peer(ggml_hexagon_session * peer) {
        अगर (यह->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } अन्यथा {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

काष्ठा ggml_backend_hexagon_device_context {
    पूर्णांक                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_बफर_type बफर_type       = {};
    ggml_backend_बफर_type host_बफर_type  = {};
    ggml_backend_बफर_type fence_बफर_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(पूर्णांक dev_id, स्थिर ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    स्थिर अक्षर * c_name() स्थिर { वापस config.name.c_str(); }

    ggml_hexagon_session * session() {
        अगर (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        वापस sess.लो();
    }
};

काष्ठा ggml_backend_hexagon_बफर_type_context {
    ggml_backend_hexagon_बफर_type_context(स्थिर std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        यह->dev_ctx = dev_ctx;
        यह->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

काष्ठा ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    पूर्णांक       fd   = -1;
    माप_प्रकार    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(माप_प्रकार size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        अगर (!base) {
            फेंको std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        अगर (fd < 0) {
            rpcmem_free(base);
            फेंको std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        यह->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        अगर (base) {
            rpcmem_free(base);
        }
    }
};

काष्ठा ggml_hexagon_shared_बफर {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    bool     mapped;
    bool     pinned;
    bool     extended;

    स्थिर अक्षर * c_name() स्थिर { वापस sess->c_name(); }
    uint8_t *    base()   स्थिर { वापस mem ? mem->base : nullptr; }
    माप_प्रकार       size()   स्थिर { वापस mem ? mem->size : 0;  }
    पूर्णांक          fd()     स्थिर { वापस mem ? mem->fd   : -1; }

    व्योम mmap(bool extended = false) {
        अगर (!यह->mem)   वापस;
        अगर (यह->mapped) वापस;

        GGML_ASSERT(!यह->pinned || !extended);

        यह->extended = extended;
        fastrpc_map_flags flags = यह->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        पूर्णांक err = fastrpc_mmap(sess->domain_id, fd(), (व्योम *) base(), 0, size(), flags);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (अचिन्हित) err);
            फेंको std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (व्योम *) base(), size(), fd(), pinned, extended);

        यह->mapped = true;
    }

    व्योम unmap() {
        अगर (!यह->mapped) वापस;

        अगर (!यह->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        अगर (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (व्योम *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (व्योम *) base(), size(), fd());

        यह->mapped = false;
    }

    व्योम alloc(माप_प्रकार size) {
        अगर (यह->mem) वापस;

        यह->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (व्योम *) base(), यह->size(), fd(), (पूर्णांक) pinned);
        अगर (यह->pinned) {
            mmap();
        }
    }

    व्योम मुक्त() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (व्योम *) base(), size(), fd());
        यह->mem  = nullptr;
    }

    ggml_hexagon_shared_बफर(ggml_hexagon_session * sess, माप_प्रकार size, bool pinned = false) {
        यह->sess     = sess;
        यह->mapped   = false;
        यह->pinned   = pinned;
        यह->extended = false;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        माप_प्रकार guard_offset = (size + 4095) & ~4095;
        माप_प्रकार योग_आकार   = guard_offset + 4096;
        अगर (!pinned && opt_dma64) {
            constexpr माप_प्रकार extended_align = 2 * 1024 * 1024;
            योग_आकार = (योग_आकार + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(योग_आकार);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_बफर(ggml_hexagon_session * sess, स्थिर ggml_hexagon_shared_बफर & other) {
        यह->sess   = sess;
        यह->mem    = other.mem;
        यह->mapped = false;
        यह->pinned = other.pinned;
        यह->extended = other.extended;
    }

    ~ggml_hexagon_shared_बफर() {
        मुक्त();
        क्रम (स्वतः * extra : tensor_extra) {
            मिटाओ extra;
        }
    }
};

काष्ठा ggml_hexagon_fence_बफर : खुला ggml_hexagon_shared_बफर {
    uint32_t              slot_गिनती = 0;
    uint32_t              slot_शीर्ष  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_बफर   backend_बफर{};

    ggml_hexagon_fence_बफर(ggml_hexagon_session * sess, ggml_backend_बफर_type_t buft, माप_प्रकार size)
        : ggml_hexagon_shared_बफर(sess, size, false /* pinned */),
          slot_गिनती(size / GGML_HEXAGON_FENCE_SLOT_आकार),
          slot_शीर्ष(0) {
        backend_बफर.buft    = buft;
        backend_बफर.context = static_cast<ggml_hexagon_shared_बफर *>(यह);
        backend_बफर.size    = size;
        mmap(false);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        अगर (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (माप_प्रकार) slot * GGML_HEXAGON_FENCE_SLOT_आकार;
        } अन्यथा अगर (slot_शीर्ष + n_slots <= slot_गिनती) {
            uint32_t slot = slot_शीर्ष;
            slot_शीर्ष += n_slots;
            ptr = base() + (माप_प्रकार) slot * GGML_HEXAGON_FENCE_SLOT_आकार;
        }
        अगर (ptr) {
            स_रखो(ptr, 0, (माप_प्रकार) n_slots * GGML_HEXAGON_FENCE_SLOT_आकार);
        }
        वापस ptr;
    }

    व्योम free_slot(व्योम * ptr, uint32_t n_slots = 1) {
        अगर (!ptr) वापस;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_आकार;
        क्रम (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

अंतरभूत uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    वापस ptr;
}

अंतरभूत व्योम ggml_hexagon_session::free_fence(व्योम * ptr, uint32_t n_slots) {
    अगर (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

अटल व्योम ggml_backend_hexagon_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(बफर->context);
    sbuf->sess->unclone_बफर(sbuf);
    मिटाओ sbuf;
}

अटल व्योम * ggml_backend_hexagon_बफर_get_base(ggml_backend_बफर_t बफर) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(बफर->context);
    वापस sbuf->base();
}

अटल क्रमागत ggml_status ggml_backend_hexagon_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(बफर->context);
    स्वतः sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (व्योम *) sbuf->base(), tensor->तथ्य, ggml_nbytes(tensor));

    स्वतः extra = नया ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_hexagon_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                   ggml_tensor *         tensor,
                                                   स्थिर व्योम *          तथ्य,
                                                   माप_प्रकार                offset,
                                                   माप_प्रकार                size) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_बफर *) बफर->context;
    स्वतः sess  = sbuf->sess;

    अगर (ggml_backend_बफर_get_usage(बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        अगर (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, तथ्य, offset, size, (पूर्णांक) बफर->usage, extra->flags);

    अगर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size);
        वापस;
    }

    अगर (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, तथ्य, size);
        वापस;
    }

    अगर (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    स_नकल(extra->shadow_buf.तथ्य() + offset, तथ्य, size);
    extra->shadow_आकार += size;

    अगर (extra->shadow_आकार >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.तथ्य(), extra->shadow_buf.size());
        extra->shadow_buf.सफा();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_आकार = 0;
    }
}

अटल व्योम ggml_backend_hexagon_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                   स्थिर ggml_tensor *   tensor,
                                                   व्योम *                तथ्य,
                                                   माप_प्रकार                offset,
                                                   माप_प्रकार                size) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_बफर *) बफर->context;
    स्वतः sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, तथ्य, offset, size, (पूर्णांक) बफर->usage, extra->flags);

    अगर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        स_नकल(तथ्य, (स्थिर अक्षर *) tensor->तथ्य + offset, size);
        वापस;
    }

    htp_unpack_tensor_tiled(तथ्य, tensor, offset, size);
}

अटल bool ggml_backend_hexagon_बफर_cpy_tensor(ggml_backend_बफर_t      बफर,
                                                   स्थिर काष्ठा ggml_tensor * src,
                                                   काष्ठा ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    वापस false;

    GGML_UNUSED(बफर);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

अटल व्योम ggml_backend_hexagon_बफर_set_tensor_2d(ggml_backend_बफर_t बफर,
                                                          ggml_tensor *         tensor,
                                                          स्थिर व्योम *          तथ्य,
                                                          माप_प्रकार                offset,
                                                          माप_प्रकार                size,
                                                          माप_प्रकार                n_copies,
                                                          माप_प्रकार                stride_tensor,
                                                          माप_प्रकार                stride_तथ्य) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_बफर *) बफर->context;
    स्वतः sess  = sbuf->sess;

    अगर (ggml_backend_बफर_get_usage(बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        अगर (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य, (पूर्णांक) बफर->usage, extra->flags);

    अगर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((uint8_t *) tensor->तथ्य + offset + i * stride_tensor, (स्थिर uint8_t *) तथ्य + i * stride_तथ्य, size);
        }
        वापस;
    }

    अगर (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल(extra->shadow_buf.तथ्य() + offset + i * stride_tensor, (स्थिर uint8_t *) तथ्य + i * stride_तथ्य, size);
    }
    extra->shadow_आकार += n_copies * size;

    अगर (extra->shadow_आकार >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.तथ्य(), extra->shadow_buf.size());
        extra->shadow_buf.सफा();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_आकार = 0;
    }
}

अटल व्योम ggml_backend_hexagon_बफर_get_tensor_2d(ggml_backend_बफर_t बफर,
                                                      स्थिर ggml_tensor *   tensor,
                                                      व्योम *                तथ्य,
                                                      माप_प्रकार                offset,
                                                      माप_प्रकार                size,
                                                      माप_प्रकार                n_copies,
                                                      माप_प्रकार                stride_tensor,
                                                      माप_प्रकार                stride_तथ्य) {
    स्वतः extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    स्वतः sbuf  = (ggml_hexagon_shared_बफर *) बफर->context;
    स्वतः sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य, (पूर्णांक) बफर->usage);

    अगर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((uint8_t *)तथ्य + i * stride_तथ्य, (स्थिर uint8_t *)tensor->तथ्य + offset + i * stride_tensor, size);
        }
        वापस;
    }

    माप_प्रकार अस्थायी_आकार      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    माप_प्रकार slice_आकार     = tensor->ne[1] * ggml_row_आकार(tensor->type, tensor->ne[0]);
    माप_प्रकार slice_offset   = offset % slice_आकार;
    माप_प्रकार row_आकार_bytes = ggml_row_आकार(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_आकार_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((अस्थायी_आकार % row_आकार_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((offset + अस्थायी_आकार) <= ggml_nbytes(tensor));

    std::vector<uint8_t> अस्थायी_buf(अस्थायी_आकार);
    htp_unpack_tensor_tiled(अस्थायी_buf.तथ्य(), tensor, offset, अस्थायी_आकार);

    क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल((uint8_t *) तथ्य + i * stride_तथ्य, अस्थायी_buf.तथ्य() + i * stride_tensor, size);
    }
}

अटल व्योम ggml_backend_hexagon_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    स्वतः sbuf = (ggml_hexagon_shared_बफर *) बफर->context;
    स्वतः sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (व्योम *) sbuf->base(), sbuf->size());
    स_रखो(sbuf->base(), value, sbuf->size());
}

अटल ggml_backend_बफर_i ggml_backend_hexagon_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_hexagon_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_बफर_init_tensor,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_hexagon_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_बफर_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_बफर_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_बफर_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_बफर_clear,
    /* .reset           = */ शून्य,
};

// ** backend buffer type

अटल व्योम ggml_backend_hexagon_host_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                        ggml_tensor *         tensor,
                                                        स्थिर व्योम *          तथ्य,
                                                        माप_प्रकार                offset,
                                                        माप_प्रकार                size) {
    स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size);
    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_hexagon_host_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                        स्थिर ggml_tensor *   tensor,
                                                        व्योम *                तथ्य,
                                                        माप_प्रकार                offset,
                                                        माप_प्रकार                size) {
    स_नकल(तथ्य, (स्थिर अक्षर *) tensor->तथ्य + offset, size);
    GGML_UNUSED(बफर);
}

अटल ggml_backend_बफर_i ggml_backend_hexagon_host_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_hexagon_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_बफर_init_tensor,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_hexagon_host_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_hexagon_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_बफर_clear,
    /* .reset           = */ शून्य,
};

// ** backend buffer type

अटल स्थिर अक्षर * ggml_backend_hexagon_बफर_type_name(ggml_backend_बफर_type_t बफर_type) {
    वापस static_cast<ggml_backend_hexagon_बफर_type_context *>(बफर_type->context)->name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_hexagon_बफर_type_alloc_बफर(
            ggml_backend_बफर_type_t बफर_type, माप_प्रकार size) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_बफर_type_context *>(बफर_type->context)->dev_ctx;
    स्वतः sess    = dev_ctx->session();
    अगर (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        वापस nullptr;
    }
    प्रयास {
        ggml_hexagon_shared_बफर * sbuf = नया ggml_hexagon_shared_बफर(sess, size, false);
        वापस ggml_backend_बफर_init(बफर_type, ggml_backend_hexagon_बफर_interface, sbuf, size);
    } पकड़ो (स्थिर std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        वापस nullptr;
    }
}

अटल ggml_backend_बफर_t ggml_backend_hexagon_host_बफर_type_alloc_बफर(
            ggml_backend_बफर_type_t बफर_type, माप_प्रकार size) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_बफर_type_context *>(बफर_type->context)->dev_ctx;
    स्वतः sess    = dev_ctx->session();
    अगर (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        वापस nullptr;
    }
    प्रयास {
        ggml_hexagon_shared_बफर * sbuf = नया ggml_hexagon_shared_बफर(sess, size, false);
        वापस ggml_backend_बफर_init(बफर_type, ggml_backend_hexagon_host_बफर_interface, sbuf, size);
    } पकड़ो (स्थिर std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        वापस nullptr;
    }
}

अटल माप_प्रकार ggml_backend_hexagon_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 128;  // HVX alignment
    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_hexagon_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर काष्ठा ggml_tensor * t) {
    अगर (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        माप_प्रकार tiled_आकार = htp_tiled_row_आकार(t->type, ne0) * ne1 * ne2 * ne3;
        वापस std::max(tiled_आकार, ggml_nbytes(t));
    }
    वापस ggml_nbytes(t);
}

अटल माप_प्रकार parse_आकार(स्थिर अक्षर * str, माप_प्रकार default_unit = 1024 * 1024) {
    अगर (!str || str[0] == '\0') {
        वापस 0;
    }
    अक्षर * end = शून्य;
    द्विगुन val = म_से_भग्न(str, &end);
    अगर (val < 0) {
        वापस 0;
    }
    अगर (end && *end) {
        जबतक (*end == ' ') end++;
        अगर (*end == 'k' || *end == 'K') {
            वापस (माप_प्रकार) (val * 1024);
        } अन्यथा अगर (*end == 'm' || *end == 'M') {
            वापस (माप_प्रकार) (val * 1024 * 1024);
        } अन्यथा अगर (*end == 'g' || *end == 'G') {
            वापस (माप_प्रकार) (val * 1024 * 1024 * 1024);
        }
    }
    वापस (माप_प्रकार) (val * default_unit);
}

अटल माप_प्रकार ggml_backend_hexagon_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    वापस opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

अटल bool ggml_backend_hexagon_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;
    GGML_UNUSED(buft);
}

अटल bool ggml_backend_hexagon_host_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    वापस true;
    GGML_UNUSED(buft);
}

काष्ठा ggml_backend_hexagon_alloc_बफर_n_plan_item {
    माप_प्रकार size;
    पूर्णांक    first;
    पूर्णांक    last;
};

using ggml_backend_hexagon_alloc_बफर_n_plan_t = std::vector<ggml_backend_hexagon_alloc_बफर_n_plan_item>;

अटल स्थिर अक्षर * ggml_hexagon_kv_layer_suffix(स्थिर काष्ठा ggml_tensor * t) {
    अगर (म_भेदन(t->name, "cache_", 6) != 0) {
        वापस शून्य;
    }
    स्थिर अक्षर * p = म_माला(t->name, "_l");
    अगर (!p || !है_अंक((अचिन्हित अक्षर)p[2])) {
        वापस शून्य;
    }
    वापस p;
}

काष्ठा ggml_backend_hexagon_alloc_unit {
    माप_प्रकार size;
    पूर्णांक    first;
    पूर्णांक    last;
};

अटल ggml_backend_hexagon_alloc_बफर_n_plan_t ggml_backend_hexagon_alloc_बफर_n_plan(
        ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    ggml_backend_hexagon_alloc_बफर_n_plan_t plan;

    स्थिर माप_प्रकार alignment = ggml_backend_buft_get_alignment(buft);
    स्थिर माप_प्रकार max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    पूर्णांक i = 0;
    जबतक (i < n_tensors) {
        काष्ठा ggml_tensor * t = tensors[i];
        माप_प्रकार unit_आकार = 0;
        पूर्णांक unit_first = i;
        पूर्णांक unit_last = i + 1;

        अगर (t->तथ्य == शून्य && t->view_src == शून्य) {
            unit_आकार += GGML_PAD(ggml_backend_buft_get_alloc_आकार(buft, t), alignment);
        }

        स्थिर अक्षर * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        जबतक (unit_last < n_tensors) {
            काष्ठा ggml_tensor * अग्र = tensors[unit_last];

            अगर (अग्र->view_src != शून्य) {
                unit_last++;
                जारी;
            }

            अगर (layer_suffix != शून्य) {
                स्थिर अक्षर * अग्र_suffix = ggml_hexagon_kv_layer_suffix(अग्र);
                अगर (अग्र_suffix != शून्य && म_भेद(layer_suffix, अग्र_suffix) == 0) {
                    अगर (अग्र->तथ्य == शून्य) {
                        unit_आकार += GGML_PAD(ggml_backend_buft_get_alloc_आकार(buft, अग्र), alignment);
                    }
                    unit_last++;
                    जारी;
                }
            }

            अवरोध;
        }

        units.push_back({ unit_आकार, unit_first, unit_last });
        i = unit_last;
    }

    माप_प्रकार cur_buf_आकार  = 0;
    पूर्णांक    cur_buf_first = 0;

    क्रम (स्थिर स्वतः & unit : units) {
        अगर (unit.size == 0) {
            जारी;
        }

        अगर (cur_buf_आकार > 0 && (cur_buf_आकार + unit.size) > max_size) {
            plan.push_back({ cur_buf_आकार, cur_buf_first, unit.first });
            cur_buf_आकार  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_आकार += unit.size;
    }

    अगर (cur_buf_आकार > 0) {
        plan.push_back({ cur_buf_आकार, cur_buf_first, n_tensors });
    }

    वापस plan;
}

अटल ggml_backend_बफर_t ggml_backend_hexagon_बफर_type_alloc_बफर_n(
        ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिर ggml_backend_hexagon_alloc_बफर_n_plan_t plan = ggml_backend_hexagon_alloc_बफर_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_बफर_t> buffers;
    buffers.reserve(plan.size());

    क्रम (स्थिर स्वतः & item : plan) {
        ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(buft, item.size);
        अगर (बफर == शून्य) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            क्रम (ggml_backend_बफर_t b : buffers) {
                ggml_backend_बफर_free(b);
            }
            वापस शून्य;
        }

        काष्ठा ggml_tallocr tallocr = ggml_tallocr_new(बफर);

        काष्ठा ggml_tensor * t_failed = शून्य;
        क्रम (पूर्णांक j = item.first; j < item.last; j++) {
            काष्ठा ggml_tensor * t = tensors[j];
            अगर (t->तथ्य == शून्य) {
                अगर (t->view_src == शून्य) {
                    अगर (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                } अन्यथा अगर (t->बफर == शून्य) {
                    अगर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                }
            } अन्यथा {
                अगर (t->view_src != शून्य && t->बफर == शून्य) {
                    अगर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                }
            }
        }
        अगर (t_failed != शून्य) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            क्रम (ggml_backend_बफर_t b : buffers) {
                ggml_backend_बफर_free(b);
            }
            ggml_backend_बफर_free(बफर);
            वापस शून्य;
        }

        buffers.push_back(बफर);
    }

    अगर (buffers.empty()) {
        वापस शून्य;
    }

    अगर (buffers.size() == 1) {
        वापस buffers[0];
    }

    वापस ggml_backend_multi_बफर_alloc_बफर(buffers.तथ्य(), buffers.size());
}

अटल माप_प्रकार ggml_backend_hexagon_बफर_type_get_alloc_आकार_n(
        ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिर ggml_backend_hexagon_alloc_बफर_n_plan_t plan = ggml_backend_hexagon_alloc_बफर_n_plan(buft, tensors, n_tensors);

    माप_प्रकार योग = 0;
    क्रम (स्थिर स्वतः & item : plan) {
        योग += item.size;
    }
    वापस योग;
}

अटल ggml_backend_बफर_type_i ggml_backend_hexagon_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_बफर_type_alloc_बफर_n,
    /* .get_alignment       = */ ggml_backend_hexagon_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_hexagon_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_बफर_type_get_alloc_आकार_n,
    /* .is_host             = */ ggml_backend_hexagon_बफर_type_is_host,
};

अटल ggml_backend_बफर_type_i ggml_backend_hexagon_host_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_बफर_type_alloc_बफर_n,
    /* .get_alignment       = */ ggml_backend_hexagon_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_hexagon_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_बफर_type_get_alloc_आकार_n,
    /* .is_host             = */ ggml_backend_hexagon_host_बफर_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(पूर्णांक dev_id, स्थिर ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    बफर_type.device  = dev;
    बफर_type.iface   = ggml_backend_hexagon_बफर_type_interface;
    बफर_type.context = नया ggml_backend_hexagon_बफर_type_context(config.name, यह);

    host_बफर_type.device  = dev;
    host_बफर_type.iface   = ggml_backend_hexagon_host_बफर_type_interface;
    host_बफर_type.context = नया ggml_backend_hexagon_बफर_type_context(config.name + "-HOST", यह);

    fence_बफर_type.device  = dev;
    fence_बफर_type.iface   = ggml_backend_hexagon_बफर_type_interface;
    fence_बफर_type.context = नया ggml_backend_hexagon_बफर_type_context(config.name + "-FENCE", यह);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    मिटाओ static_cast<ggml_backend_hexagon_बफर_type_context *>(बफर_type.context);
    मिटाओ static_cast<ggml_backend_hexagon_बफर_type_context *>(host_बफर_type.context);
    मिटाओ static_cast<ggml_backend_hexagon_बफर_type_context *>(fence_बफर_type.context);
}

अटल bool ggml_backend_बफर_is_hexagon(स्थिर काष्ठा ggml_backend_बफर * b) {
    वापस b->buft->iface.get_alignment == ggml_backend_hexagon_बफर_type_get_alignment;
}

काष्ठा ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<पूर्णांक, पूर्णांक>                b_map; // buffer fd   to index
    std::unordered_map<स्थिर ggml_tensor*, पूर्णांक> t_map; // tensor ptr  to index
    std::unordered_multimap<व्योम*, पूर्णांक>         d_map; // tensor data to index

    अचिन्हित पूर्णांक n_bufs;     // num buffers in the batch
    अचिन्हित पूर्णांक n_tens;     // num tensors ...
    अचिन्हित पूर्णांक n_ops;      // num ops ...
    माप_प्रकार       b_vmem;     // sum of non-extended buffer sizes

    अचिन्हित पूर्णांक n_bufs_max;
    अचिन्हित पूर्णांक n_tens_max;
    अचिन्हित पूर्णांक n_ops_max;
    माप_प्रकार       b_vmem_max;

    व्योम reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.सफा();
        t_map.सफा();
        d_map.सफा();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, माप_प्रकार batch_आकार, माप_प्रकार max_vmem) {
        यह->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_आकार;
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

    bool empty() स्थिर { वापस n_ops == 0; }

    // add buffer and return its index
    पूर्णांक add_बफर(ggml_hexagon_shared_बफर * sbuf) {
        // Lookup by fd
        स्वतः it = b_map.find(sbuf->fd());
        अगर (it != b_map.end()) { वापस it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        पूर्णांक bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        अगर (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (व्योम*) sbuf->base(), (माप_प्रकार) b.size, b_vmem);

        वापस bi;
    }

    bool same_shape(स्थिर htp_tensor * h, स्थिर ggml_tensor * t) स्थिर {
        स्वतः extra = (ggml_hexagon_tensor_extra *) t->extra;

        स्थिर bool is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        स्थिर int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        स्थिर int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_आकार(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? माप(ggml_fp16_t) : t->nb[0];
        क्रमागत ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        वापस (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    पूर्णांक add_tensor(स्थिर ggml_tensor * t) {
        स्वतः extra = (ggml_hexagon_tensor_extra *) t->extra;
        स्वतः sbuf  = static_cast<ggml_hexagon_shared_बफर *>(t->बफर->context);

        // First lookup by tensor data
        स्वतः range = d_map.equal_range(t->तथ्य);
        क्रम (स्वतः it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            अगर (same_shape(h, t)) { वापस it->second; }
        }

        // Lookup by tensor ptr
        स्वतः it = t_map.find(t);
        अगर (it != t_map.end()) { वापस it->second; }

        // Add new tensor to the batch
        पूर्णांक ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->तथ्य, ti});

        uint64_t t_offset = (uint8_t *) t->तथ्य - sbuf->base();
        माप_प्रकार   t_आकार   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_बफर(sbuf);
        h.ti    = ti;
        h.तथ्य  = t_offset;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        स्थिर bool is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        अगर (is_tiled) {
            अगर (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = माप(ggml_fp16_t);
            } अन्यथा {
                h.nb[0] = t->nb[0];
            }
            स्थिर int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            स्थिर int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_आकार(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_आकार  = h.size;
        } अन्यथा {
            h.size  = t_आकार;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        अगर ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        अगर ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        अगर ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (व्योम*) t->तथ्य, (माप_प्रकार) t_offset, t_आकार, h.flags,
                (माप_प्रकार) h.ne[0], (माप_प्रकार) h.ne[1], (माप_प्रकार) h.ne[2], (माप_प्रकार) h.ne[3]);

        वापस ti;
    }

    bool fit_op(स्थिर htp_opnode & गाँठ) स्थिर {
        अगर (n_ops >= n_ops_max) वापस false;

        // check how much extras we will need
        माप_प्रकार extra_bufs = 0;
        माप_प्रकार extra_vmem = 0;
        माप_प्रकार extra_tens = 0;

        पूर्णांक seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णांक n_seen_bufs = 0;

        स्वतः fit_tensor = [&](स्थिर ggml_tensor *t) {
            अगर (!t) वापस;
            अगर (!t_map.count(t)) {
                extra_tens++;

                स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(t->बफर->context);
                पूर्णांक fd = sbuf->fd();
                अगर (!b_map.count(fd)) {
                    क्रम (पूर्णांक i = 0; i < n_seen_bufs; i++) {
                        अगर (seen_bufs[i] == fd) वापस;
                    }
                    अगर (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    अगर (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        क्रम (स्थिर स्वतः * src : गाँठ.get_inputs()) {
            fit_tensor(src);
        }
        क्रम (स्थिर स्वतः * output : गाँठ.get_outputs()) {
            fit_tensor(output);
        }

        अगर ((extra_bufs + n_bufs) > n_bufs_max) वापस false;
        अगर ((extra_tens + n_tens) > n_tens_max) वापस false;
        अगर ((extra_vmem + b_vmem) > b_vmem_max) वापस false;

        वापस true;
    }

    // assumes that fit_op() was called first and returned true
    व्योम add_op(स्थिर htp_opnode & गाँठ) {
        // Add new op

        अचिन्हित पूर्णांक n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = गाँठ;

        htp_op_desc &o = h_ops[n];
        स_नकल(o.params,        गाँठ.गाँठ->op_params, माप(गाँठ.गाँठ->op_params));
        स_नकल(o.kernel_params, गाँठ.kernel_params,   माप(o.kernel_params));
        o.opcode = गाँठ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        स्वतः inputs = गाँठ.get_inputs();
        क्रम (अचिन्हित पूर्णांक i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        स्वतः outputs = गाँठ.get_outputs();
        क्रम (अचिन्हित पूर्णांक i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    व्योम sort_buffers() {
        अगर (n_bufs <= 1) वापस;

        std::vector<पूर्णांक> order(n_bufs);
        क्रम (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) { order[i] = (पूर्णांक) i; }

        std::stable_sort(order.begin(), order.end(), [&](पूर्णांक a, पूर्णांक b) {
            वापस h_bufs[a].size > h_bufs[b].size;
        });

        bool already_sorted = true;
        क्रम (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) {
            अगर (order[i] != (पूर्णांक) i) {
                already_sorted = false;
                अवरोध;
            }
        }
        अगर (already_sorted) वापस;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        क्रम (अचिन्हित पूर्णांक new_bi = 0; new_bi < n_bufs; new_bi++) {
            पूर्णांक old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        क्रम (अचिन्हित पूर्णांक i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        क्रम (अचिन्हित पूर्णांक i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    व्योम update_mdev_group(uint32_t mdev_idx) {
        अगर (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    bool try_fuse_common(std::initializer_list<स्थिर ggml_tensor *> tensors) स्थिर {
        माप_प्रकार extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        पूर्णांक seen_bufs[HTP_OP_MAX_BUFS];
        पूर्णांक n_seen_bufs = 0;

        क्रम (स्थिर स्वतः * t : tensors) {
            अगर (!t || t_map.count(t)) {
                जारी;
            }
            extra_tens++;
            स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(t->बफर->context);
            पूर्णांक fd = sbuf->fd();
            अगर (!b_map.count(fd)) {
                bool found = false;
                क्रम (पूर्णांक i = 0; i < n_seen_bufs; i++) {
                    अगर (seen_bufs[i] == fd) {
                        found = true;
                        अवरोध;
                    }
                }
                अगर (!found) {
                    अगर (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    अगर (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        अगर ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            वापस false;
        }

        वापस true;
    }

    bool try_fuse_common(स्थिर ggml_tensor * t1, स्थिर ggml_tensor * t2) स्थिर {
        वापस try_fuse_common({t1, t2});
    }

    bool try_fuse_allreduce_add(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0 || opt_ar_select != 2) वापस false;
        अगर (गाँठ.opcode != HTP_OP_ADD) वापस false;

        htp_opnode & last_गाँठ = ops[n_ops - 1];
        अगर (last_गाँठ.opcode != HTP_OP_ALLREDUCE) वापस false;

        स्वतः * ar_kparams = (काष्ठा htp_allreduce_kernel_params *) last_गाँठ.kernel_params;
        स्थिर uint32_t rank    = (uint32_t) ar_kparams->rank;
        स्थिर uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        स्थिर ggml_tensor * ar_local = last_गाँठ.inputs[rank];
        स्थिर ggml_tensor * add_src0 = गाँठ.src0();
        स्थिर ggml_tensor * add_src1 = गाँठ.src1();
        स्थिर ggml_tensor * add_dst  = गाँठ.dst();

        अगर (!ggml_hexagon_tensor_is_fuseable(ar_local)) वापस false;

        स्थिर ggml_tensor * res_tensor;
        अगर (add_src0 == ar_local || add_src0->तथ्य == ar_local->तथ्य) {
            res_tensor = add_src1;
        } अन्यथा अगर (add_src1 == ar_local || add_src1->तथ्य == ar_local->तथ्य) {
            res_tensor = add_src0;
        } अन्यथा {
            वापस false;
        }

        अगर (ar_local->type != res_tensor->type) वापस false;

        स्थिर bool is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        स्थिर bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        अगर (!is_same_shape && !is_row_bcast) वापस false;

        अगर (is_same_shape) {
            अगर (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                वापस false;
            }
            अगर (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                वापस false;
            }
        }
        अगर (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            वापस false;
        }

        // scatter is only valid for in-place add within max outputs
        स्थिर bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->तथ्य == ar_local->तथ्य;

        काष्ठा htp_allreduce_kernel_params new_kparams;
        अगर (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, true, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            वापस false;
        }

        स्थिर bool scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        क्रम (uint32_t r = 0; r < n_ranks; r++) {
            स्थिर ggml_tensor * ar_src = last_गाँठ.inputs[r];
            अगर (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                जारी;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            अगर (scatter_ok && r == rank) {
                जारी;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            वापस false;
        }

        अगर (!try_fuse_common(res_tensor, add_dst)) {
            वापस false;
        }

        last_गाँठ.opcode = HTP_OP_ALLREDUCE_ADD;
        last_गाँठ.name   = "ALLREDUCE+ADD";
        last_गाँठ.inputs.push_back(res_tensor);
        last_गाँठ.outputs.सफा();
        last_गाँठ.outputs.push_back(add_dst);
        last_गाँठ.fused.push_back(गाँठ.गाँठ);
        स_नकल(last_गाँठ.kernel_params, &new_kparams, माप(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        स_नकल(o.kernel_params, &new_kparams, माप(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        अगर (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            क्रम (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            क्रम (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } अन्यथा {
            o.dst[0] = add_tensor(add_dst);
            क्रम (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (पूर्णांक) new_kparams.mode, (पूर्णांक) new_kparams.n_dsts);
        वापस true;
    }

    bool try_fuse_rms_norm_mul(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0) वापस false;
        अगर (गाँठ.opcode != HTP_OP_MUL) वापस false;

        htp_opnode & last_गाँठ = ops[n_ops - 1];
        अगर (last_गाँठ.opcode != HTP_OP_RMS_NORM) वापस false;

        स्थिर ggml_tensor * mul_src0 = गाँठ.src0();
        स्थिर ggml_tensor * mul_src1 = गाँठ.src1();
        स्थिर ggml_tensor * rms_out  = last_गाँठ.dst();

        अगर (!ggml_hexagon_tensor_is_fuseable(rms_out)) वापस false;

        स्थिर ggml_tensor * weight;
        अगर (mul_src0 == rms_out || mul_src0->तथ्य == rms_out->तथ्य) {
            weight = mul_src1;
        } अन्यथा अगर (mul_src1 == rms_out || mul_src1->तथ्य == rms_out->तथ्य) {
            weight = mul_src0;
        } अन्यथा {
            वापस false;
        }

        स्थिर ggml_tensor * src0 = last_गाँठ.src0();

        अगर (src0->ne[0] != weight->ne[0] || src0->ne[0] != गाँठ.dst()->ne[0]) {
            वापस false;
        }

        स्थिर bool is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        स्थिर bool is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        अगर (!is_row_bcast && !is_same_shape) वापस false;

        अगर (!ggml_are_same_shape(src0, गाँठ.dst())) {
            वापस false;
        }
        अगर (ggml_is_contiguous(src0) != ggml_is_contiguous(गाँठ.dst())) {
            वापस false;
        }

        काष्ठा htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, गाँठ.dst(), &new_kparams
        );

        अगर ((माप_प्रकार) new_kparams.vtcm_आकार > sess->vtcm_आकार) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_आकार, sess->vtcm_आकार);
            वापस false;
        }

        अगर (!try_fuse_common(weight, गाँठ.dst())) {
            वापस false;
        }

        last_गाँठ.opcode = HTP_OP_RMS_NORM_MUL;
        last_गाँठ.name   = "RMS_NORM+MUL";
        last_गाँठ.inputs.सफा();
        last_गाँठ.inputs.push_back(src0);
        last_गाँठ.inputs.push_back(weight);
        last_गाँठ.outputs.सफा();
        last_गाँठ.outputs.push_back(गाँठ.dst());
        last_गाँठ.fused.push_back(गाँठ.गाँठ);
        स_नकल(last_गाँठ.kernel_params, &new_kparams, माप(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        स_नकल(o.kernel_params, &new_kparams, माप(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        क्रम (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाँठ.dst());
        क्रम (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        वापस true;
    }

    bool try_fuse_mul_mat_add(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0) वापस false;
        अगर (गाँठ.opcode != HTP_OP_ADD) वापस false;

        htp_opnode & last_गाँठ = ops[n_ops - 1];
        अगर (last_गाँठ.opcode != HTP_OP_MUL_MAT) वापस false;

        स्थिर ggml_tensor * add_src0 = गाँठ.src0();
        स्थिर ggml_tensor * add_src1 = गाँठ.src1();
        स्थिर ggml_tensor * mm_out   = last_गाँठ.dst();

        अगर (!ggml_hexagon_tensor_is_fuseable(mm_out)) वापस false;

        स्थिर ggml_tensor * src2;
        अगर (add_src0 == mm_out || add_src0->तथ्य == mm_out->तथ्य) {
            src2 = add_src1;
        } अन्यथा अगर (add_src1 == mm_out || add_src1->तथ्य == mm_out->तथ्य) {
            src2 = add_src0;
        } अन्यथा {
            वापस false;
        }

        स्थिर ggml_tensor * src0 = last_गाँठ.src0();
        स्थिर ggml_tensor * src1 = last_गाँठ.src1();

        अगर (src2->type != GGML_TYPE_F32) वापस false;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        अगर (!ggml_are_same_shape(mm_out, गाँठ.dst())) वापस false;

        स्थिर काष्ठा htp_mm_kernel_params * orig_kparams = (स्थिर काष्ठा htp_mm_kernel_params *) last_गाँठ.kernel_params;
        काष्ठा htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, गाँठ.dst(), &kparams);
        अगर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            वापस false;
        }

        अगर (orig_kparams->collapse != kparams.collapse) {
            वापस false;
        }

        स्थिर पूर्णांक src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        स्थिर bool can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        अगर (!can_fuse) वापस false;

        अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
            वापस false;
        }

        अगर (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            अगर (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                वापस false;
            }
        }

        अगर (!try_fuse_common(src2, गाँठ.dst())) {
            वापस false;
        }

        last_गाँठ.opcode = HTP_OP_MUL_MAT_ADD;
        last_गाँठ.name   = "MUL_MAT+ADD";
        last_गाँठ.inputs.सफा();
        last_गाँठ.inputs.push_back(src0);
        last_गाँठ.inputs.push_back(src1);
        last_गाँठ.inputs.push_back(src2);
        last_गाँठ.outputs.सफा();
        last_गाँठ.outputs.push_back(गाँठ.dst());
        last_गाँठ.fused.push_back(गाँठ.गाँठ);
        स_नकल(last_गाँठ.kernel_params, &kparams, माप(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        स_नकल(o.kernel_params, &kparams, माप(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        क्रम (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(गाँठ.dst());
        क्रम (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        वापस true;
    }

    bool try_fuse_mul_mat_nx(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0 || गाँठ.opcode != HTP_OP_MUL_MAT) वापस false;
        अगर (!is_mergeable_mul_mat(गाँठ.गाँठ)) वापस false;

        स्थिर ggml_tensor * w_in = गाँठ.src0();
        स्थिर ggml_tensor * x_in = गाँठ.src1();
        स्थिर ggml_tensor * d_in = गाँठ.dst();

        htp_opnode & last_गाँठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        अगर (last_गाँठ.opcode == HTP_OP_MUL_MAT_NX) {
            स्थिर uint32_t curr_n = (uint32_t) last_गाँठ.outputs.size();
            अगर (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                वापस false;
            }

            स्थिर ggml_tensor * w0 = last_गाँठ.inputs[0];
            स्थिर ggml_tensor * x  = last_गाँठ.inputs[curr_n];

            अगर (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                वापस false;
            }
            अगर (!last_गाँठ.fused.empty() && (mm_is_hmx_eligible(last_गाँठ.fused[0]) != mm_is_hmx_eligible(गाँठ.गाँठ))) {
                वापस false;
            }

            स्थिर काष्ठा htp_mm_kernel_params * orig_kparams = (स्थिर काष्ठा htp_mm_kernel_params *) last_गाँठ.kernel_params;
            स्थिर bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            अगर (orig_kparams->collapse && !collapse) {
                वापस false;
            }

            काष्ठा htp_mm_kernel_params kparams;
            अगर (collapse) {
                स्थिर ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            अगर (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                वापस false;
            }
            अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
                वापस false;
            }

            अगर (!try_fuse_common(w_in, d_in)) {
                वापस false;
            }

            last_गाँठ.inputs[curr_n] = w_in;
            last_गाँठ.inputs.push_back(x);
            last_गाँठ.outputs.push_back(d_in);
            last_गाँठ.fused.push_back(गाँठ.गाँठ);
            स_नकल(last_गाँठ.kernel_params, &kparams, माप(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            स_नकल(o.kernel_params, &kparams, माप(kparams));

            क्रम (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_गाँठ.inputs[s]);
            }
            क्रम (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            क्रम (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाँठ.outputs[d]);
            }
            क्रम (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            वापस true;
        }

        // Case 2: last_node is single MUL_MAT
        अगर (last_गाँठ.opcode == HTP_OP_MUL_MAT) {
            अगर (!is_mergeable_mul_mat_pair(last_गाँठ.गाँठ, गाँठ.गाँठ)) {
                वापस false;
            }

            स्थिर ggml_tensor * w0 = last_गाँठ.src0();
            स्थिर ggml_tensor * x  = last_गाँठ.src1();
            स्थिर ggml_tensor * w1 = गाँठ.src0();
            स्थिर ggml_tensor * dst_0 = last_गाँठ.dst();
            स्थिर ggml_tensor * dst_1 = गाँठ.dst();

            स्थिर काष्ठा htp_mm_kernel_params * orig_kparams = (स्थिर काष्ठा htp_mm_kernel_params *) last_गाँठ.kernel_params;
            स्थिर bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            अगर (orig_kparams->collapse && !collapse) {
                वापस false;
            }

            काष्ठा htp_mm_kernel_params kparams;
            अगर (collapse) {
                स्थिर ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } अन्यथा {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            अगर (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                वापस false;
            }
            अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
                वापस false;
            }

            अगर (!try_fuse_common(w1, गाँठ.dst())) {
                वापस false;
            }

            last_गाँठ.opcode = HTP_OP_MUL_MAT_NX;
            last_गाँठ.name   = "MUL_MAT_NX";
            last_गाँठ.inputs.सफा();
            last_गाँठ.inputs.push_back(w0);
            last_गाँठ.inputs.push_back(w1);
            last_गाँठ.inputs.push_back(x);
            last_गाँठ.outputs.सफा();
            last_गाँठ.outputs.push_back(dst_0);
            last_गाँठ.outputs.push_back(dst_1);
            last_गाँठ.fused.push_back(गाँठ.गाँठ);
            स_नकल(last_गाँठ.kernel_params, &kparams, माप(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            स_नकल(o.kernel_params, &kparams, माप(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            क्रम (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            क्रम (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            वापस true;
        }

        वापस false;
    }

    bool try_fuse_mul_mat_id_nx(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0 || गाँठ.opcode != HTP_OP_MUL_MAT_ID) वापस false;
        अगर (!is_mergeable_mul_mat_id(गाँठ.गाँठ)) वापस false;

        स्थिर ggml_tensor * w_in   = गाँठ.src0();
        स्थिर ggml_tensor * x_in   = गाँठ.src1();
        स्थिर ggml_tensor * ids_in = गाँठ.गाँठ->src[2];
        स्थिर ggml_tensor * d_in   = गाँठ.dst();

        htp_opnode & last_गाँठ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        अगर (last_गाँठ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            स्थिर uint32_t curr_n = (uint32_t) last_गाँठ.outputs.size();
            अगर (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                वापस false;
            }

            स्थिर ggml_tensor * w0  = last_गाँठ.inputs[0];
            स्थिर ggml_tensor * x   = last_गाँठ.inputs[curr_n];
            स्थिर ggml_tensor * ids = last_गाँठ.inputs[curr_n + 1];

            अगर (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                वापस false;
            }
            अगर (!last_गाँठ.fused.empty() && (mm_is_hmx_eligible(last_गाँठ.fused[0]) != mm_is_hmx_eligible(गाँठ.गाँठ))) {
                वापस false;
            }

            काष्ठा htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            अगर (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                वापस false;
            }
            अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
                वापस false;
            }

            अगर (!try_fuse_common(w_in, d_in)) {
                वापस false;
            }

            last_गाँठ.inputs[curr_n] = w_in;
            last_गाँठ.inputs[curr_n + 1] = x;
            last_गाँठ.inputs.push_back(ids);
            last_गाँठ.outputs.push_back(d_in);
            last_गाँठ.fused.push_back(गाँठ.गाँठ);
            स_नकल(last_गाँठ.kernel_params, &kparams, माप(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            स_नकल(o.kernel_params, &kparams, माप(kparams));

            क्रम (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_गाँठ.inputs[s]);
            }
            क्रम (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            क्रम (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_गाँठ.outputs[d]);
            }
            क्रम (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            वापस true;
        }

        // Case 2: last_node is single MUL_MAT_ID
        अगर (last_गाँठ.opcode == HTP_OP_MUL_MAT_ID) {
            अगर (!is_mergeable_mul_mat_id_pair(last_गाँठ.गाँठ, गाँठ.गाँठ)) {
                वापस false;
            }

            स्थिर ggml_tensor * w0  = last_गाँठ.src0();
            स्थिर ggml_tensor * x   = last_गाँठ.src1();
            स्थिर ggml_tensor * ids = last_गाँठ.गाँठ->src[2];
            स्थिर ggml_tensor * w1  = गाँठ.src0();

            काष्ठा htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, गाँठ.dst(), 2, &kparams);
            अगर (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                वापस false;
            }
            अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
                वापस false;
            }

            अगर (!try_fuse_common(w1, गाँठ.dst())) {
                वापस false;
            }

            स्थिर ggml_tensor * dst_0 = last_गाँठ.dst();
            स्थिर ggml_tensor * dst_1 = गाँठ.dst();

            last_गाँठ.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_गाँठ.name   = "MUL_MAT_ID_NX";
            last_गाँठ.inputs.सफा();
            last_गाँठ.inputs.push_back(w0);
            last_गाँठ.inputs.push_back(w1);
            last_गाँठ.inputs.push_back(x);
            last_गाँठ.inputs.push_back(ids);
            last_गाँठ.outputs.सफा();
            last_गाँठ.outputs.push_back(dst_0);
            last_गाँठ.outputs.push_back(dst_1);
            last_गाँठ.fused.push_back(गाँठ.गाँठ);
            स_नकल(last_गाँठ.kernel_params, &kparams, माप(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            स_नकल(o.kernel_params, &kparams, माप(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            क्रम (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            क्रम (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            वापस true;
        }

        वापस false;
    }

    bool try_fuse_gdn_cpy(स्थिर htp_opnode & गाँठ) {
        अगर (n_ops == 0 || गाँठ.opcode != HTP_OP_CPY) वापस false;

        htp_opnode & last_गाँठ = ops[n_ops - 1];
        अगर (last_गाँठ.opcode != HTP_OP_GATED_DELTA_NET) वापस false;
        अगर (last_गाँठ.outputs.size() != 1) वापस false;

        स्थिर ggml_tensor * gdn_out  = last_गाँठ.dst();
        स्थिर ggml_tensor * cpy_गाँठ = गाँठ.गाँठ;
        स्थिर ggml_tensor * cpy_src  = गाँठ.src0();
        स्थिर ggml_tensor * cpy_dst  = गाँठ.dst();

        अगर (!cpy_src || !cpy_dst || !cpy_dst->तथ्य) वापस false;
        अगर (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) वापस false;
        अगर ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_गाँठ->flags & GGML_TENSOR_FLAG_OUTPUT)) वापस false;

        स्थिर ggml_tensor * v = last_गाँठ.गाँठ->src[2];
        अगर (!v) वापस false;

        स्थिर int64_t S_v      = v->ne[0];
        स्थिर int64_t H        = v->ne[1];
        स्थिर int64_t n_tokens = v->ne[2];
        स्थिर int64_t n_seqs   = v->ne[3];
        स्थिर int64_t K        = ggml_get_op_params_i32(last_गाँठ.गाँठ, 0);
        स्थिर माप_प्रकार  पुच्छ_off = (माप_प्रकार) S_v * H * n_tokens * n_seqs * माप(भग्न);

        स्थिर int64_t D         = S_v * S_v * H;
        स्थिर int64_t n_written = std::min<int64_t>(n_tokens, K);

        अगर (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->तथ्य != gdn_out->तथ्य) ||
            cpy_src->view_offs != पुच्छ_off || !ggml_is_contiguous(cpy_src)) {
            वापस false;
        }

        अगर (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != माप(भग्न)) {
            वापस false;
        }
        अगर (n_seqs > 1 && cpy_dst->nb[1] != (माप_प्रकार) D * माप(भग्न)) {
            वापस false;
        }
        अगर (n_written > 1) {
            अगर (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (माप_प्रकार) D * n_seqs * माप(भग्न)) {
                वापस false;
            }
        }

        अगर (!try_fuse_common({cpy_dst})) {
            वापस false;
        }

        last_गाँठ.name += "+CPY";
        last_गाँठ.outputs.push_back(cpy_dst);
        last_गाँठ.fused.push_back(गाँठ.गाँठ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        क्रम (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        वापस true;
    }

    bool try_fuse(स्थिर htp_opnode & गाँठ) {
        अगर (!opt_opfusion) वापस false;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(गाँठ)) वापस true;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(गाँठ))  वापस true;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(गाँठ))   वापस true;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(गाँठ))    वापस true;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(गाँठ)) वापस true;
        अगर (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(गाँठ))       वापस true;
        वापस false;
    }
};

काष्ठा ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

काष्ठा ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_बफर *shm_buf;
    माप_प्रकार                      shm_blk_आकार;
    माप_प्रकार                      depth;

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, माप_प्रकार batch_आकार, माप_प्रकार depth) : depth(depth) {
        माप_प्रकार n_bufs    = HTP_OP_MAX_BUFS;
        माप_प्रकार n_ops     = batch_आकार;
        माप_प्रकार n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        माप_प्रकार tr_आकार = 0;
        अगर (opt_profile == 3) {
            tr_आकार = (HTP_MAX_NTHREADS + 1) * opt_optrace * माप(htp_trace_desc);
        }

        shm_blk_आकार = माप(htp_buf_desc)  * n_bufs    +
                       माप(htp_tensor)    * n_tensors +
                       माप(htp_op_desc)   * n_ops     +
                       माप(htp_prof_desc) * n_ops     +
                       tr_आकार;

        shm_buf = नया ggml_hexagon_shared_बफर(sess, shm_blk_आकार * depth, true /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        अगर (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_आकार, depth, shm_buf->size(), shm_blk_आकार);
        }
    }

    ~ggml_hexagon_opqueue() {
        मिटाओ shm_buf;
    }

    माप_प्रकार shm_आकार() स्थिर { वापस shm_buf ? shm_buf->size() : 0; }

    // push new batch
    bool धकेलो(htp_opbatch_req& req, dspqueue_बफर& dbuf, स्थिर ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(माप(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(माप(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(माप(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(माप(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(माप(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(माप(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        अगर (seq - shm_buf->sess->batch_rsp_seq > depth) { वापस false; }

        स्थिर uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        स्थिर माप_प्रकार b_आकार = माप(htp_buf_desc)  * req.n_bufs;
        स्थिर माप_प्रकार t_आकार = माप(htp_tensor)    * req.n_tensors;
        स्थिर माप_प्रकार o_आकार = माप(htp_op_desc)   * req.n_ops;
        स्थिर माप_प्रकार p_आकार = माप(htp_prof_desc) * req.n_ops;

        माप_प्रकार tr_आकार = 0;
        अगर (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_आकार = (HTP_MAX_NTHREADS + 1) * req.n_traces * माप(htp_trace_desc);
        } अन्यथा {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((माप_प्रकार) slot * shm_blk_आकार);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_बफर_FLAG_FLUSH_SENDER | DSPQUEUE_बफर_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_आकार + t_आकार + o_आकार + p_आकार + tr_आकार;

        GGML_ASSERT(dbuf.size <= shm_blk_आकार);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_आकार;
        uint8_t * t_ptr = m_ptr; m_ptr += t_आकार;
        uint8_t * o_ptr = m_ptr;

        स_नकल(b_ptr, (व्योम *) op_batch->h_bufs.तथ्य(), b_आकार);
        स_नकल(t_ptr, (व्योम *) op_batch->h_tens.तथ्य(), t_आकार);
        स_नकल(o_ptr, (व्योम *) op_batch->h_ops.तथ्य(),  o_आकार);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (अचिन्हित दीर्घ दीर्घ) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_आकार, t_आकार, o_आकार, (माप_प्रकार) dbuf.size);

        अगर (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            क्रम (अचिन्हित पूर्णांक i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (व्योम *) b[i].base, (माप_प्रकार) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            क्रम (अचिन्हित पूर्णांक i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (अचिन्हित दीर्घ दीर्घ) t[i].तथ्य, t[i].size,
                            (माप_प्रकार) t[i].ne[0], (माप_प्रकार) t[i].ne[1], (माप_प्रकार) t[i].ne[2], (माप_प्रकार) t[i].ne[3]);
            }
        }

        वापस true;
    }

    व्योम निकालो(htp_opbatch_rsp rsp, dspqueue_बफर dbuf) {
        स्थिर uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        स्थिर माप_प्रकार b_आकार = माप(htp_buf_desc)  * rsp.n_bufs;
        स्थिर माप_प्रकार t_आकार = माप(htp_tensor)    * rsp.n_tensors;
        स्थिर माप_प्रकार o_आकार = माप(htp_op_desc)   * rsp.n_ops;
        स्थिर माप_प्रकार p_आकार = माप(htp_prof_desc) * rsp.n_ops;

        माप_प्रकार tr_आकार = 0;
        uint32_t n_traces = 0;
        अगर (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_आकार = (HTP_MAX_NTHREADS + 1) * n_traces * माप(htp_trace_desc);
        }

        स्थिर माप_प्रकार m_आकार = b_आकार + t_आकार + o_आकार + p_आकार + tr_आकार;
        GGML_ASSERT(m_आकार <= shm_blk_आकार);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (अचिन्हित दीर्घ दीर्घ) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (माप_प्रकार) dbuf.size, b_आकार, t_आकार, o_आकार);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_आकार + t_आकार + o_आकार);

        अगर (rsp.n_ops > 0) {
            स्वतः & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            स्थिर htp_prof_desc * pd = (स्थिर htp_prof_desc *) p_ptr;
            स्थिर htp_trace_desc * trace_events = nullptr;
            अगर (opt_profile == 3) {
                trace_events = (स्थिर htp_trace_desc *) (p_ptr + p_आकार);
            }

            अगर (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            क्रम (uint32_t i = 0; i < rsp.n_ops; i++) {
                अगर (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            अगर (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

व्योम ggml_hexagon_session::flush_peers() {
    स्वतः vpeers = std::move(virt_peers);
    virt_peers.सफा();
    क्रम (स्वतः * peer : vpeers) {
        peer->flush_sync();
    }

    स्वतः ppeers = std::move(phys_peers);
    phys_peers.सफा();
    क्रम (स्वतः * peer : ppeers) {
        peer->flush_async();
    }

    क्रम (स्वतः & sub : यह->mdev.sessions) {
        sub->flush_peers();
    }
}

व्योम ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

व्योम ggml_hexagon_session::flush_pending(bool all) {
    क्रम (स्वतः & sub : यह->mdev.sessions) {
        sub->flush_pending(all);
        अगर (sub->last_error > HTP_STATUS_OK) {
            यह->last_error = sub->last_error.load();
        }
    }

    जबतक (यह->batch_rsp_seq < यह->batch_req_seq) {
        काष्ठा htp_opbatch_rsp rsp;
        uint32_t               rsp_आकार;
        uint32_t               flags;

        काष्ठा dspqueue_बफर dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        स्थिर uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        पूर्णांक err = dspqueue_read(यह->queue, &flags, 1, &n_dbufs, &dbuf, माप(rsp), &rsp_आकार, (uint8_t *) &rsp, timeo);
        अगर (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            जारी;
        }

        अगर (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (अचिन्हित) err);
        }

        // Basic sanity checks
        अगर (rsp_आकार != माप(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", यह->c_name(), rsp_आकार, n_dbufs);
        }

        अगर (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", यह->c_name(), status_to_str(rsp.status));
            यह->last_error = rsp.status;
            क्रम (स्वतः & sub : यह->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->निकालो(rsp, dbuf);

        GGML_ASSERT(rsp.seq == यह->batch_rsp_seq + 1);
        यह->batch_rsp_seq = rsp.seq;

        अगर (!all) अवरोध;
    }
}

व्योम ggml_hexagon_session::flush_sync(bool all) {
    flush_async();
    flush_pending(all);
}

व्योम ggml_hexagon_session::start_batch() {
    अगर (यह->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

व्योम ggml_hexagon_session::flush_batch(माप_प्रकार min_ops) {
    अगर (op_batch->n_ops < min_ops) { वापस; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_बफर dbuf{};

    स्थिर uint64_t seq = ++यह->batch_req_seq;

    op_batch->update_mdev_group(यह->mdev.idx);

    अगर (!op_queue->धकेलो(req, dbuf, op_batch, seq)) {
        flush_pending(false);
        op_queue->धकेलो(req, dbuf, op_batch, seq);
    }

    क्रम (स्वतः & sub : यह->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_बफर sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        अगर (!sub->op_queue->धकेलो(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(false);
            sub->op_queue->धकेलो(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        पूर्णांक err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, माप(sub_req), (स्थिर uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        अगर (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (अचिन्हित) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", यह->c_name(), dbuf.ptr, dbuf.size);

    पूर्णांक err = dspqueue_write(यह->queue, 0, 1, &dbuf, माप(req), (स्थिर uint8_t*) &req, DSPQUEUE_TIMEOUT);
    अगर (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", यह->c_name(), (अचिन्हित) err);
    }

    op_batch->reset();
}

व्योम ggml_hexagon_session::enqueue_op(स्थिर htp_opnode & गाँठ) {
    स्वतः clone_tensor_बफर = [यह](स्थिर ggml_tensor * t) {
        स्वतः sbuf = यह->mmap_tensor(t);
        अगर (!sbuf) वापस;
        अगर (sbuf->sess != यह) {
            यह->clone_बफर(sbuf);
        }
        क्रम (स्वतः & sub : यह->mdev.sessions) {
            sub->clone_बफर(sbuf);
        }
    };

    क्रम (स्वतः t : गाँठ.get_inputs()) {
        clone_tensor_बफर(t);
    }
    क्रम (स्वतः t : गाँठ.get_outputs()) {
        clone_tensor_बफर(t);
    }

    अगर (opt_opfusion && op_batch->try_fuse(गाँठ)) {
        वापस;
    }

    अगर (!op_batch->fit_op(गाँठ)) {
        flush_async();
    }

    अगर (op_batch->empty()) {
        start_batch();
    }

    अगर (!op_batch->fit_op(गाँठ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(गाँठ);
}

व्योम ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_गाँठ(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = यह->mdev_fence_slot;

    अटल ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.बफर       = &यह->fence_buf->backend_बफर;
    dummy_t.extra        = &fence_extra;
    dummy_t.तथ्य         = (व्योम *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_आकार;
    dummy_t.ne[1]        = (int64_t) यह->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_आकार;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) यह->mdev.idx;

    ggml_tensor * गाँठ = group_गाँठ.add_dummy(dummy_t);
    गाँठ->src[0] = गाँठ;
    group_गाँठ.init(गाँठ);
    group_गाँठ.outputs.सफा();
    group_गाँठ.name = "MDEV_GROUP";

    क्रम (स्वतः & sub : यह->mdev.sessions) {
        sub->clone_बफर(यह->fence_buf);
    }

    op_batch->add_op(group_गाँठ);
}

व्योम ggml_hexagon_session::enqueue_cpy(स्थिर ggml_tensor * src, ggml_tensor * dst, स्थिर ggml_tensor * sync_tensor, uint32_t fence_seq) {
    स्थिर bool with_fence = sync_tensor != nullptr;
    htp_opnode cpy_गाँठ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* गाँठ = cpy_गाँठ.add_dummy(*dst);
    गाँठ->op     = GGML_OP_CPY;
    गाँठ->src[0] = const_cast<ggml_tensor *>(src);
    गाँठ->src[1] = with_fence ? cpy_गाँठ.add_dummy(*sync_tensor) : nullptr;
    अगर (with_fence) {
        गाँठ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_गाँठ.init(गाँठ);
    अगर (with_fence) {
        cpy_गाँठ.name = "CPY+FENCE";
    }
    स्थिर bool ok = ggml_hexagon_precompute_cpy_params(यह, गाँठ, (काष्ठा htp_copy_kernel_params *) cpy_गाँठ.kernel_params);
    स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_copy_kernel_params *) cpy_गाँठ.kernel_params;
    अगर (ok && !with_fence && kparams->योग_elems == 0) {
        वापस;
    }
    यह->enqueue_op(cpy_गाँठ);
}

व्योम ggml_hexagon_session::enqueue_fence(स्थिर ggml_tensor * sync_tensor, uint32_t fence_seq, bool रुको) {
    htp_opnode sync_गाँठ(HTP_OP_FENCE);

    ggml_tensor* गाँठ = sync_गाँठ.add_dummy(*sync_tensor);
    गाँठ->op           = GGML_OP_NONE;
    गाँठ->src[0]       = गाँठ;
    गाँठ->op_params[0] = (int32_t) fence_seq;
    गाँठ->op_params[1] = रुको ? 0 : 1;

    sync_गाँठ.init(गाँठ);
    sync_गाँठ.name = रुको ? "FENCE_WAIT" : "FENCE_SIGNAL";
    यह->enqueue_op(sync_गाँठ);
}

अटल bool ggml_hexagon_precompute_allreduce_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    काष्ठा htp_allreduce_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    स्थिर uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    स्थिर uint32_t elem_आकार = (dst->type == GGML_TYPE_F16) ? माप(ggml_fp16_t) : माप(भग्न);
    स्थिर bool is_contiguous = ggml_is_contiguous(dst);

    स्थिर uint32_t ne0 = (uint32_t) dst->ne[0];
    स्थिर uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    स्थिर bool use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    अगर (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        अगर (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            स्थिर uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            स्थिर uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            स्थिर uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } अन्यथा अगर (use_1d) {
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
        अगर (use_1d) {
            स्थिर uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            स्थिर uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            स्थिर uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            स्थिर uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } अन्यथा {
            स्थिर uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            स्थिर uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            स्थिर uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            स्थिर uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    अगर (use_1d) {
        स्थिर uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        स्थिर माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_बफर_गिनती(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        अगर (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_आकार_per_thread = 2 * block_elems * elem_आकार;
        kparams->vtcm_आकार            = n_vtcm_buffers * kparams->vtcm_आकार_per_thread;

        जबतक ((माप_प्रकार) kparams->vtcm_आकार > sess->vtcm_आकार && block_elems > 128) {
            स्थिर माप_प्रकार max_bytes_per_buf = sess->vtcm_आकार / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((माप_प्रकार) (max_bytes_per_buf / elem_आकार), 128);
            अगर (block_elems < 128) अवरोध;
            kparams->block_elems          = block_elems;
            kparams->vtcm_आकार_per_thread = 2 * block_elems * elem_आकार;
            kparams->vtcm_आकार            = n_vtcm_buffers * kparams->vtcm_आकार_per_thread;
        }

        अगर (sess->vtcm_आकार < (माप_प्रकार) kparams->vtcm_आकार || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_आकार, sess->vtcm_आकार);
            वापस false;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        वापस true;
    } अन्यथा {
        स्थिर uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        स्थिर माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_बफर_गिनती(n_ranks, n_threads, has_add, is_row_bcast);

        स्थिर uint32_t row_bytes = ne0 * elem_आकार;
        स्थिर uint32_t row_आकार_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_आकार_aligned = row_आकार_aligned;

        स्थिर uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_आकार_per_thread = 2 * (block_rows * row_आकार_aligned);
        kparams->vtcm_आकार            = n_vtcm_buffers * kparams->vtcm_आकार_per_thread;

        जबतक ((माप_प्रकार) kparams->vtcm_आकार > sess->vtcm_आकार && block_rows > 1) {
            स्थिर माप_प्रकार max_rows_per_buf = sess->vtcm_आकार / (n_vtcm_buffers * 2 * row_आकार_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_आकार_per_thread = 2 * (block_rows * row_आकार_aligned);
            kparams->vtcm_आकार            = n_vtcm_buffers * kparams->vtcm_आकार_per_thread;
            अगर (max_rows_per_buf == 0) अवरोध;
        }

        अगर (sess->vtcm_आकार < (माप_प्रकार) kparams->vtcm_आकार || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_आकार, sess->vtcm_आकार);
            वापस false;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        वापस true;
    }
}

व्योम ggml_hexagon_session::enqueue_allreduce(
    स्थिर ggml_tensor * dst,
    स्थिर std::vector<स्थिर ggml_tensor *> & src_tensors,
    स्थिर std::vector<स्थिर ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_गाँठ(HTP_OP_ALLREDUCE);

    ggml_tensor* गाँठ = ar_गाँठ.add_dummy(*dst);
    गाँठ->op           = GGML_OP_NONE;
    गाँठ->op_params[0] = (int32_t) fence_seq_entry;
    गाँठ->op_params[1] = (int32_t) fence_seq_exit;

    ar_गाँठ.init(गाँठ);

    ar_गाँठ.inputs.सफा();
    क्रम (माप_प्रकार i = 0; i < src_tensors.size(); i++) {
        ar_गाँठ.inputs.push_back(src_tensors[i]);
    }
    क्रम (माप_प्रकार i = 0; i < sync_tensors.size(); i++) {
        ar_गाँठ.inputs.push_back(ar_गाँठ.add_dummy(*sync_tensors[i]));
    }

    ar_गाँठ.outputs.सफा();
    क्रम (माप_प्रकार i = 0; i < src_tensors.size(); i++) {
        ar_गाँठ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        यह, dst, rank, n_ranks, false, false, /*is_shard_ok=*/ false,
        (काष्ठा htp_allreduce_kernel_params *) ar_गाँठ.kernel_params);

    ar_गाँठ.name = "ALLREDUCE";
    यह->enqueue_op(ar_गाँठ);
}

ggml_hexagon_shared_बफर * ggml_hexagon_session::mmap_tensor(स्थिर ggml_tensor * t) {
    अगर (!t) वापस nullptr;

    स्वतः sbuf = static_cast<ggml_hexagon_shared_बफर *>(t->बफर->context);
    अगर (!sbuf->mapped) {
        स्थिर bool is_weight = ggml_backend_बफर_get_usage(t->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS;
        स्थिर bool extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    वापस sbuf;
}

bool ggml_hexagon_session::clone_बफर(स्थिर ggml_hexagon_shared_बफर * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    अगर (sbuf->sess == यह) वापस true;

    स्वतः mem = sbuf->mem;
    पूर्णांक   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    अगर (यह->cloned_buffers.find(fd) != यह->cloned_buffers.end()) वापस true;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", यह->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    स्वतः clone = std::make_unique<ggml_hexagon_shared_बफर>(यह, *sbuf);
    प्रयास {
        clone->mmap(sbuf->extended);
    } पकड़ो (स्थिर std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", यह->c_name(), exc.what());
        वापस false;
    }

    यह->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(यह);
    वापस true;
}

व्योम ggml_hexagon_session::release_बफर(स्थिर ggml_hexagon_shared_बफर * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    स्वतः mem = sbuf->mem;
    पूर्णांक   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    स्वतः it = यह->cloned_buffers.find(fd);
    अगर (it != यह->cloned_buffers.end()) {
        स्वतः clone = std::move(it->second);
        यह->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(यह);
}

व्योम ggml_hexagon_session::unclone_बफर(स्थिर ggml_hexagon_shared_बफर * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    स्वतः mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    क्रम (स्वतः * sess : sessions) {
        sess->release_बफर(sbuf);
    }
}

अटल माप_प्रकार ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_बफर *> sbufs;

    स्थिर माप_प्रकार MiB = 1024 * 1024;
    स्थिर माप_प्रकार GiB = MiB  * 1024;

    माप_प्रकार vmem = 0;
    माप_प्रकार step = 256u * MiB;

    प्रयास {
        sbufs.push_back(नया ggml_hexagon_shared_बफर(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(नया ggml_hexagon_shared_बफर(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(नया ggml_hexagon_shared_बफर(sess, GiB, true)); vmem += GiB;

        जबतक (1) {
            sbufs.push_back(नया ggml_hexagon_shared_बफर(sess, step, true));
            vmem += step;
        }
    } पकड़ो (...) { }

    क्रम (स्वतः b : sbufs) { मिटाओ b; }

    वापस vmem - step; // backoff to account for overhead from internal mappings
}

व्योम ggml_hexagon_session::allocate(स्थिर ggml_hexagon_device_config & config) noexcept(false) {
    पूर्णांक phys_idx = config.physical_idx;
    पूर्णांक virt_idx = config.virtual_idx;

    यह->valid_session = false;
    यह->valid_handle  = false;
    यह->valid_queue   = false;
    यह->valid_iface   = false;

    यह->name          = config.name;
    यह->phys_idx      = phys_idx;
    यह->virt_idx      = virt_idx;
    यह->domain_id     = config.domain_id;
    यह->session_id    = 0;
    यह->batch_req_seq = 0;
    यह->batch_rsp_seq = 0;
    यह->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", यह->name.c_str(), यह->domain_id, phys_idx, virt_idx);

    अगर (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        फेंको std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    स्थिर std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    अगर (virt_idx > 0) {
        काष्ठा remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<अक्षर *>(dom_name.c_str());
        n.session_name     = const_cast<अक्षर *>(यह->name.c_str());
        n.session_name_len = यह->name.size();
        n.session_id       = virt_idx;

        पूर्णांक err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (व्योम *) &n, माप(n));
        अगर (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           यह->c_name(), phys_idx, virt_idx, err);
            फेंको std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        यह->session_id    = n.session_id;
        यह->domain_id     = n.effective_domain_id;
        यह->valid_session = true;
    } अन्यथा {
        काष्ठा remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<अक्षर *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        पूर्णांक err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (व्योम *) &eff, माप(eff));
        अगर (err == AEE_SUCCESS) {
            यह->domain_id = eff.effective_domain_id;
        } अन्यथा {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           यह->name.c_str(), err, यह->domain_id);
        }
    }

    // Enable unsigned modules
    {
        काष्ठा remote_rpc_control_unsigned_module u;
        u.domain = यह->domain_id;
        u.enable = 1;
        पूर्णांक err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (व्योम *) &u, माप(u));
        अगर (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", यह->c_name(), err);
            फेंको std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    अक्षर session_uri[256];
    {
        अक्षर htp_uri[256];
        snprintf(htp_uri, माप(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        काष्ठा remote_rpc_get_uri u = {};
        u.session_id      = यह->session_id;
        u.domain_name     = const_cast<अक्षर *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<अक्षर *>(htp_uri);
        u.module_uri_len  = म_माप(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = माप(session_uri);

        पूर्णांक err = remote_session_control(FASTRPC_GET_URI, (व्योम *) &u, माप(u));
        अगर (err != AEE_SUCCESS) {
            snprintf(session_uri, माप(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), यह->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          यह->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    पूर्णांक err = htp_iface_open(session_uri, &यह->handle);
    अगर (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", यह->c_name(), session_uri, err);
        फेंको std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    यह->valid_handle = true;

    // Query HW info and resolve session options
    {
        अचिन्हित पूर्णांक hw_n_threads = 0;
        अचिन्हित पूर्णांक hw_n_hvx     = 0;
        अचिन्हित पूर्णांक hw_n_hmx     = 0;
        अचिन्हित दीर्घ दीर्घ hw_vtcm_आकार = 0;
        पूर्णांक hw_err = htp_iface_hwinfo(यह->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_आकार);
        अगर (hw_err == 0) {
            स्थिर uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            यह->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (माप_प्रकार) max_n_threads) : max_n_threads;
            यह->n_hvx     = यह->n_threads;
            यह->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            यह->vtcm_आकार = (uint64_t)hw_vtcm_आकार;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          यह->c_name(), यह->n_threads, यह->n_hvx, यह->n_hmx,
                          (अचिन्हित दीर्घ दीर्घ)(यह->vtcm_आकार / (1024 * 1024)));
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", यह->c_name(), hw_err);
            स्थिर uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            यह->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (माप_प्रकार) HTP_MAX_NTHREADS) : default_n_threads;
            यह->n_hvx     = यह->n_threads;
            यह->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            यह->vtcm_आकार = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        काष्ठा remote_rpc_control_latency l;
        l.enable = 1;

        पूर्णांक err = remote_handle64_control(यह->handle, DSPRPC_CONTROL_LATENCY, (व्योम *) &l, माप(l));
        अगर (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (अचिन्हित) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", यह->c_name(),
                  यह->session_id, यह->domain_id, session_uri, (अचिन्हित दीर्घ) यह->handle);

    स्थिर माप_प्रकार req_q_आकार = (माप(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    स्थिर माप_प्रकार rsp_q_आकार = (माप(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(यह->domain_id,
                          0,              // Flags
                          req_q_आकार,     // Request  queue size (in bytes)
                          rsp_q_आकार,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (व्योम *) यह,  // Callback context
                          &queue);
    अगर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", यह->name.c_str(), (अचिन्हित) err);
        फेंको std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    यह->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &यह->queue_id);
    अगर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (अचिन्हित) err);
        फेंको std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    अगर (opt_etm) {
        err = htp_iface_etm(यह->handle, 1);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (अचिन्हित) err);
        }
    }

    // Allocate buffers and state for op batching
    यह->op_queue = नया ggml_hexagon_opqueue(यह, opt_opbatch, opt_opqueue);

    यह->fence_buf = नया ggml_hexagon_fence_बफर(यह, &dev_ctx->fence_बफर_type, 64 * 1024);
    अगर (यह->mdev.count > 1) {
        यह->mdev_fence_slot = यह->alloc_fence(यह->mdev.count);
    }

    अगर (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(यह);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", यह->c_name(), opt_vmem);
    }
    स्थिर माप_प्रकार shm_आकार = यह->op_queue->shm_आकार();
    यह->max_vmem = (opt_vmem > shm_आकार) ? (opt_vmem - shm_आकार) : opt_vmem;

    यह->op_batch = नया ggml_hexagon_opbatch(यह, opt_opbatch, यह->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(यह->handle, यह->session_id, यह->queue_id, यह->n_threads, opt_nhmx, यह->max_vmem, (uint32_t) opt_power_mode);
    अगर (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", यह->c_name(), (अचिन्हित) err);
        फेंको std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    यह->valid_iface = true;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", यह->c_name(), htp_power_mode_to_str(opt_power_mode));

    अगर (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(यह->handle, opt_profile, &pmu_conf);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (अचिन्हित) err);
        }
    }
}

व्योम ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", यह->name.c_str());

    यह->mdev.sessions.सफा();

    पूर्णांक err;

    अगर (यह->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(यह->handle);
        अगर (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (अचिन्हित) err);
        }
    }

    मिटाओ यह->op_batch;
    मिटाओ यह->op_queue;
    क्रम (स्वतः & it : यह->cpy_fence_slots) {
        free_fence((व्योम *) it.second, 1);
    }
    यह->cpy_fence_slots.सफा();

    अगर (यह->fence_buf) {
        unclone_बफर(यह->fence_buf);
        मिटाओ यह->fence_buf;
        यह->fence_buf = nullptr;
    }
    जबतक (!यह->cloned_buffers.empty()) {
        release_बफर(यह->cloned_buffers.begin()->second.लो());
    }

    अगर (opt_etm) {
        err = htp_iface_etm(यह->handle, 0);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (अचिन्हित) err);
        }
    }

    अगर (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(यह->handle, 0, &pmu_conf);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (अचिन्हित) err);
        }
    }

    अगर (यह->valid_queue) {
        err = dspqueue_close(queue);
        अगर (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (अचिन्हित) err);
        }
    }

    अगर (यह->valid_handle) {
        htp_iface_close(यह->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(स्थिर ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_गिनती) noexcept(false) {
    यह->dev        = dev;
    यह->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    यह->mdev.idx   = mdev_idx;
    यह->mdev.count = mdev_गिनती > 0 ? mdev_गिनती : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)यह) & 0xFFFF;

    प्रयास {
        allocate(config);
        अगर (यह->mdev.idx == 0 && !config.mdev_group.empty()) {
            क्रम (माप_प्रकार i = 0; i < config.mdev_group.size(); i++) {
                यह->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], यह->dev, (uint32_t) (i + 1), यह->mdev.count));
            }
        }
    } पकड़ो (स्थिर std::exception & exc) {
        release();
        फेंको;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();
}

// ** backend interface

अटल bool ggml_hexagon_flash_attn_is_hmx_eligible(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * q,
    स्थिर काष्ठा ggml_tensor * k,
    स्थिर काष्ठा ggml_tensor * v,
    स्थिर काष्ठा ggml_tensor * sinks
) {
    अगर (sess->n_hmx == 0) {
        वापस false;
    }

    अगर (opt_fa_select < 2) {
        वापस false;
    }

    अगर ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        वापस false;
    }

    स्थिर uint32_t DK = q->ne[0];
    स्थिर uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    अगर (DK % 8 != 0 || DV % 8 != 0) {
        वापस false;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    अगर (opt_fa_select > 2) {
        वापस true;
    }

    स्थिर uint32_t M = q->ne[1];

    // Prefill or batched decode
    अगर (M > 1) {
        वापस true;
    }

    स्थिर uint32_t n_शीर्ष = q->ne[2];
    स्थिर uint32_t n_kv_heads = k->ne[2];
    स्थिर uint32_t G = n_kv_heads > 0 ? n_शीर्ष / n_kv_heads : 1;
    स्थिर uint32_t S = k->ne[1];
    स्थिर uint32_t योग_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    अगर (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < योग_threads))) {
        वापस true;
    }

    // Tile alignment for 32-row HMX tiles
    स्थिर bool is_tile_aligned = (G > 0 && (32 % G == 0));
    अगर (!is_tile_aligned) {
        अगर (sess->n_threads >= 6) {
            वापस false;
        }
        वापस S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    अगर (DK == 256) {
        s_cross = 4096;
    } अन्यथा अगर (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } अन्यथा {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    वापस S >= s_cross;
}

अटल bool ggml_hexagon_precompute_flash_attn_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_fa_kernel_params * kparams
) {
    अगर (opt_fa_select < 1) {
        वापस false;
    }

    स_रखो(kparams, 0, माप(*kparams));

    स्थिर काष्ठा ggml_tensor * q    = op->src[0];
    स्थिर काष्ठा ggml_tensor * k    = op->src[1];
    स्थिर काष्ठा ggml_tensor * v    = op->src[2];
    स्थिर काष्ठा ggml_tensor * mask = op->src[3];

    स्थिर uint32_t neq0 = q->ne[0];  // head_dim (DK)
    स्थिर uint32_t neq1 = q->ne[1];  // n_tokens
    स्थिर uint32_t neq2 = q->ne[2];  // n_heads

    स्थिर uint32_t nek1 = k->ne[1];  // kv_len

    स्थिर uint32_t nev0 = v->ne[0];  // head_dim (DV)

    स्थिर uint32_t DK = neq0;
    स्थिर uint32_t DV = nev0;

    स्थिर uint32_t n_kv_heads = k->ne[2];
    स्थिर uint32_t G          = neq2 / n_kv_heads;

    भग्न scale         = 1.0f;
    भग्न max_bias      = 0.0f;
    भग्न logit_softcap = 0.0f;
    स_नकल(&scale,         &op->op_params[0], माप(भग्न));
    स_नकल(&max_bias,      &op->op_params[1], माप(भग्न));
    स_नकल(&logit_softcap, &op->op_params[2], माप(भग्न));

    अगर (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->शीर्ष_split = (opt_fa_शीर्ष_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    स्थिर uint32_t n_शीर्ष = q->ne[2];
    kparams->n_शीर्ष_log2 = 1u << (uint32_t) std::न्यूनमान(std::log2(n_शीर्ष));
    kparams->m0 = std::घात(2.0f, -(max_bias) / kparams->n_शीर्ष_log2);
    kparams->m1 = std::घात(2.0f, -(max_bias / 2.0f) / kparams->n_शीर्ष_log2);

    // Check HMX eligibility
    स्थिर काष्ठा ggml_tensor * sinks = op->src[4];
    अगर (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        स्थिर uint32_t DK_pad = hex_round_up(DK, 64);
        स्थिर uint32_t DV_pad = hex_round_up(DV, 64);
        माप_प्रकार Br = 0, Bc = 0;
        पूर्णांक ret = hmx_fa_find_chunk_आकार(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_आकार, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_शीर्ष);
        अगर (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            अगर (logit_softcap == 0.0f) {
                kparams->scale = scale * (भग्न) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } अन्यथा {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (भग्न) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_आकार = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_शीर्ष);

            स्थिर माप_प्रकार row_vec_bytes = hex_align_up(Bc * माप(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            स्थिर माप_प्रकार m_पंक्ति_bytes = hex_align_up(Bc * माप(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_पंक्ति_bytes / माप(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_मूल्यावली(G);
            अगर (mask) {
                kparams->src3_div2 = init_fastdiv_मूल्यावली(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_मूल्यावली(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            वापस true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    स्थिर माप_प्रकार आकार_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    स्थिर माप_प्रकार आकार_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    स्थिर माप_प्रकार आकार_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_आकार = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_शीर्ष, sess->n_threads);

    kparams->u.hvx.आकार_q_row_padded = आकार_q_row_padded;
    kparams->u.hvx.आकार_k_row_padded = आकार_k_row_padded;
    kparams->u.hvx.आकार_v_row_padded = आकार_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_मूल्यावली(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_मूल्यावली(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_मूल्यावली(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_मूल्यावली(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_मूल्यावली(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_मूल्यावली(q->ne[3]/v->ne[3]);
    अगर (mask) {
        kparams->src3_div2 = init_fastdiv_मूल्यावली(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_मूल्यावली(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    वापस true;
}

अटल bool ggml_hexagon_supported_flash_attn_ext(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * src3 = op->src[3];
    स्थिर काष्ठा ggml_tensor * src4 = op->src[4];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    अगर ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        वापस false;
    }

    अगर (src3 && src3->type != GGML_TYPE_F16) {  // mask
        वापस false;
    }

    अगर (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        वापस false;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        वापस false;
    }

    काष्ठा htp_fa_kernel_params kparams;
    अगर (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        वापस false;
    }

    अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_आकार, sess->vtcm_आकार);
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_gated_delta_net(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    अगर (opt_gdn_select < 1) {
        वापस false;
    }

    स्थिर काष्ठा ggml_tensor * q     = op->src[0];
    स्थिर काष्ठा ggml_tensor * k     = op->src[1];
    स्थिर काष्ठा ggml_tensor * v     = op->src[2];
    स्थिर काष्ठा ggml_tensor * g     = op->src[3];
    स्थिर काष्ठा ggml_tensor * beta  = op->src[4];
    स्थिर काष्ठा ggml_tensor * state = op->src[5];
    स्थिर काष्ठा ggml_tensor * dst   = op;

    अगर (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        वापस false;
    }

    स्थिर int64_t S_v      = v->ne[0];
    स्थिर int64_t H        = v->ne[1];
    स्थिर int64_t n_tokens = v->ne[2];
    स्थिर int64_t n_seqs   = v->ne[3];
    स्थिर int64_t K        = ggml_get_op_params_i32(op, 0);

    अगर (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        वापस false;
    }
    अगर (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        वापस false;
    }
    अगर ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        वापस false;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    अगर (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        वापस false;
    }
    अगर (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        वापस false;
    }

    स्थिर uint32_t योग_rows = (uint32_t) (H * n_seqs);
    स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, योग_rows);

    स्थिर bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    अगर (can_use_hmx) {
        काष्ठा htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        अगर (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_आकार, योग_rows, sess->vtcm_आकार, n_threads, true, &n_heads_batch)) {
            वापस false;
        }
    } अन्यथा {
        काष्ठा htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        अगर (layout.योग_bytes > sess->vtcm_आकार) {
            वापस false;
        }
    }

    वापस true;
}

अटल bool ggml_hexagon_matmul_is_hmx_eligible(
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    bool is_matmul_id,
    bool is_batched
) {
    GGML_UNUSED(dst);

    अगर (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        वापस false;
    }

    स्थिर पूर्णांक ne00  = src0->ne[0];
    स्थिर पूर्णांक ne01  = src0->ne[1];
    स्थिर पूर्णांक ne11  = src1->ne[1];
    स्थिर पूर्णांक ne12  = src1->ne[2];
    स्थिर पूर्णांक wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    अगर (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        वापस false;
    }

    // HMX supports F16, F32, and repack quantized types.
    अगर (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        वापस false;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    अगर (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        वापस false;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    स्थिर bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    अगर (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        वापस false;
    }
    अगर (!is_matmul_id && is_batched && is_tiled) {
        वापस false;
    }

    // HMX assumes contiguous row-major layout.
    अगर (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        वापस false;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    स्थिर पूर्णांक m = is_matmul_id ? ne12 : ne11;
    अगर (m <= HTP_MM_HMX_MIN_NROWS) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(dst);
}

अटल bool ggml_hexagon_precompute_hmx_mm_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * act,
    स्थिर काष्ठा ggml_tensor * dst,
    पूर्णांक wtype,
    पूर्णांक ne00_padded,
    पूर्णांक ne01_padded,
    पूर्णांक ne02,
    पूर्णांक ne11,
    पूर्णांक ne12,
    पूर्णांक ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    माप_प्रकार src2_आकार,
    माप_प्रकार vtcm_budget,
    काष्ठा htp_mm_kernel_params * kparams
) {
    स्थिर bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिर पूर्णांक aligned_tile_आकार = is_tiled ? htp_tensor_get_tile_padded_आकार(wtype) : 0;
    स्थिर पूर्णांक n_threads = (पूर्णांक)sess->n_threads;
    स्थिर पूर्णांक ne10 = act->ne[0];

    पूर्णांक m_for_solver = ne11;
    पूर्णांक m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    अगर (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        स्थिर bool dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : true;
        स्थिर bool act_row_split = ggml_hexagon_can_row_partition(act);
        अगर (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (पूर्णांक) sess->mdev.count - 1) / (पूर्णांक) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    स्थिर bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(m_for_solver);

    स्थिर bool is_batched_val = is_matmul_id ? false : is_batched;
    स्थिर पूर्णांक group_आकार = (ne02 > 0 ? ne12 / ne02 : 1);

    माप_प्रकार m_chunk = 0;
    माप_प्रकार n_chunk = 0;
    माप_प्रकार vtcm_आकार = 0;
    bool use_grouped = false;
    पूर्णांक act_threads_selected = 0;

    अगर (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_आकार > 1) {
        // Try grouped path first
        अगर (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_आकार, n_threads, pipeline, src2_आकार, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_आकार)) {
            use_grouped = true;
        }
    }

    अगर (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        पूर्णांक m_id_rows = 0;
        अगर (dst && is_matmul_id) {
            स्थिर पूर्णांक n_experts = ne02 > 0 ? ne02 : 1;
            स्थिर माप_प्रकार योग_expert_rows = (माप_प्रकार) dst->ne[1] * dst->ne[2];
            पूर्णांक m_per_expert = (पूर्णांक) ((योग_expert_rows + n_experts - 1) / n_experts);
            अगर (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (पूर्णांक) sess->mdev.count - 1) / (पूर्णांक) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        स्थिर uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        अगर (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_आकार, src2_आकार, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_आकार)) {
            वापस false;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_आकार = is_tiled ? htp_tensor_get_tile_आकार(wtype) : 0;
    kparams->aligned_tile_आकार = aligned_tile_आकार;
    kparams->act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
    kparams->vtcm_आकार = vtcm_आकार;
    kparams->vtcm_src0_आकार = 0;
    kparams->div_n_act_threads = init_fastdiv_मूल्यावली(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_मूल्यावली(ne00_padded);
    kparams->vtcm_act_आकार     = 0;
    kparams->vtcm_bias_आकार    = (int32_t) src2_आकार;
    kparams->vtcm_dst_आकार     = 0;

    अगर (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    वापस true;
}

अटल व्योम ggml_hexagon_precompute_hvx_mm_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    पूर्णांक wtype,
    bool is_tiled,
    पूर्णांक ne02,
    पूर्णांक ne03,
    पूर्णांक ne10,
    पूर्णांक ne11,
    पूर्णांक ne12,
    पूर्णांक ne13,
    bool is_matmul_id,
    स्थिर माप_प्रकार src2_row_आकार,
    माप_प्रकार vtcm_budget,
    काष्ठा htp_mm_kernel_params * kparams
) {
    अगर (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    स्थिर पूर्णांक src1_nrows = ne11 * ne12 * ne13;

    अगर (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_आकार = htp_tensor_get_tile_आकार(wtype);
        kparams->aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);

        स्थिर bool k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        अगर (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (पूर्णांक) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);

            काष्ठा htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            क्रम (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_आकार, 0, d, true, false
                );
                अगर (L.योग_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    अवरोध;
                }
            }
            अगर (L.योग_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                वापस;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_आकार      = L.योग_bytes;
            kparams->vtcm_src0_आकार = L.src0_bytes;
            kparams->vtcm_act_आकार  = L.act_bytes;
            kparams->vtcm_bias_आकार = 0;
            kparams->vtcm_dst_आकार  = L.dst_bytes;
            वापस;
        } अन्यथा {
            bool try_tiled = (k_align && opt_mm_select >= 1);
            अगर (try_tiled) {
                kparams->act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
                अगर (src1_nrows < (पूर्णांक) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } अन्यथा {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                काष्ठा htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                क्रम (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_आकार, d, false, false
                    );
                    अगर (L.योग_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        अवरोध;
                    }
                }

                uint32_t m_chunk = 0;
                अगर (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_आकार, best_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = best_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_आकार      = L.योग_bytes;
                    kparams->vtcm_src0_आकार = L.src0_bytes;
                    kparams->vtcm_act_आकार  = L.act_bytes;
                    kparams->vtcm_bias_आकार = L.bias_bytes;
                    kparams->vtcm_dst_आकार  = L.dst_bytes;
                    वापस;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            वापस;
        }
    } अन्यथा अगर (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    } अन्यथा अगर (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        काष्ठा htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        अगर (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_आकार, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_आकार = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_आकार = L.योग_bytes;
            kparams->vtcm_src0_आकार = L.src0_bytes;
            kparams->vtcm_act_आकार  = L.act_bytes;
            kparams->vtcm_bias_आकार = L.bias_bytes;
            kparams->vtcm_dst_आकार = L.dst_bytes;
            kparams->n_prefetch = 16;
            वापस;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    } अन्यथा अगर (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        काष्ठा htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        अगर (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_आकार, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_आकार = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_आकार = L.योग_bytes;
            kparams->vtcm_src0_आकार = L.src0_bytes;
            kparams->vtcm_act_आकार  = L.act_bytes;
            kparams->vtcm_bias_आकार = L.bias_bytes;
            kparams->vtcm_dst_आकार = L.dst_bytes;
            kparams->n_prefetch = 16;
            वापस;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    } अन्यथा अगर (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        काष्ठा htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        अगर (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_आकार, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_आकार = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_आकार = L.योग_bytes;
            kparams->vtcm_src0_आकार = L.src0_bytes;
            kparams->vtcm_act_आकार  = L.act_bytes;
            kparams->vtcm_bias_आकार = L.bias_bytes;
            kparams->vtcm_dst_आकार = L.dst_bytes;
            kparams->n_prefetch = 16;
            वापस;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    } अन्यथा {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    }
}

अटल व्योम ggml_hexagon_precompute_matmul_params_impl(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    स्थिर माप_प्रकार src2_row_आकार,
    स्थिर माप_प्रकार src2_आकार,
    काष्ठा htp_mm_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर पूर्णांक ne00 = src0->ne[0];
    स्थिर पूर्णांक ne01 = src0->ne[1];
    स्थिर पूर्णांक ne02 = src0->ne[2];
    स्थिर पूर्णांक ne03 = src0->ne[3];

    स्थिर पूर्णांक ne10 = src1->ne[0];
    स्थिर पूर्णांक ne11 = src1->ne[1];
    स्थिर पूर्णांक ne12 = src1->ne[2];
    स्थिर पूर्णांक ne13 = src1->ne[3];

    स्थिर bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिर पूर्णांक wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    स्थिर पूर्णांक ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    स्थिर पूर्णांक ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    स्थिर पूर्णांक ne01_tiled  = hex_round_up(ne01, 32);

    स्थिर bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    स्थिर bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    स्थिर माप_प्रकार vtcm_budget = sess->vtcm_आकार;

    // Check HMX eligibility and try precomputing HMX parameters
    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    अगर (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        अगर (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_आकार, vtcm_budget, kparams)) {
            जाओ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_आकार, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_मूल्यावली(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_मूल्यावली(ne11);
    kparams->div_r2       = init_fastdiv_मूल्यावली(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_मूल्यावली(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_मूल्यावली(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
अटल bool ggml_hexagon_rows_stride(स्थिर int64_t * ne, स्थिर माप_प्रकार * nb, माप_प्रकार * stride) {
    माप_प्रकार s = 0, अग्र = 0;
    क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; i++) {
        अगर (ne[i] == 1) जारी;
        अगर (s == 0) { s = nb[i]; अग्र = s * ne[i]; जारी; }
        अगर (nb[i] != अग्र) वापस false;
        अग्र *= ne[i];
    }
    *stride = s ? s : nb[1];
    वापस true;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
अटल bool ggml_hexagon_matmul_can_collapse(स्थिर काष्ठा ggml_tensor * src0, स्थिर काष्ठा ggml_tensor * src1, स्थिर काष्ठा ggml_tensor * dst) {
    माप_प्रकार s1, sd;
    वापस (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_आकार(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_आकार(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

अटल bool ggml_hexagon_matmul_add_can_collapse(
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * src2,
    स्थिर काष्ठा ggml_tensor * dst
) {
    अगर (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        वापस false;
    }
    अगर (!src2) {
        वापस true;
    }
    स्थिर int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    अगर (src2_nrows == 1) {
        वापस true;
    }
    माप_प्रकार s2;
    वापस src2->nb[0] == ggml_type_आकार(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

अटल ggml_tensor ggml_hexagon_tensor_collapse_rows(स्थिर काष्ठा ggml_tensor * t) {
    माप_प्रकार stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    वापस c;
}

अटल व्योम ggml_hexagon_precompute_matmul_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_mm_kernel_params * kparams
) {
    अगर (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        स्थिर ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        स्थिर ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        वापस;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

अटल व्योम ggml_hexagon_precompute_fused_matmul_add_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * src2,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_mm_kernel_params * kparams
) {
    अगर (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        स्थिर ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        स्थिर ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        स्थिर ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        स्थिर काष्ठा ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        स्थिर माप_प्रकार src2_आकार = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_आकार, kparams);
        kparams->collapse = 1;
        वापस;
    }
    स्थिर माप_प्रकार src2_आकार = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_आकार, kparams);
}

अटल bool ggml_hexagon_precompute_द्विआधारी_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    uint32_t op,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_द्विआधारी_kernel_params * kparams,
    स्थिर काष्ठा ggml_tensor * src2
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर माप_प्रकार elem_आकार = ggml_type_आकार(src0->type);
    स्थिर माप_प्रकार src0_row_आकार = src0->ne[0] * elem_आकार;
    स्थिर माप_प्रकार src1_row_आकार = src1->ne[0] * elem_आकार;
    स्थिर माप_प्रकार dst_row_आकार  = dst->ne[0]  * elem_आकार;

    स्थिर माप_प्रकार src0_row_आकार_aligned = hex_round_up(src0_row_आकार, 128);
    स्थिर माप_प्रकार src1_row_आकार_aligned = hex_round_up(src1_row_आकार, 128);
    स्थिर माप_प्रकार dst_row_आकार_aligned  = hex_round_up(dst_row_आकार,  128);

    अगर (!htp_द्विआधारी_strides_safe(src0, src1, dst, elem_आकार)) {
        वापस false;
    }

    स्थिर bool is_add_id = op == HTP_OP_ADD_ID;
    स्थिर bool is_scalar = !is_add_id && src1->ne[0] == 1;
    स्थिर bool is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    स्थिर bool is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    स्थिर bool is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    स्थिर bool is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    स्थिर bool is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    अगर (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        स्थिर uint32_t योग_elems = (uint32_t) ggml_nelements(src0);
        स्थिर uint32_t n_threads = sess->n_threads;
        स्थिर uint32_t max_chunk_elems = 32768 / elem_आकार;
        स्थिर uint32_t min_chunk_elems = 256;
        स्थिर uint32_t target_chunk_elems = hex_round_up((योग_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        स्थिर uint32_t chunk_आकार = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        स्थिर uint32_t chunk_bytes = hex_round_up(chunk_आकार * elem_आकार, 128);

        kparams->kernel_type           = HTP_द्विआधारी_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_बफर       = 1;
        kparams->src0_row_आकार_aligned = src0_row_आकार_aligned;
        kparams->src1_row_आकार_aligned = is_scalar_broadcast ? 0 : src1_row_आकार_aligned;
        kparams->dst_row_आकार_aligned  = dst_row_आकार_aligned;
        kparams->src1_आकार             = 0;
        kparams->chunk_आकार            = chunk_आकार;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        काष्ठा htp_द्विआधारी_vtcm_layout L;
        htp_द्विआधारी_vtcm_layout_build(&L, kparams, sess->vtcm_आकार);
        अगर (L.योग_bytes == 0 || L.योग_bytes > sess->vtcm_आकार) {
            वापस false;
        }

        kparams->vtcm_आकार = L.योग_bytes;
        वापस true;
    }

    क्रमागत htp_द्विआधारी_kernel_type kernel_type;
    माप_प्रकार src1_आकार = 0;
    माप_प्रकार src2_आकार = 0;

    अगर (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_द्विआधारी_KERNEL_ADD_ID;
        src1_आकार = hex_round_up(src1->ne[1] * src1_row_आकार_aligned, 128);
        src2_आकार = hex_round_up(ggml_nelements(src2) * माप(int32_t), 128);
        अगर (src1_आकार + src2_आकार > sess->vtcm_आकार) {
            वापस false;
        }
    } अन्यथा अगर (is_row_bcast) {
        kernel_type = HTP_द्विआधारी_KERNEL_ROW_BCAST;
        src1_आकार = src1_row_आकार_aligned;
    } अन्यथा अगर (is_scalar) {
        src1_आकार = hex_round_up(ggml_nelements(src1) * elem_आकार, 128);
        अगर (src1_आकार > sess->vtcm_आकार) {
            वापस false;
        }
        kernel_type = HTP_द्विआधारी_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा अगर (is_same_shape) {
        kernel_type = HTP_द्विआधारी_KERNEL_SAME_SHAPE;
    } अन्यथा अगर (is_complex) {
        src1_आकार = hex_round_up(ggml_nelements(src1) * elem_आकार + 128, 128);
        अगर (src1_आकार > sess->vtcm_आकार) {
            वापस false;
        }
        kernel_type = HTP_द्विआधारी_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } अन्यथा {
        src1_आकार = hex_round_up(ggml_nelements(src1) * elem_आकार + 128, 128);
        अगर (src1_आकार > sess->vtcm_आकार) {
            वापस false;
        }
        kernel_type = HTP_द्विआधारी_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_आकार_aligned = src0_row_आकार_aligned;
    kparams->src1_row_आकार_aligned = src1_row_आकार_aligned;
    kparams->dst_row_आकार_aligned  = dst_row_आकार_aligned;
    kparams->src1_आकार             = src1_आकार;
    kparams->src2_आकार             = src2_आकार;

    स्थिर bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    स्थिर bool src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    स्थिर bool dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    स्थिर bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    स्थिर bool src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    स्थिर bool dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    bool split_at_ne01 = false;
    अगर (src0->ne[2] > 1) {
        अगर (kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } अन्यथा अगर (kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } अन्यथा {
            split_at_ne01 = true;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    bool split_at_ne02 = false;
    अगर (src0->ne[3] > 1) {
        अगर (kernel_type == HTP_द्विआधारी_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } अन्यथा अगर (kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } अन्यथा {
            split_at_ne02 = true;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    काष्ठा htp_द्विआधारी_vtcm_layout L;
    htp_द्विआधारी_vtcm_layout_build(&L, kparams, sess->vtcm_आकार);
    अगर (L.rows_per_बफर == 0 || L.योग_bytes > sess->vtcm_आकार) {
        अगर (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            स्थिर uint32_t योग_elems = (uint32_t) ggml_nelements(src0);
            स्थिर uint32_t n_threads = sess->n_threads;
            स्थिर uint32_t max_chunk_elems = 32768 / elem_आकार;
            स्थिर uint32_t min_chunk_elems = 256;
            स्थिर uint32_t target_chunk_elems = hex_round_up((योग_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            स्थिर uint32_t chunk_आकार = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            स्थिर uint32_t chunk_bytes = hex_round_up(chunk_आकार * elem_आकार, 128);

            kparams->kernel_type           = HTP_द्विआधारी_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_बफर       = 1;
            kparams->src1_row_आकार_aligned = is_scalar_broadcast ? 0 : src1_row_आकार_aligned;
            kparams->src1_आकार             = 0;
            kparams->chunk_आकार            = chunk_आकार;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_द्विआधारी_vtcm_layout_build(&L, kparams, sess->vtcm_आकार);
            अगर (L.योग_bytes == 0 || L.योग_bytes > sess->vtcm_आकार) {
                वापस false;
            }

            kparams->vtcm_आकार = L.योग_bytes;
            वापस true;
        }
        वापस false;
    }

    kparams->rows_per_बफर = L.rows_per_बफर;
    kparams->vtcm_आकार       = L.योग_bytes;

    वापस true;
}

अटल व्योम ggml_hexagon_precompute_unary_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    uint32_t op,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_unary_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    स्थिर माप_प्रकार elem_आकार = ggml_type_आकार(src0->type);

    स्थिर माप_प्रकार src0_तथ्य_row_आकार = src0->ne[0] * elem_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = dst->ne[0]  * ggml_type_आकार(dst->type);

    स्थिर माप_प्रकार src0_row_आकार_aligned = hex_round_up(src0_तथ्य_row_आकार, 128);
    स्थिर माप_प्रकार dst_row_आकार_aligned  = hex_round_up(dst_तथ्य_row_आकार,  128);

    kparams->src0_row_आकार_aligned = src0_row_आकार_aligned;
    kparams->dst_row_आकार_aligned  = dst_row_आकार_aligned;

    माप_प्रकार src1_तथ्य_row_आकार = 0;
    माप_प्रकार src1_row_आकार_aligned = 0;
    bool broadcast_weight = false;

    अगर (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_तथ्य_row_आकार = src1->ne[0] * ggml_type_आकार(src1->type);
        src1_row_आकार_aligned = hex_round_up(src1_तथ्य_row_आकार, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_आकार_aligned = src1_row_आकार_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    काष्ठा htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                broadcast_weight, n_threads, sess->vtcm_आकार, elem_आकार,
                                &col_tile, &vtcm_row_per_thread);

    kparams->col_tile = col_tile;
    kparams->vtcm_row_per_thread = vtcm_row_per_thread;
    kparams->vtcm_आकार = L.योग_bytes;

    kparams->vtcm_src0_आकार_per_thread = L.src0_bytes;
    kparams->vtcm_src1_आकार_per_thread = L.src1_bytes;
    kparams->vtcm_dst_आकार_per_thread  = L.dst_bytes;

    kparams->vtcm_src0_आकार = L.src0_bytes * n_threads;
    kparams->vtcm_src1_आकार = L.src1_bytes * n_threads;
    kparams->vtcm_dst_आकार  = L.dst_bytes * n_threads;

    kparams->block = col_tile ? 0 : ((L.src0_bytes / 2) / src0_row_आकार_aligned);

    स्थिर uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_मूल्यावली(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_मूल्यावली(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_मूल्यावली(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_मूल्यावली(tiles_per_row);
}

अटल व्योम ggml_hexagon_precompute_get_rows_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_get_rows_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t ne02 = src0->ne[2];
    स्थिर uint32_t ne03 = src0->ne[3];

    स्थिर uint32_t ne10 = src1->ne[0];
    स्थिर uint32_t ne11 = src1->ne[1];
    स्थिर uint32_t ne12 = src1->ne[2];
    स्थिर uint32_t nr = ne10 * ne11 * ne12;

    स्थिर bool tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिर पूर्णांक wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    अगर (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } अन्यथा अगर (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } अन्यथा {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    स्थिर uint32_t योग_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, योग_tasks);

    काष्ठा htp_get_rows_vtcm_layout vtcm_layout = {};
    जबतक (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, योग_tasks, src1->type, kparams->n_threads);
        अगर (vtcm_layout.योग_bytes <= sess->vtcm_आकार) {
            अवरोध;
        }
        --kparams->n_threads;
    }

    अगर (kparams->n_threads == 0 && योग_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, योग_tasks, src1->type, 1);
    }

    kparams->vtcm_आकार = (योग_tasks == 0) ? 0 : vtcm_layout.योग_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (योग_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->योग_tasks = योग_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_मूल्यावली(ne10) : fastdiv_मूल्यावली{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_मूल्यावली(ne10 * ne11) : fastdiv_मूल्यावली{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_मूल्यावली(ne02) : fastdiv_मूल्यावली{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_मूल्यावली(ne03) : fastdiv_मूल्यावली{0, 0};
}

अटल व्योम ggml_hexagon_precompute_set_rows_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0, //मूल्यावली
    स्थिर काष्ठा ggml_tensor * src1, // indices
    स्थिर काष्ठा ggml_tensor * dst,  // destination
    काष्ठा htp_set_rows_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->योग_tasks = nr;

    kparams->div_ne11 = init_fastdiv_मूल्यावली(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_मूल्यावली(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_मूल्यावली(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_मूल्यावली(src0->ne[2]);

    काष्ठा htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_आकार = vtcm_layout.योग_bytes;
}

अटल व्योम ggml_hexagon_precompute_softmax_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_softmax_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    भग्न scale = 1.0f;
    भग्न max_bias = 0.0f;
    स_नकल(&scale,    &op->op_params[0], माप(भग्न));
    स_नकल(&max_bias, &op->op_params[1], माप(भग्न));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    स्थिर uint32_t n_शीर्ष = src0->ne[2];
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) न्यूनमान(log2(n_शीर्ष));
    kparams->n_शीर्ष      = n_शीर्ष;
    kparams->n_शीर्ष_log2 = n_शीर्ष_log2;

    अगर (max_bias > 0.0f && n_शीर्ष_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_शीर्ष_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);
    } अन्यथा {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t ne10 = src1 ? src1->ne[0] : 1;

    काष्ठा htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_threads);

    kparams->n_threads                 = n_threads;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_per_thread     = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_आकार                 = (uint32_t) layout.योग_bytes;
    kparams->vtcm_src0_आकार_per_thread = (uint32_t) layout.src0_bytes_per_thread;
    kparams->vtcm_src1_आकार_per_thread = (uint32_t) layout.src1_bytes_per_thread;
    kparams->vtcm_dst_आकार_per_thread  = (uint32_t) layout.dst_bytes_per_thread;
    kparams->src0_row_आकार_aligned     = (uint32_t) layout.src0_spad_half_आकार;
    kparams->src1_row_आकार_aligned     = (uint32_t) layout.src1_spad_half_आकार;
    kparams->dst_row_आकार_aligned      = (uint32_t) layout.dst_spad_half_आकार;
    kparams->src0_spad_half_आकार       = (uint32_t) layout.src0_spad_half_आकार;
    kparams->src1_spad_half_आकार       = (uint32_t) layout.src1_spad_half_आकार;
    kparams->dst_spad_half_आकार        = (uint32_t) layout.dst_spad_half_आकार;
    अगर (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } अन्यथा अगर (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } अन्यथा {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    अगर (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_मूल्यावली(src0->ne[1]);
    अगर (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_मूल्यावली(src0->ne[2]);
    स्थिर uint32_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिर uint32_t ne13 = src1 ? src1->ne[3] : 1;
    अगर (ne12 > 0) kparams->div_ne12 = init_fastdiv_मूल्यावली(ne12);
    अगर (ne13 > 0) kparams->div_ne13 = init_fastdiv_मूल्यावली(ne13);
}

अटल व्योम ggml_hexagon_precompute_rope_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_rope_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    स्थिर uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    काष्ठा htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);

    kparams->n_threads              = n_threads;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_per_thread  = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_आकार              = (uint32_t) layout.योग_bytes;
    kparams->spad_per_thread        = (uint32_t) layout.bytes_per_thread;
    kparams->theta_cache_offset     = (uint32_t) layout.theta_cache_आकार_aligned;
    kparams->src0_row_आकार_aligned  = (uint32_t) layout.src0_row_आकार_aligned;
    kparams->freq_factors_offset    = (uint32_t) (layout.bytes_per_thread * n_threads);
    kparams->freq_factors_आकार      = (uint32_t) layout.freq_factors_आकार_aligned;

    अगर (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_मूल्यावली(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_मूल्यावली(dst->ne[1]);
    }
}

अटल व्योम ggml_hexagon_precompute_ssm_conv_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_ssm_conv_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर uint32_t d_conv  = (uint32_t) src1->ne[0];
    स्थिर uint32_t d_inner = (uint32_t) src0->ne[1];
    स्थिर uint32_t n_t     = (uint32_t) dst->ne[1];
    स्थिर uint32_t n_s     = (uint32_t) dst->ne[2];
    स्थिर uint32_t ncs     = (uint32_t) src0->ne[0];

    स्थिर uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    स्थिर uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    स्थिर uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    स्थिर uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * माप(भग्न), 128) + 128;
    स्थिर uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * माप(भग्न), 128);
    स्थिर uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    अगर (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        स्थिर uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * माप(भग्न), 128);
        स्थिर uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * माप(भग्न), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        स्थिर uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * माप(भग्न), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } अन्यथा {
        स्थिर माप_प्रकार vtcm_budget = (sess->vtcm_आकार > 0 ? sess->vtcm_आकार / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        स्थिर uint32_t src0_block_T    = hex_round_up(ncs * 32 * माप(भग्न), 128);
        स्थिर माप_प्रकार   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        स्थिर माप_प्रकार   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * माप(भग्न)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        अगर (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        अगर (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        अगर (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        स्थिर uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * माप(भग्न), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * माप(भग्न), 128);
    }

    kparams->vtcm_src0_आकार_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_आकार_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_आकार_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_आकार = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_आकार = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_आकार  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_आकार      = kparams->vtcm_src0_आकार + kparams->vtcm_src1_आकार + kparams->vtcm_dst_आकार;
}

अटल व्योम ggml_hexagon_precompute_gated_delta_net_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_gdn_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर काष्ठा ggml_tensor * q     = op->src[0];
    स्थिर काष्ठा ggml_tensor * k     = op->src[1];
    स्थिर काष्ठा ggml_tensor * v     = op->src[2];
    स्थिर काष्ठा ggml_tensor * g     = op->src[3];
    स्थिर काष्ठा ggml_tensor * state = op->src[5];

    स्थिर uint32_t S_v      = (uint32_t) v->ne[0];
    स्थिर uint32_t H        = (uint32_t) v->ne[1];
    स्थिर uint32_t n_tokens = (uint32_t) v->ne[2];
    स्थिर uint32_t n_seqs   = (uint32_t) v->ne[3];
    स्थिर uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    स्थिर uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    स्थिर uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    स्थिर uint32_t योग_rows = H * n_seqs;
    स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, योग_rows);

    स्थिर bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    काष्ठा htp_gdn_hmx_vtcm_layout hmx_layout;
    काष्ठा htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    अगर (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_आकार, योग_rows, sess->vtcm_आकार, n_threads, true, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_आकार      = HTP_GDN_CHUNK_आकार;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_आकार - 1) / HTP_GDN_CHUNK_आकार;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_आकार       = (uint32_t) hmx_layout.योग_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.योग_bytes / (n_threads > 0 ? n_threads : 1));
    } अन्यथा {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_threads);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_per_thread = (uint32_t) hvx_layout.bytes_per_thread;
        kparams->vtcm_आकार       = (uint32_t) hvx_layout.योग_bytes;
    }

    kparams->n_threads           = n_threads;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_tokens            = n_tokens;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->योग_rows          = योग_rows;
    kparams->rows_per_thread     = (योग_rows + kparams->n_threads - 1) / kparams->n_threads;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / माप(भग्न));
    kparams->state_आकार_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((भग्न) S_v);

    अगर (H > 0)                  kparams->div_H         = init_fastdiv_मूल्यावली(H);
    अगर (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_मूल्यावली((uint32_t) q->ne[1]);
    अगर (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_मूल्यावली((uint32_t) k->ne[1]);
    अगर (rq3 > 0)                kparams->div_rq3       = init_fastdiv_मूल्यावली(rq3);
    अगर (rk3 > 0)                kparams->div_rk3       = init_fastdiv_मूल्यावली(rk3);
    अगर (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_मूल्यावली(kparams->n_threads);
}

अटल व्योम ggml_hexagon_precompute_sort_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    bool is_शीर्ष_k,
    काष्ठा htp_sort_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    स्थिर uint32_t योग_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t ne00       = src0->ne[0];
    स्थिर uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    अगर (!is_शीर्ष_k) {
        order = ((स्थिर int32_t *) op->op_params)[0];
    }

    स्थिर uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिर माप_प्रकार vtcm_budget = sess->vtcm_आकार > 0 ? sess->vtcm_आकार : (8 * 1024 * 1024);

    काष्ठा htp_sort_vtcm_layout layout;
    bool ok = htp_sort_solve_layout(&layout, ne00, योग_rows, k, n_threads_max, vtcm_budget, is_शीर्ष_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->योग_rows        = (int32_t) योग_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) योग_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_शीर्ष_k          = is_शीर्ष_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_आकार         = (int32_t) layout.योग_bytes;
    kparams->phase1_slot_आकार  = (int32_t) layout.phase1_slot_आकार;
    kparams->merge_मूल्यावली_off  = (int32_t) layout.merge_मूल्यावली_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

अटल व्योम ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    स्थिर uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    स्थिर int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    अगर (numer_hi < 0) {
        *ox_hi = 0;
    } अन्यथा {
        स्थिर uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    अगर (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

अटल bool ggml_hexagon_pool_shape_fits(
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * dst
) {
    वापस ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

अटल व्योम ggml_hexagon_precompute_pool_2d_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * dst,
    काष्ठा htp_pool_2d_kernel_params * kparams,
    bool is_pool_1d
) {
    स_रखो(kparams, 0, माप(*kparams));

    स्थिर uint32_t src_x = (uint32_t) src0->ne[0];
    स्थिर uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    स्थिर uint32_t dst_x = (uint32_t) dst->ne[0];
    स्थिर uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    स्थिर uint32_t src_plane_bytes = src_x * src_y * माप(भग्न);
    स्थिर uint32_t dst_plane_bytes = dst_x * dst_y * माप(भग्न);
    स्थिर uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    स्थिर uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    काष्ठा htp_pool_vtcm_layout layout;
    स्थिर bool ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_आकार);
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
    kparams->src_plane_bytes_aligned = (uint32_t) layout.src_spad_half_आकार;
    kparams->dst_plane_bytes_aligned = (uint32_t) layout.dst_spad_half_आकार;
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
    kparams->avg_divide_गिनती = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    स्थिर bool narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (भग्न) (kparams->kernel_x * kparams->kernel_y);
}

अटल bool ggml_hexagon_precompute_concat_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_concat_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!src0 || !src1 || !dst) {
        वापस false;
    }

    पूर्णांक dim = ((स्थिर int32_t *) op->op_params)[0];
    अगर (dim < 0 || dim >= GGML_MAX_DIMS) {
        वापस false;
    }
    kparams->dim = dim;

    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        वापस false;
    }
    अगर (src0->type != dst->type || src1->type != dst->type) {
        वापस false;
    }

    स्थिर uint32_t type_आकार = ggml_type_आकार(dst->type);

    क्रम (पूर्णांक d = 0; d < GGML_MAX_DIMS; d++) {
        स्थिर int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        अगर (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            वापस false;
        }
    }

    स्थिर uint32_t max_gather_ne0 = 128 / type_आकार;
    स्थिर uint32_t योग_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    स्थिर bool b_dense = (src1->nb[0] == type_आकार);
    स्थिर bool gather_strides_ok =
        src0->nb[1] >= (माप_प्रकार) src0->ne[0] * type_आकार &&
        (b_dense ? src1->nb[1] >= (माप_प्रकार) src1->ne[0] * type_आकार
                 : src1->nb[0] >= (माप_प्रकार) src1->ne[1] * type_आकार);

    अगर (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_आकार && dst->nb[1] == (माप_प्रकार) dst->ne[0] * type_आकार &&
        src0->nb[0] == type_आकार &&
        (src1->nb[0] == type_आकार || src1->nb[1] == type_आकार) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        अगर (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_आकार, योग_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_आकार)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_आकार   = kparams->u.gather.slice * sess->n_threads;
            वापस true;
        }
    }

    स्थिर bool dma_strides_ok = (src0->nb[0] == type_आकार && src1->nb[0] == type_आकार && dst->nb[0] == type_आकार);

    अगर (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        वापस true;
    }

    स्थिर bool is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    स्थिर bool is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    स्थिर bool transposed_rows_ok = (src0->nb[0] == type_आकार && src1->nb[1] == type_आकार && dst->nb[0] == type_आकार);

    अगर (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        स्थिर uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        काष्ठा htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_आकार, n_threads);

        अगर (sess->vtcm_आकार > 0 && layout.योग_bytes > sess->vtcm_आकार) {
            वापस false;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_आकार                         = layout.योग_bytes;
        kparams->u.transposed.spad0_आकार_per_thread = layout.src0_spad_आकार_per_thread;
        kparams->u.transposed.spad1_आकार_per_thread = layout.src1_spad_आकार_per_thread;
        वापस true;
    }

    वापस false;
}

अटल bool ggml_hexagon_precompute_cpy_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_copy_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!src0 || !dst) {
        वापस false;
    }

    अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        वापस false;
    }
    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        वापस false;
    }

    स्थिर int64_t nelem_src = ggml_nelements(src0);
    स्थिर int64_t nelem_dst = ggml_nelements(dst);
    अगर (nelem_src != nelem_dst || nelem_src < 0) {
        वापस false;
    }

    स्थिर uint32_t src_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    kparams->src0_type_आकार = (uint8_t) src_type_आकार;
    kparams->dst_type_आकार  = (uint8_t) dst_type_आकार;
    kparams->योग_elems    = (uint32_t) nelem_src;

    अगर (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        वापस true;
    }

    अगर (nelem_src == 1) {
        अगर (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            वापस true;
        }
        अगर ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            वापस true;
        }
        वापस false;
    }

    स्थिर bool sametype = (src0->type == dst->type);

    bool same_extents = true;
    क्रम (पूर्णांक d = 0; d < GGML_MAX_DIMS; d++) {
        अगर (src0->ne[d] != dst->ne[d]) {
            same_extents = false;
            अवरोध;
        }
    }

    स्थिर bool transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_आकार) || (dst->nb[0] != dst_type_आकार) ||
                            (src0->nb[1] < (माप_प्रकार) src0->ne[0] * src_type_आकार) || (dst->nb[1] < (माप_प्रकार) dst->ne[0] * dst_type_आकार);
    स्थिर bool sameshape  = same_extents && !transposed;

    स्थिर bool src_is_contiguous = ggml_is_contiguous(src0);
    स्थिर bool dst_is_contiguous = ggml_is_contiguous(dst);

    अगर (sametype) {
        अगर (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            वापस true;
        }

        स्थिर uint32_t max_gather_ne0 = 128 / src_type_आकार;
        स्थिर uint32_t योग_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        अगर ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_आकार && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (माप_प्रकार) src0->ne[0] * src_type_आकार &&
            योग_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            अगर (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, false,
                                             src_type_आकार, योग_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_आकार)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_आकार   = kparams->u.gather.slice * sess->n_threads;
                वापस true;
            }
        }

        अगर (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->योग_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            वापस true;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_मूल्यावली((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_मूल्यावली((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_मूल्यावली((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_मूल्यावली((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_मूल्यावली((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_मूल्यावली((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        वापस true;
    }

    अगर (!sameshape) {
        वापस false;
    }

    स्थिर bool valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    अगर (!valid_conversion) {
        वापस false;
    }

    स्थिर uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    काष्ठा htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_आकार, dst_type_आकार, n_threads);

    अगर (sess->vtcm_आकार > 0 && layout.योग_bytes > sess->vtcm_आकार) {
        वापस false;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->योग_rows              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_threads               = (uint8_t) n_threads;
    kparams->vtcm_आकार               = layout.योग_bytes;
    kparams->u.convert.src0_buf_आकार = layout.src0_buf_आकार;
    kparams->u.convert.dst_buf_आकार  = layout.dst_buf_आकार;
    kparams->u.convert.spad0_आकार_per_thread = layout.spad0_आकार_per_thread;
    kparams->u.convert.spad1_आकार_per_thread = layout.spad1_आकार_per_thread;
    kparams->u.convert.src0_row_stride = layout.src0_row_stride;
    kparams->u.convert.dst_row_stride  = layout.dst_row_stride;
    kparams->u.convert.blk_rows        = layout.blk_rows;
    kparams->u.convert.div_ne01      = init_fastdiv_मूल्यावली((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_मूल्यावली((uint32_t) (src0->ne[2] * src0->ne[1]));

    वापस true;
}

अटल व्योम ggml_hexagon_precompute_fused_mmnx_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0, // W0
    स्थिर काष्ठा ggml_tensor * act,  // x
    int32_t n_weights,
    काष्ठा htp_mm_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->n_threads = sess->n_threads;

    स्थिर पूर्णांक ne00 = src0->ne[0];
    स्थिर पूर्णांक ne01 = src0->ne[1];
    स्थिर पूर्णांक ne02 = src0->ne[2];
    स्थिर पूर्णांक ne03 = src0->ne[3];

    स्थिर पूर्णांक ne10 = act->ne[0];
    स्थिर पूर्णांक ne11 = act->ne[1];
    स्थिर पूर्णांक ne12 = act->ne[2];
    स्थिर पूर्णांक ne13 = act->ne[3];

    स्थिर bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    स्थिर पूर्णांक wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    स्थिर पूर्णांक ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    स्थिर पूर्णांक ne11_padded = hex_round_up(ne11, 32);
    स्थिर पूर्णांक ne01_tiled  = hex_round_up(ne01, 32);

    स्थिर माप_प्रकार vtcm_budget = sess->vtcm_आकार;
    स्थिर bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    अगर (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, false, is_batched)) {
        अगर (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            जाओ finalize;
        }
    }

    अगर (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        वापस;
    }

    {
        स्थिर पूर्णांक act_nrows = ne11 * ne12 * ne13;
        स्थिर माप_प्रकार act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
        स्थिर माप_प्रकार src0_row_आकार = src0->nb[1];

        स्थिर uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        क्रम (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            काष्ठा htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_आकार, act_row_आकार, 0, d, false, true
            );
            अगर (L.योग_bytes <= sess->vtcm_आकार) {
                best_n_prefetch = d;
                अवरोध;
            }
        }

        काष्ठा htp_mm_hvx_vtcm_layout L;
        bool try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_आकार, act_row_आकार, 0, best_n_prefetch, false, true
        );

        अगर (try_tiled && L.योग_bytes <= sess->vtcm_आकार) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->tile_आकार         = htp_tensor_get_tile_आकार(wtype);
            kparams->aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
            kparams->act_row_आकार      = act_row_आकार;
            kparams->vtcm_src0_आकार    = L.src0_bytes;
            kparams->vtcm_act_आकार     = L.act_bytes;
            kparams->vtcm_bias_आकार    = 0;
            kparams->vtcm_dst_आकार     = L.dst_bytes;
            kparams->vtcm_आकार         = L.योग_bytes;
            kparams->n_prefetch        = best_n_prefetch;
            kparams->n_weights         = n_weights;
        } अन्यथा {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            वापस;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_मूल्यावली(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_मूल्यावली(ne11);
    kparams->div_r2       = init_fastdiv_मूल्यावली(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_मूल्यावली(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_मूल्यावली(ne12);
}

अटल व्योम ggml_hexagon_precompute_fused_mmidnx_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * src0, // W0
    स्थिर काष्ठा ggml_tensor * act,  // x
    स्थिर काष्ठा ggml_tensor * dst,  // dst0
    int32_t n_weights,
    काष्ठा htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

अटल bool ggml_hexagon_op_mul_mat_use_fwht(स्थिर काष्ठा ggml_tensor * dst) {
    अगर (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        वापस false;
    }

    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    स्थिर int64_t n = src1->ne[0];
    अगर (n < 32 || (n & (n - 1)) != 0) {
        वापस false;
    }

    वापस src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

अटल bool ggml_hexagon_supported_mul_mat(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * dst) {
    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    अगर (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        अगर (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            वापस false;
        }
        काष्ठा htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        वापस ukparams.vtcm_row_per_thread > 0 && (माप_प्रकार) ukparams.vtcm_आकार <= sess->vtcm_आकार;
    }

    अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        वापस false;
    }

    चयन (src0->type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q2_K: {
            अगर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                वापस false;
            }

            स्थिर int64_t qk = ggml_blck_आकार(src0->type);
            अगर (src0->ne[0] % qk) {
                वापस false;
            }

            अगर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                वापस false;
            }
            अगर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                वापस false;
            }
            अवरोध;
        }

        हाल GGML_TYPE_F16:
            अगर (src0->nb[1] < src0->nb[0]) {
                वापस false;
            }
            अगर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                वापस false;
            }
            अगर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                वापस false;
            }
            अवरोध;

        हाल GGML_TYPE_F32:
            अगर (src1->type != GGML_TYPE_F32) {
                वापस false;
            }
            अगर (src0->nb[1] < src0->nb[0]) {
                वापस false;
            }
            अगर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                वापस false;
            }
            अगर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                वापस false;
            }
            अवरोध;

        हाल GGML_TYPE_BF16:
            अगर (src0->nb[1] < src0->nb[0]) {
                वापस false;
            }
            अगर (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                वापस false;
            }
            अगर (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                वापस false;
            }
            अवरोध;

        शेष:
            वापस false;
    }

    काष्ठा htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    अगर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_mul_mat_id(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        वापस false;
    }

    अगर (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        वापस false;
    }

    चयन (src0->type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q2_K:
            अगर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                वापस false;
            }

            अगर (src0->ne[0] % ggml_blck_आकार(src0->type) != 0) {
                वापस false;
            }
            अवरोध;

        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_BF16:
            अगर (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                वापस false;
            }
            अगर (src0->ne[0] % 32 != 0) {
                वापस false;
            }
            अवरोध;

        शेष:
            वापस false;
    }

    काष्ठा htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    अगर (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_आकार, sess->vtcm_आकार);
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_द्विआधारी(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type == GGML_TYPE_F32) {
        अगर (src1->type != GGML_TYPE_F32) {
            वापस false;
        }
        अगर (dst->type != GGML_TYPE_F32) {
            वापस false;
        }
    }
    अन्यथा अगर (src0->type == GGML_TYPE_F16) {
        अगर (src1->type != GGML_TYPE_F16) {
            वापस false;
        }
        अगर (dst->type != GGML_TYPE_F16) {
            वापस false;
        }
    }
    अन्यथा {
        वापस false;
    }

    अगर (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        वापस false;
    }
    अगर (!ggml_are_same_shape(src0, dst)) {
        वापस false;
    }
    अगर (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        वापस false;
    }

    काष्ठा htp_द्विआधारी_kernel_params kparams;
    वापस ggml_hexagon_precompute_द्विआधारी_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

अटल bool ggml_hexagon_supported_add_id(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!src2) {
        वापस false;
    }
    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        वापस false;
    }
    अगर (!ggml_are_same_shape(src0, dst)) {
        वापस false;
    }
    अगर (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        वापस false;
    }
    अगर (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        वापस false;
    }
    अगर (src0->nb[0] != माप(भग्न) || src1->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न) || src2->nb[0] != माप(int32_t)) {
        वापस false;
    }

    // REVISIT: add support for non-contiguous tensors
    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        वापस false;
    }
    अगर (src2->nb[1] < src2->ne[0] * माप(int32_t)) {
        वापस false;
    }

    काष्ठा htp_द्विआधारी_kernel_params kparams;
    वापस ggml_hexagon_precompute_द्विआधारी_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

अटल bool ggml_hexagon_supported_unary(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        वापस false;
    }
    अगर (dst->type != src0->type) {
        वापस false;
    }
    अगर (!ggml_is_contiguous_rows(src0)) {
        वापस false;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    अगर (src0->type == GGML_TYPE_F16) {
        चयन (op->op) {
            हाल GGML_OP_NORM:
            हाल GGML_OP_RMS_NORM:
            हाल GGML_OP_L2_NORM:
            हाल GGML_OP_SCALE:
            हाल GGML_OP_CLAMP:
            हाल GGML_OP_SQR:
            हाल GGML_OP_SQRT:
            हाल GGML_OP_LOG:
                अवरोध;
            हाल GGML_OP_UNARY:
                अगर (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    वापस false;
                }
                अवरोध;
            शेष:
                वापस false;
        }
    }

    अगर (!ggml_are_same_shape(src0, dst)) {
        वापस false;
    }

    // dst must be contiguous; src0 may be non-contiguous
    अगर (!ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_योग(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_योग_rows(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    // TODO: add support for non-contigiuos tensors
    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_argmax(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (dst->type != GGML_TYPE_I32) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_pool_2d(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर int32_t * params = op->op_params;

    अगर (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        वापस false;
    }
    अगर (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        वापस false;
    }
    अगर (!ggml_hexagon_pool_shape_fits(src0, op)) {
        वापस false;
    }

    स्थिर int32_t kernel_x = params[1];
    स्थिर int32_t kernel_y = params[2];
    स्थिर int32_t stride_x = params[3];
    स्थिर int32_t stride_y = params[4];
    स्थिर int32_t pad_x    = params[5];
    स्थिर int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    अगर (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        वापस false;
    }

    स्थिर uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    स्थिर uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    काष्ठा htp_pool_vtcm_layout layout;
    वापस htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_आकार);
}

अटल bool ggml_hexagon_supported_pool_1d(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर int32_t * params = op->op_params;

    अगर (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        वापस false;
    }
    अगर (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        वापस false;
    }
    अगर (!ggml_hexagon_pool_shape_fits(src0, op)) {
        वापस false;
    }

    स्थिर int32_t kernel = params[1];
    स्थिर int32_t stride = params[2];
    स्थिर int32_t pad    = params[3];
    अगर (kernel <= 0 || stride <= 0 || pad < 0) {
        वापस false;
    }

    स्थिर uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    स्थिर uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    काष्ठा htp_pool_vtcm_layout layout;
    वापस htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_आकार);
}

अटल bool ggml_hexagon_supported_activations(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        वापस false;
    }
    अगर (dst->type != src0->type) {
        वापस false;
    }

    अगर (!ggml_is_contiguous_1(src0)) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(dst)) {
        वापस false;
    }

    अगर (src1) {
        अगर (src1->type != src0->type) {
            वापस false;
        }
        अगर (!ggml_are_same_shape(src0, src1)) {
            वापस false;
        }
        अगर (!ggml_is_contiguous_1(src1)) {
            वापस false;
        }
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_softmax(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src2) {
        वापस false;  // FIXME: add support for sinks
    }

    अगर (src0->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (src0->ne[2] > 512) {
        वापस false;
    }

    अगर (src1) {
        अगर (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            वापस false;
        }
        अगर (src0->ne[0] != src1->ne[0]) {
            वापस false;
        }
        अगर (src1->ne[1] < src0->ne[1]) {
            वापस false;
        }
        अगर (src0->ne[2] % src1->ne[2] != 0) {
            वापस false;
        }
        अगर (src0->ne[3] % src1->ne[3] != 0) {
            वापस false;
        }
    }

    अगर (src1) {
        अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            वापस false;
        }
    } अन्यथा {
        अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            वापस false;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    स्थिर int64_t ne0 = src0->ne[0];
    अगर (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        वापस false;
    }

    // HVX vector size constraints for softmax
    #घोषणा SOFTMAX_MAX_ROW_आकार 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    अगर (ne0 > SOFTMAX_MAX_ROW_आकार) {
        वापस false;
    }

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    काष्ठा htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    अगर (layout.योग_bytes > sess->vtcm_आकार) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_set_rows(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0]; //मूल्यावली
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1]; // indices
    स्थिर काष्ठा ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    अगर (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        वापस false;
    }

    अगर (src0->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        वापस false;
    }

    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_get_rows(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0]; //मूल्यावली
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1]; // indices
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        वापस false;
    }

    अगर (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        वापस false;
    }

    अगर (src0->type == dst->type) {
        अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            वापस false;
        }
    } अन्यथा अगर (src0->type == GGML_TYPE_I32) {
        वापस false;
    } अन्यथा अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    अगर (ggml_is_empty(op)) {
        वापस true;
    }

    काष्ठा htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    अगर (kparams.n_threads == 0 || (माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_argsort(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0]; //मूल्यावली
    स्थिर काष्ठा ggml_tensor * dst  = op;         // indices

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        वापस false;
    }

    स्थिर uint32_t योग_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिर माप_प्रकार vtcm_budget = sess->vtcm_आकार > 0 ? sess->vtcm_आकार : (8 * 1024 * 1024);

    काष्ठा htp_sort_vtcm_layout layout;
    अगर (!htp_sort_solve_layout(&layout, src0->ne[0], योग_rows, dst->ne[0], n_threads_max, vtcm_budget, false)) {
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_शीर्ष_k(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0]; //मूल्यावली
    स्थिर काष्ठा ggml_tensor * dst  = op;         // indices

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        वापस false;
    }

    स्थिर uint32_t योग_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    स्थिर माप_प्रकार vtcm_budget = sess->vtcm_आकार > 0 ? sess->vtcm_आकार : (8 * 1024 * 1024);

    काष्ठा htp_sort_vtcm_layout layout;
    अगर (!htp_sort_solve_layout(&layout, src0->ne[0], योग_rows, dst->ne[0], n_threads_max, vtcm_budget, true)) {
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_rope(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * src2 = op->src[2];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!ggml_are_same_shape(src0, dst)) {
        वापस false;
    }

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        वापस false;
    }

    अगर (src0->ne[0] <= 0) {
        वापस false;
    }

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    अगर (src0_nrows == 0) {
        वापस false;
    }

    स्थिर int32_t * op_params = &op->op_params[0];
    स्थिर पूर्णांक n_dims = op_params[1];
    स्थिर पूर्णांक mode   = op_params[2];
    स्थिर पूर्णांक n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    अगर (n_dims < 0 || n_dims % 2 != 0) {
        वापस false;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    अगर (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        वापस false;
    }

    भग्न freq_base;
    स_नकल(&freq_base, op_params + 5, माप(भग्न));
    अगर (freq_base < 0.0f) {
        वापस false;
    }

    अगर (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        वापस false;
    }

    स्थिर bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    अगर (mode == GGML_ROPE_TYPE_VISION) {
        अगर (n_dims != (पूर्णांक) (src0->ne[0] / 2) || n_offs != 0) {
            वापस false;
        }
    }

    अगर (is_mrope) {
        स्थिर int32_t * sections = op_params + 11;
        अगर (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            वापस false;
        }
    }

    स्थिर int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    अगर (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        वापस false;
    }

    अगर (src2) {
        अगर (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            वापस false;
        }
        अगर (src2->ne[0] < (n_dims / 2)) {
            वापस false;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    अगर (src0->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न)) {
        वापस false;
    }
    अगर (src0->nb[1] < src0->ne[0] * माप(भग्न) || dst->nb[1] < dst->ne[0] * माप(भग्न)) {
        वापस false;
    }

    स्थिर uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    स्थिर uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    काष्ठा htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    अगर (layout.योग_bytes > sess->vtcm_आकार) {
        वापस false;
    }

    वापस true;
}

अटल bool ggml_hexagon_supported_ssm_conv(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    // Only support FP32 for now
    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    // Check IO tensor shapes and dims
    अगर (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        वापस false; // src0 should be effectively 3D
    }

    स्थिर पूर्णांक d_conv  = src1->ne[0];
    स्थिर पूर्णांक d_inner = src0->ne[1];
    स्थिर पूर्णांक n_t = dst->ne[1];
    स्थिर पूर्णांक n_s = dst->ne[2];

    अगर (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        वापस false;
    }
    अगर (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        वापस false;
    }
    अगर (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        वापस false;
    }
    अगर (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        वापस false;
    }
    अगर (src0->nb[0] != माप(भग्न) || src1->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न)) {
        वापस false;
    }
    अगर (src0->nb[1] != src0->ne[0] * माप(भग्न) || src1->nb[1] != src1->ne[0] * माप(भग्न)) {
        वापस false;
    }

    काष्ठा htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    अगर ((माप_प्रकार) kparams.vtcm_आकार > sess->vtcm_आकार) {
        वापस false;
    }

    वापस true;
}

अटल bool im2col_3d_calc_output_आकार(uint32_t निवेश, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    अगर (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        वापस false;
    }

    स्थिर int64_t numerator = (int64_t) निवेश + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    अगर (numerator < 0) {
        वापस false;
    }

    स्थिर uint64_t size = (uint64_t) (numerator / stride) + 1;
    अगर (size == 0 || size > UINT32_MAX) {
        वापस false;
    }

    *output = (uint32_t) size;
    वापस true;
}

अटल bool ggml_hexagon_precompute_im2col_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_im2col_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!src0 || !src1 || !dst) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        वापस false;
    }

    अगर (src1->nb[0] != माप(भग्न) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    स्थिर int32_t * op_params = (स्थिर int32_t *) op->op_params;
    स्थिर int32_t s0 = op_params[0];
    स्थिर int32_t s1 = op_params[1];
    स्थिर int32_t p0 = op_params[2];
    स्थिर int32_t p1 = op_params[3];
    स्थिर int32_t d0 = op_params[4];
    स्थिर int32_t d1 = op_params[5];
    स्थिर bool is_2D = (op_params[6] == 1);

    अगर (s0 <= 0 || d0 <= 0 || p0 < 0) {
        वापस false;
    }
    अगर (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        वापस false;
    }

    स्थिर uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    स्थिर uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    स्थिर uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    स्थिर uint32_t IW = (uint32_t) src1->ne[0];
    स्थिर uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    स्थिर uint32_t KW = (uint32_t) src0->ne[0];
    स्थिर uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    स्थिर uint32_t OW = (uint32_t) dst->ne[1];

    अगर (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        वापस false;
    }

    स्थिर uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    स्थिर uint64_t योग_rows_64   = (uint64_t) N * OH;
    अगर (patch_stride_64 > UINT32_MAX || योग_rows_64 > UINT32_MAX) {
        वापस false;
    }
    स्थिर uint32_t patch_stride = (uint32_t) patch_stride_64;
    स्थिर uint32_t योग_rows   = (uint32_t) योग_rows_64;

    अगर (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        वापस false;
    }
    अगर (is_2D && dst->ne[2] != OH) {
        वापस false;
    }

    स्थिर uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? माप(ggml_fp16_t) : माप(भग्न);

    bool is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    अगर (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    स्थिर uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    स्थिर uint32_t n_threads   = (योग_rows < max_threads) ? योग_rows : max_threads;
    स्थिर uint32_t vtcm_आकार   = (sess && sess->vtcm_आकार > 0) ? sess->vtcm_आकार : 0;

    अगर (is_patchembed) {
        स्थिर uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * माप(भग्न);
        स्थिर uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        अगर (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            स्थिर uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            स्थिर uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            काष्ठा htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            अगर (vtcm_आकार == 0 || layout.योग_bytes <= vtcm_आकार) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_आकार            = layout.योग_bytes;
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
                kparams->योग_rows           = योग_rows;
                वापस true;
            }
        }
    }

    क्रम (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        स्थिर uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        अगर (Wb > UINT32_MAX) {
            अगर (owb == 1) अवरोध;
            जारी;
        }

        स्थिर uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * माप(भग्न);
        स्थिर uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        अगर (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            अगर (owb == 1) अवरोध;
            जारी;
        }

        स्थिर uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        स्थिर uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        काष्ठा htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        अगर (vtcm_आकार == 0 || layout.योग_bytes <= vtcm_आकार) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_आकार            = layout.योग_bytes;
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
            kparams->योग_rows           = योग_rows;
            वापस true;
        }
        अगर (owb == 1) अवरोध;
    }

    वापस false;
}

अटल bool ggml_hexagon_precompute_im2col_3d_params(
    स्थिर काष्ठा ggml_hexagon_session * sess,
    स्थिर काष्ठा ggml_tensor * op,
    काष्ठा htp_im2col_kernel_params * kparams
) {
    स_रखो(kparams, 0, माप(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (!src0 || !src1 || !dst) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        वापस false;
    }

    अगर (src1->nb[0] != माप(भग्न) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    स्थिर int32_t * params = (स्थिर int32_t *) op->op_params;
    स्थिर int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    स्थिर int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    स्थिर int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    स्थिर int32_t ic = params[9];

    अगर (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        वापस false;
    }

    अगर (src1->ne[3] % ic != 0) {
        वापस false;
    }

    स्थिर uint32_t IC = (uint32_t) ic;
    स्थिर uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    अगर (N == 0) {
        वापस false;
    }

    uint32_t OW, OH, OD;
    अगर (!im2col_3d_calc_output_आकार(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_आकार(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_आकार(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        वापस false;
    }

    स्थिर uint32_t KD = (uint32_t) src0->ne[2];
    स्थिर uint32_t KH = (uint32_t) src0->ne[1];
    स्थिर uint32_t KW = (uint32_t) src0->ne[0];
    स्थिर uint32_t ID = (uint32_t) src1->ne[2];
    स्थिर uint32_t IH = (uint32_t) src1->ne[1];
    स्थिर uint32_t IW = (uint32_t) src1->ne[0];

    स्थिर uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    स्थिर uint64_t योग_rows_64   = (uint64_t) N * OD * OH;
    स्थिर uint64_t dst_ne3_64      = (uint64_t) OD * N;

    अगर (patch_stride_64 > UINT32_MAX || योग_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        वापस false;
    }
    स्थिर uint32_t patch_stride = (uint32_t) patch_stride_64;
    स्थिर uint32_t योग_rows   = (uint32_t) योग_rows_64;
    स्थिर uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    अगर (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        वापस false;
    }

    स्थिर uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? माप(ggml_fp16_t) : माप(भग्न);

    स्थिर bool is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    स्थिर uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    स्थिर uint32_t n_threads   = (योग_rows < max_threads) ? योग_rows : max_threads;
    स्थिर uint32_t vtcm_आकार   = (sess && sess->vtcm_आकार > 0) ? sess->vtcm_आकार : 0;

    अगर (is_patchembed) {
        स्थिर uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * माप(भग्न);
        स्थिर uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        अगर (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            स्थिर uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            स्थिर uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            काष्ठा htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            अगर (vtcm_आकार == 0 || layout.योग_bytes <= vtcm_आकार) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_आकार            = layout.योग_bytes;
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
                kparams->योग_rows           = योग_rows;
                वापस true;
            }
        }
    }

    क्रम (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        स्थिर uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        अगर (Wb > UINT32_MAX) {
            अगर (owb == 1) अवरोध;
            जारी;
        }

        स्थिर uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * माप(भग्न);
        स्थिर uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        अगर (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            अगर (owb == 1) अवरोध;
            जारी;
        }

        स्थिर uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        स्थिर uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        काष्ठा htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        अगर (vtcm_आकार == 0 || layout.योग_bytes <= vtcm_आकार) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_आकार            = layout.योग_bytes;
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
            kparams->योग_rows           = योग_rows;
            वापस true;
        }
        अगर (owb == 1) अवरोध;
    }

    वापस false;
}

अटल bool ggml_hexagon_supported_im2col(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    काष्ठा htp_im2col_kernel_params kparams;
    वापस ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

अटल bool ggml_hexagon_supported_im2col_3d(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    काष्ठा htp_im2col_kernel_params kparams;
    वापस ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

अटल bool ggml_hexagon_supported_pad(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    स्थिर int32_t lp0 = ((स्थिर int32_t *) op->op_params)[0];
    स्थिर int32_t rp0 = ((स्थिर int32_t *) op->op_params)[1];
    स्थिर int32_t circular = ((स्थिर int32_t *) op->op_params)[8];

    अगर (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_cumsum(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_diag(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    // diag only supports F32 currently
    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    // Input must have ne[1] == 1 (vector input)
    अगर (src0->ne[1] != 1) {
        वापस false;
    }

    // Output must be square in first two dimensions
    अगर (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_solve_tri(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0]; // A
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1]; // B
    स्थिर काष्ठा ggml_tensor * dst  = op;         // X

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (src0->ne[0] != src0->ne[1]) {
        वापस false;
    }

    अगर (src0->ne[1] != src1->ne[1]) {
        वापस false;
    }

    अगर (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        वापस false;
    }

    अगर (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        वापस false;
    }

    वापस true;

    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_tri(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32) { वापस false; }
    अगर (dst->type  != GGML_TYPE_F32) { वापस false; }
    अगर (!ggml_are_same_shape(src0, dst)) { वापस false; }
    अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { वापस false; }

    वापस true;

    GGML_UNUSED(sess);
}

अटल स्थिर अक्षर * ggml_backend_hexagon_name(ggml_backend_t backend) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    वापस sess->c_name();
}

अटल व्योम ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    मिटाओ backend;
}

अटल htp_op_code op_remap_to_htp(स्थिर ggml_tensor * t) {
    चयन (t->op) {
        हाल GGML_OP_FLASH_ATTN_EXT:  वापस HTP_OP_FLASH_ATTN_EXT;
        हाल GGML_OP_MUL_MAT:
            अगर (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                वापस HTP_OP_HADAMARD;
            }
            वापस HTP_OP_MUL_MAT;
        हाल GGML_OP_MUL_MAT_ID:      वापस HTP_OP_MUL_MAT_ID;
        हाल GGML_OP_MUL:             वापस HTP_OP_MUL;
        हाल GGML_OP_ADD:             वापस HTP_OP_ADD;
        हाल GGML_OP_ADD_ID:          वापस HTP_OP_ADD_ID;
        हाल GGML_OP_SUB:             वापस HTP_OP_SUB;
        हाल GGML_OP_DIV:             वापस HTP_OP_DIV;
        हाल GGML_OP_DUP:
        हाल GGML_OP_CPY:             वापस HTP_OP_CPY;
        हाल GGML_OP_CONT:            वापस HTP_OP_CPY;
        हाल GGML_OP_GET_ROWS:        वापस HTP_OP_GET_ROWS;
        हाल GGML_OP_SET_ROWS:        वापस HTP_OP_SET_ROWS;
        हाल GGML_OP_योग:             वापस HTP_OP_योग;
        हाल GGML_OP_योग_ROWS:        वापस HTP_OP_योग_ROWS;
        हाल GGML_OP_POOL_2D:         वापस HTP_OP_POOL_2D;
        हाल GGML_OP_POOL_1D:         वापस HTP_OP_POOL_1D;
        हाल GGML_OP_ARGSORT:         वापस HTP_OP_ARGSORT;
        हाल GGML_OP_शीर्ष_K:           वापस HTP_OP_शीर्ष_K;
        हाल GGML_OP_ARGMAX:          वापस HTP_OP_ARGMAX;
        हाल GGML_OP_NORM:            वापस HTP_OP_NORM;
        हाल GGML_OP_L2_NORM:         वापस HTP_OP_L2_NORM;
        हाल GGML_OP_RMS_NORM:        वापस HTP_OP_RMS_NORM;
        हाल GGML_OP_CONCAT:          वापस HTP_OP_CONCAT;
        हाल GGML_OP_SCALE:           वापस HTP_OP_SCALE;
        हाल GGML_OP_CLAMP:           वापस HTP_OP_CLAMP;
        हाल GGML_OP_LEAKY_RELU:      वापस HTP_OP_LEAKY_RELU;
        हाल GGML_OP_SQR:             वापस HTP_OP_SQR;
        हाल GGML_OP_SQRT:            वापस HTP_OP_SQRT;
        हाल GGML_OP_LOG:             वापस HTP_OP_UNARY_LOG;
        हाल GGML_OP_SOFT_MAX:        वापस HTP_OP_SOFTMAX;
        हाल GGML_OP_SSM_CONV:        वापस HTP_OP_SSM_CONV;
        हाल GGML_OP_GATED_DELTA_NET: वापस HTP_OP_GATED_DELTA_NET;
        हाल GGML_OP_ROPE:            वापस HTP_OP_ROPE;
        हाल GGML_OP_REPEAT:          वापस HTP_OP_REPEAT;
        हाल GGML_OP_CUMSUM:          वापस HTP_OP_CUMSUM;
        हाल GGML_OP_FILL:            वापस HTP_OP_FILL;
        हाल GGML_OP_DIAG:            वापस HTP_OP_DIAG;
        हाल GGML_OP_SOLVE_TRI:       वापस HTP_OP_SOLVE_TRI;
        हाल GGML_OP_TRI:             वापस HTP_OP_TRI;
        हाल GGML_OP_PAD:             वापस HTP_OP_PAD;
        हाल GGML_OP_IM2COL:          वापस HTP_OP_IM2COL;
        हाल GGML_OP_IM2COL_3D:       वापस HTP_OP_IM2COL_3D;
        हाल GGML_OP_ROLL:            वापस HTP_OP_ROLL;

        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(t)) {
                हाल GGML_UNARY_OP_SILU:       वापस HTP_OP_UNARY_SILU;
                हाल GGML_UNARY_OP_GELU:       वापस HTP_OP_UNARY_GELU;
                हाल GGML_UNARY_OP_GELU_QUICK: वापस HTP_OP_UNARY_GELU_QUICK;
                हाल GGML_UNARY_OP_GELU_ERF:   वापस HTP_OP_UNARY_GELU_ERF;
                हाल GGML_UNARY_OP_SIGMOID:    वापस HTP_OP_UNARY_SIGMOID;
                हाल GGML_UNARY_OP_NEG:        वापस HTP_OP_UNARY_NEG;
                हाल GGML_UNARY_OP_EXP:        वापस HTP_OP_UNARY_EXP;
                हाल GGML_UNARY_OP_SOFTPLUS:   वापस HTP_OP_UNARY_SOFTPLUS;
                हाल GGML_UNARY_OP_TANH:       वापस HTP_OP_UNARY_TANH;
                हाल GGML_UNARY_OP_ABS:        वापस HTP_OP_UNARY_ABS;
                हाल GGML_UNARY_OP_RELU:       वापस HTP_OP_UNARY_RELU;
                हाल GGML_UNARY_OP_STEP:       वापस HTP_OP_UNARY_STEP;
            शेष:
                अवरोध;
            }
            अवरोध;

        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(t)) {
                हाल GGML_GLU_OP_SWIGLU:       वापस HTP_OP_GLU_SWIGLU;
                हाल GGML_GLU_OP_SWIGLU_OAI:   वापस HTP_OP_GLU_SWIGLU_OAI;
                हाल GGML_GLU_OP_SWIGLU_CLAMP: वापस HTP_OP_GLU_SWIGLU_CLAMP;
                हाल GGML_GLU_OP_GEGLU:        वापस HTP_OP_GLU_GEGLU;
                हाल GGML_GLU_OP_GEGLU_QUICK:  वापस HTP_OP_GLU_GEGLU_QUICK;
                हाल GGML_GLU_OP_GEGLU_ERF:    वापस HTP_OP_GLU_GEGLU_ERF;
                शेष: अवरोध;
            }
            अवरोध;

        शेष:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    वापस HTP_OP_INVALID;
}

अटल अंतरभूत bool op_is_compute(ggml_tensor *गाँठ)
{
    वापस !ggml_op_is_empty(गाँठ->op) && !ggml_is_empty(गाँठ) && (गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

अटल bool mm_is_hmx_eligible(स्थिर ggml_tensor * t) {
    अगर (opt_nhmx == 0) { वापस false; }

    स्थिर ggml_tensor * src0 = t->src[0];
    स्थिर ggml_tensor * src1 = t->src[1];

    अगर (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        स्थिर ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        वापस ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, false, false);
    }

    स्थिर bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    स्थिर bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    वापस ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

अटल bool is_supported_mul_mat_nx_kernel(स्थिर ggml_tensor * src0, स्थिर काष्ठा htp_mm_kernel_params * kparams) {
    अगर (kparams->n_hmx) {
        वापस kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    अगर (!ggml_hexagon_tensor_is_tiled(src0)) {
        वापस false;
    }

    वापस kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

अटल bool is_supported_mul_mat_id_nx_kernel(स्थिर ggml_tensor * src0, स्थिर काष्ठा htp_mm_kernel_params * kparams) {
    अगर (kparams->n_hmx) {
        वापस kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    अगर (!ggml_hexagon_tensor_is_tiled(src0)) {
        वापस false;
    }

    वापस kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

अटल bool is_mergeable_mul_mat(स्थिर ggml_tensor * t) {
    अगर (t->op != GGML_OP_MUL_MAT) वापस false;
    अगर (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) वापस false;

    स्थिर ggml_tensor * src0 = t->src[0];
    स्थिर ggml_tensor * src1 = t->src[1];
    अगर (src0->ne[2] != 1 || src0->ne[3] != 1) वापस false;

    अगर (mm_is_hmx_eligible(t)) {
        वापस htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    अगर (src1->type != GGML_TYPE_F32) वापस false;

    वापस ggml_hexagon_tensor_is_tiled(src0);
}

अटल bool is_mergeable_mul_mat_pair(स्थिर ggml_tensor * n1, स्थिर ggml_tensor * n2) {
    अगर (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        वापस false;
    }
    अगर (n1->src[1] != n2->src[1]) {
        वापस false;
    }
    अगर (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        वापस false;
    }
    अगर (n1->src[0]->type != n2->src[0]->type) {
        वापस false;
    }
    अगर (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        वापस false;
    }
    अगर (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        वापस false;
    }
    वापस true;
}

अटल bool is_mergeable_mul_mat_id(स्थिर ggml_tensor * t) {
    अगर (t->op != GGML_OP_MUL_MAT_ID) वापस false;

    स्थिर ggml_tensor * src0 = t->src[0];
    अगर (mm_is_hmx_eligible(t)) {
        वापस htp_is_hmx_weight_type(src0->type);
    }

    वापस ggml_hexagon_tensor_is_tiled(src0);
}

अटल bool is_mergeable_mul_mat_id_pair(स्थिर ggml_tensor * n1, स्थिर ggml_tensor * n2) {
    अगर (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        वापस false;
    }
    अगर (n1->src[1] != n2->src[1]) {
        वापस false;
    }
    अगर (n1->src[2] != n2->src[2]) {
        वापस false;
    }
    अगर (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        वापस false;
    }
    अगर (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        वापस false;
    }
    अगर (n1->src[0]->type != n2->src[0]->type) {
        वापस false;
    }
    अगर (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        वापस false;
    }
    अगर (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        वापस false;
    }
    वापस true;
}

अटल ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);

    अगर (sess->last_error > HTP_STATUS_OK) {
        वापस GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    स्थिर std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    अगर (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } अन्यथा {
        // Tag fusable tensors in graph
        क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
            स्वतः * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            अगर (!extra) जारी;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            अगर (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } अन्यथा अगर ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                अगर ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_गाँठ_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        क्रम (पूर्णांक i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            अगर (!op_is_compute(n)) {
                जारी;
            }

            htp_opnode गाँठ(HTP_OP_INVALID, n);
            गाँठ.opcode = op_remap_to_htp(n);
            अगर (गाँठ.opcode == HTP_OP_MUL_MAT || गाँठ.opcode == HTP_OP_MUL_MAT_ID || गाँठ.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = गाँठ.गाँठ->src[0];
                अगर (ggml_hexagon_needs_repack(src0)) {
                    अगर (opt_verbose) {
                        अक्षर dims_buf[64];
                        अक्षर strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, माप(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, माप(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    स_नकल(raw.तथ्य(), src0->तथ्य, raw.size());
                    htp_repack_tensor_tiled(src0, raw.तथ्य(), raw.size());
                    स्वतः * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            अगर (गाँठ.opcode == HTP_OP_MUL_MAT || गाँठ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    गाँठ.गाँठ->src[0], गाँठ.गाँठ->src[1], गाँठ.गाँठ,
                    (काष्ठा htp_mm_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_MUL || गाँठ.opcode == HTP_OP_ADD || गाँठ.opcode == HTP_OP_ADD_ID || गाँठ.opcode == HTP_OP_SUB || गाँठ.opcode == HTP_OP_DIV) {
                स्थिर ggml_tensor * src1 = गाँठ.गाँठ->src[1];
                स्थिर ggml_tensor * src2 = (गाँठ.opcode == HTP_OP_ADD_ID) ? गाँठ.गाँठ->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_द्विआधारी_params(sess,
                    गाँठ.opcode, गाँठ.गाँठ->src[0], src1, गाँठ.गाँठ,
                    (काष्ठा htp_द्विआधारी_kernel_params *) गाँठ.kernel_params, src2));
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_fa_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (htp_op_is_unary(गाँठ.opcode)) {
                स्वतः inputs = गाँठ.get_inputs();
                स्थिर काष्ठा ggml_tensor * src0 = inputs[0];
                स्थिर काष्ठा ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    गाँठ.opcode, src0, src1, गाँठ.dst(),
                    (काष्ठा htp_unary_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    गाँठ.गाँठ->src[0], गाँठ.गाँठ->src[1], गाँठ.dst(),
                    (काष्ठा htp_get_rows_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    गाँठ.गाँठ->src[0], गाँठ.गाँठ->src[1], गाँठ.dst(),
                    (काष्ठा htp_set_rows_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_rope_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    गाँठ.गाँठ->src[0], गाँठ.गाँठ->src[1], गाँठ.dst(),
                    (काष्ठा htp_ssm_conv_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_softmax_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_gdn_kernel_params *)गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_ARGSORT || गाँठ.opcode == HTP_OP_शीर्ष_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    गाँठ.गाँठ,
                    गाँठ.opcode == HTP_OP_शीर्ष_K,
                    (काष्ठा htp_sort_kernel_params *) गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_POOL_2D || गाँठ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, गाँठ.गाँठ->src[0], गाँठ.dst(),
                    (काष्ठा htp_pool_2d_kernel_params *)गाँठ.kernel_params,
                    गाँठ.opcode == HTP_OP_POOL_1D);
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_concat_kernel_params *) गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_CPY || गाँठ.opcode == HTP_OP_CPY_FENCE) {
                स्थिर bool ok = ggml_hexagon_precompute_cpy_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_copy_kernel_params *) गाँठ.kernel_params
                );
                स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_copy_kernel_params *) गाँठ.kernel_params;
                अगर (ok && गाँठ.opcode == HTP_OP_CPY && kparams->योग_elems == 0) {
                    जारी;
                }
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_im2col_kernel_params *) गाँठ.kernel_params
                );
            } अन्यथा अगर (गाँठ.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    गाँठ.गाँठ,
                    (काष्ठा htp_im2col_kernel_params *) गाँठ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(गाँठ));
        }

        अगर (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } अन्यथा {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    क्रम (स्थिर स्वतः & गाँठ : *nodes_ptr) {
        sess->enqueue_op(गाँठ);
    }

    अगर (sess->last_error > HTP_STATUS_OK) {
        वापस GGML_STATUS_FAILED;
    }

    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    अगर (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

क्रमागत ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

काष्ठा ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

काष्ठा ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    व्योम reset() {
        ranges.सफा();
    }

    व्योम add(स्थिर ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    bool check(स्थिर ggml_hexagon_mem_range & mr) स्थिर {
        क्रम (स्थिर स्वतः & cmp : ranges) {
            अगर (mr.pb != cmp.pb) {
                जारी;
            }
            अगर (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                जारी;
            }
            अगर (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                वापस false;
            }
        }
        वापस true;
    }
};

अटल ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(स्थिर ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    स्थिर ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    अगर (tensor->बफर) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->बफर,
            /*.p0 =*/ (uint64_t) tensor->तथ्य,
            /*.p1 =*/ (uint64_t) tensor->तथ्य + ggml_backend_buft_get_alloc_आकार(tensor->बफर->buft, tensor),
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
    वापस mr;
}

अटल व्योम ggml_hexagon_mem_ranges_add_गाँठ(ggml_hexagon_mem_ranges & mrs, स्थिर htp_opnode & गाँठ) {
    अगर (गाँठ.is_empty()) वापस;

    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (गाँठ.गाँठ->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(गाँठ.गाँठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            अगर (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(गाँठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

अटल bool ggml_hexagon_mem_ranges_check_गाँठ(स्थिर ggml_hexagon_mem_ranges & mrs, स्थिर htp_opnode & गाँठ) {
    अगर (गाँठ.is_empty()) वापस true;

    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (गाँठ.गाँठ->src[i]) {
            अगर (!mrs.check(ggml_hexagon_mem_range_from_tensor(गाँठ.गाँठ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                वापस false;
            }
        }
    }
    क्रम (स्थिर स्वतः * fused : गाँठ.fused) {
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            अगर (fused->src[i]) {
                अगर (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    वापस false;
                }
            }
        }
    }
    वापस mrs.check(ggml_hexagon_mem_range_from_tensor(गाँठ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

अटल std::vector<पूर्णांक> ggml_hexagon_graph_optimize_reorder(स्थिर std::vector<htp_opnode> & nodes) {
    स्थिर पूर्णांक n = nodes.size();

    std::vector<पूर्णांक> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    क्रम (पूर्णांक i0 = 0; i0 < n; i0++) {
        अगर (used[i0]) {
            जारी;
        }

        स्थिर स्वतः & node0 = nodes[i0];

        अगर (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = true;
            जारी;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr पूर्णांक N_FORWARD = 16;

        std::vector<पूर्णांक> ढेर_2;
        ढेर_2.push_back(i0);

        mrs.reset();

        क्रम (पूर्णांक i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            अगर (used[i1]) {
                जारी;
            }

            स्थिर स्वतः & node1 = nodes[i1];

            अगर (node1.stackable() && node1.same_निवेश(node0) && ggml_hexagon_mem_ranges_check_गाँठ(mrs, node1)) {
                ढेर_2.push_back(i1);
            } अन्यथा {
                ggml_hexagon_mem_ranges_add_गाँठ(mrs, node1);
            }
        }

        क्रम (पूर्णांक idx : ढेर_2) {
            res.push_back(idx);
            used[idx] = true;
        }
    }

    वापस res;
}

अटल व्योम ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    स्थिर पूर्णांक n = gf->n_nodes;

    constexpr पूर्णांक MAX_FUSE = 16;

    क्रमागत ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    क्रम (पूर्णांक i = 0; i < n; i++) {
        htp_opnode गाँठ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        अगर (गाँठ.op() == GGML_OP_ADD ||
            गाँठ.op() == GGML_OP_NORM ||
            गाँठ.op() == GGML_OP_RMS_NORM) {
            ops[0] = गाँठ.op();

            पूर्णांक f = i + 1;
            जबतक (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                अगर (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    अवरोध;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            क्रम (; f > 1; f--) {
                अगर (ggml_can_fuse(gf, i, ops, f)) {
                    अवरोध;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            क्रम (पूर्णांक k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                गाँठ.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(गाँठ));
    }

    स्थिर स्वतः order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        पूर्णांक j = 0;
        क्रम (स्थिर स्वतः i : order) {
            स्थिर स्वतः & गाँठ = nodes[i];

            gf->nodes[j++] = गाँठ.गाँठ;

            क्रम (स्वतः * fused : गाँठ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

अटल uint64_t ggml_hexagon_session_कुंजी(स्थिर ggml_hexagon_session * sess) {
    वापस ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

अटल bool ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    स्वतः sbuf_dst = sess_dst->mmap_tensor(dst);

    अगर (!sess_src->clone_बफर(sbuf_dst)) { वापस false; }

    स्थिर uint64_t src_कुंजी = ggml_hexagon_session_कुंजी(sess_src);
    स्वतः & fence_slot = sess_dst->cpy_fence_slots[src_कुंजी];
    अगर (!fence_slot) {
        fence_slot = (अस्थिर uint32_t *) sess_dst->alloc_fence(1);
    }

    अगर (!sess_src->clone_बफर(sess_dst->fence_buf)) { वापस false; }

    अगर (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    अटल ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.बफर = &sess_dst->fence_buf->backend_बफर;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.तथ्य   = (व्योम *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = माप(int32_t);
    fence_tensor.nb[1]  = माप(int32_t);
    fence_tensor.nb[2]  = माप(int32_t);
    fence_tensor.nb[3]  = माप(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ true);

    sess_dst->add_peer(sess_src);

    वापस true;
}

अटल bool ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    स्वतः sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    अगर (!sess_dst->clone_बफर(sbuf_src)) { वापस false; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    वापस true;
}

अटल bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    अगर (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        वापस false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    अगर (!dst->extra) {
        ggml_backend_बफर_init_tensor(dst->बफर, dst);
    }

    स्वतः * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    स्थिर स्वतः * src_extra = static_cast<स्थिर ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    स्वतः sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    स्वतः sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    अगर (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        वापस true;
    }

    अगर (sess_src->phys_idx != sess_dst->phys_idx)
        वापस ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    वापस ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

अटल ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = नया ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (अस्थिर uint32_t *) sess->alloc_fence(1);

    अटल ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.बफर = &sess->fence_buf->backend_बफर;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.तथ्य   = (व्योम *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = माप(int32_t);
    hex_event->fence_tensor.nb[1]  = माप(int32_t);
    hex_event->fence_tensor.nb[2]  = माप(int32_t);
    hex_event->fence_tensor.nb[3]  = माप(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (व्योम *)hex_event, (व्योम *)hex_event->fence_slot);

    वापस नया ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

अटल व्योम ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    अगर (hex_event->seq == 0) {
        वापस;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (व्योम *)hex_event, hex_event->seq, (व्योम *)hex_event->fence_slot);

    स्वतः * fence = reinterpret_cast<स्थिर अस्थिर std::atomic<uint32_t> *>(hex_event->fence_slot);

    अगर ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    जबतक (true) {
        अगर ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            अगर (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            अवरोध;
        }
        std::this_thread::yield();
    }
}

अटल व्योम ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    स्वतः * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (व्योम *)hex_event);
    hex_event->fence_sess->free_fence((व्योम *) hex_event->fence_slot, 1);
    मिटाओ hex_event;
    मिटाओ event;
}

अटल व्योम ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    स्वतः * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

अटल व्योम ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    स्वतः hex_event = static_cast<ggml_hexagon_event *>(event->context);

    अगर (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ false);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (व्योम *)hex_event, hex_event->seq, (व्योम *)hex_event->fence_slot);
}

अटल व्योम ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    स्वतः hex_event = static_cast<ggml_hexagon_event *>(event->context);

    अगर (hex_event->seq == 0) {
        वापस;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (व्योम *)hex_event, hex_event->seq, (व्योम *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    अगर (sess->phys_idx == hex_event->sess->phys_idx) {
        अगर (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        वापस;
    }

    sess->clone_बफर(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ true);
}

अटल व्योम ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, tensor->बफर ? (पूर्णांक) tensor->बफर->usage : -1);
    ggml_backend_tensor_set(tensor, तथ्य, offset, size);
}

अटल व्योम ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, tensor->बफर ? (पूर्णांक) tensor->बफर->usage : -1);
    sess->flush_sync();
    अगर (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, तथ्य, offset, size);
}

अटल व्योम ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     काष्ठा ggml_tensor * tensor,
                                                     स्थिर व्योम * तथ्य,
                                                     माप_प्रकार offset,
                                                     माप_प्रकार size,
                                                     माप_प्रकार n_copies,
                                                     माप_प्रकार stride_tensor,
                                                     माप_प्रकार stride_तथ्य) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य, tensor->बफर ? (पूर्णांक) tensor->बफर->usage : -1);
    ggml_backend_tensor_set_2d(tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

अटल व्योम ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     स्थिर काष्ठा ggml_tensor * tensor,
                                                     व्योम * तथ्य,
                                                     माप_प्रकार offset,
                                                     माप_प्रकार size,
                                                     माप_प्रकार n_copies,
                                                     माप_प्रकार stride_tensor,
                                                     माप_प्रकार stride_तथ्य) {
    स्वतः sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य, tensor->बफर ? (पूर्णांक) tensor->बफर->usage : -1);
    sess->flush_sync();
    अगर (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

अटल काष्ठा ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

अटल ggml_guid_t ggml_backend_hexagon_guid() {
    अटल ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    वापस &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    वापस backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

अटल ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    वापस नया ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

अटल स्थिर अक्षर * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    वापस dev_ctx->c_name();

    GGML_UNUSED(dev);
}

अटल स्थिर अक्षर * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    वापस "Hexagon";
    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    *मुक्त  = opt_mbuf_योग;
    *योग = opt_mbuf_योग;

    GGML_UNUSED(dev);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ false,
    };
}

अटल ggml_backend_बफर_type_t ggml_backend_hexagon_device_get_बफर_type(ggml_backend_dev_t dev) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    वापस &dev_ctx->बफर_type;
}

अटल ggml_backend_बफर_type_t ggml_backend_hexagon_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    अगर (!opt_hostbuf) {
        वापस शून्य;
    }
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    वापस &dev_ctx->host_बफर_type;
}

अटल bool ggml_hexagon_supported_cpy(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    काष्ठा htp_copy_kernel_params kparams;
    वापस ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

अटल bool ggml_hexagon_supported_cont(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    काष्ठा htp_copy_kernel_params kparams;
    वापस ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

अटल bool ggml_hexagon_supported_repeat(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    GGML_UNUSED(sess);
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    // Support f32 and f16
    अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) वापस false;

    // src and dst must be the same type
    अगर (src0->type != dst->type) वापस false;

    // dst dims must be multiples of src dims
    अगर (dst->ne[0] % src0->ne[0] != 0) वापस false;
    अगर (dst->ne[1] % src0->ne[1] != 0) वापस false;
    अगर (dst->ne[2] % src0->ne[2] != 0) वापस false;
    अगर (dst->ne[3] % src0->ne[3] != 0) वापस false;

    // require contiguous tensors (no transposition)
    अगर (ggml_is_transposed(src0) || ggml_is_transposed(dst)) वापस false;

    वापस true;
}

अटल bool ggml_hexagon_supported_concat(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    काष्ठा htp_concat_kernel_params kparams;
    वापस ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

अटल bool ggml_hexagon_supported_fill(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * dst = op;

    अगर (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        वापस false;
    }

    वापस true;
    GGML_UNUSED(sess);
}

अटल bool ggml_hexagon_supported_roll(स्थिर काष्ठा ggml_hexagon_session * sess, स्थिर काष्ठा ggml_tensor * op) {
    GGML_UNUSED(sess);

    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * dst  = op;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (!ggml_are_same_shape(src0, dst)) {
        वापस false;
    }

    अगर (src0->nb[0] != ggml_type_आकार(src0->type) || dst->nb[0] != ggml_type_आकार(dst->type)) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(dst)) {
        वापस false;
    }

    वापस true;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
अटल bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    स्वतः sess    = dev_ctx->session();

    // reject ops that match the filter
    अगर (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        वापस false;
    }

    bool supp = false;
    चयन (op->op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
            supp = true;
            अवरोध;

        हाल GGML_OP_MUL:
        हाल GGML_OP_ADD:
        हाल GGML_OP_SUB:
        हाल GGML_OP_DIV:
            supp = ggml_hexagon_supported_द्विआधारी(sess, op);
            अवरोध;

        हाल GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            अवरोध;

        हाल GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            अवरोध;

        हाल GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            अवरोध;

        हाल GGML_OP_NORM:
        हाल GGML_OP_L2_NORM:
        हाल GGML_OP_RMS_NORM:
        हाल GGML_OP_SCALE:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            अवरोध;

        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            अवरोध;

        हाल GGML_OP_योग:
            supp = ggml_hexagon_supported_योग(sess, op);
            अवरोध;

        हाल GGML_OP_योग_ROWS:
            supp = ggml_hexagon_supported_योग_rows(sess, op);
            अवरोध;

        हाल GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            अवरोध;

        हाल GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            अवरोध;

        हाल GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            अवरोध;

        हाल GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            अवरोध;

        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(op)) {
                हाल GGML_UNARY_OP_NEG:
                हाल GGML_UNARY_OP_EXP:
                हाल GGML_UNARY_OP_SIGMOID:
                हाल GGML_UNARY_OP_SOFTPLUS:
                हाल GGML_UNARY_OP_TANH:
                हाल GGML_UNARY_OP_ABS:
                हाल GGML_UNARY_OP_SILU:
                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_GELU_QUICK:
                हाल GGML_UNARY_OP_GELU_ERF:
                हाल GGML_UNARY_OP_RELU:
                हाल GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    अवरोध;
                शेष:
                    supp = false;
                    अवरोध;
            }
            अवरोध;

        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(op)) {
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_SWIGLU_OAI:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                हाल GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    अवरोध;
                शेष:
                    supp = false;
                    अवरोध;
            }
            अवरोध;

        हाल GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            अवरोध;

        हाल GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            अवरोध;

        हाल GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            अवरोध;

        हाल GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            अवरोध;

        हाल GGML_OP_DUP:
        हाल GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            अवरोध;

        हाल GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            अवरोध;

        हाल GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            अवरोध;

        हाल GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            अवरोध;

        हाल GGML_OP_शीर्ष_K:
            supp = ggml_hexagon_supported_शीर्ष_k(sess, op);
            अवरोध;

        हाल GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            अवरोध;

        हाल GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            अवरोध;

        हाल GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            अवरोध;

        हाल GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            अवरोध;

        हाल GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            अवरोध;

        हाल GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            अवरोध;

        हाल GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            अवरोध;

        हाल GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            अवरोध;

        हाल GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            अवरोध;

        हाल GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            अवरोध;

        हाल GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            अवरोध;

        हाल GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            अवरोध;

        शेष:
            अवरोध;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    वापस supp;
}

अटल bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    bool supp = (buft == &dev_ctx->host_बफर_type) || (buft == &dev_ctx->बफर_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    वापस supp;
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ शून्य,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ शून्य,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    क्रम (माप_प्रकार i = 0; i < opt_ndev; i++) {
        स्थिर स्वतः & cfg = opt_device_configs[i];
        अगर (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } अन्यथा {
            std::string peers_str;
            क्रम (स्थिर स्वतः & p : cfg.mdev_group) {
                अगर (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = नया ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    क्रम (माप_प्रकार i = 0; i < opt_ndev; i++) {
        स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        मिटाओ dev_ctx;
    }
}

अटल स्थिर अक्षर * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    वापस "HTP";
    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_hexagon_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    वापस opt_ndev;
    GGML_UNUSED(reg);
}

अटल ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    स्वतः hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    अगर (सूचकांक >= opt_ndev || !hreg->devices[सूचकांक].context) {
        वापस nullptr;
    }

    वापस &hreg->devices[सूचकांक];
}

// ** communication context for tensor-split allreduce

अटल व्योम * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) {
    अगर (n_backends < 2 || n_backends > 4) {
        वापस nullptr;
    }

    क्रम (माप_प्रकार i = 0; i < n_backends; ++i) {
        अगर (!ggml_backend_is_hexagon(backends[i])) {
            वापस nullptr;
        }
    }

    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        क्रम (माप_प्रकार j = i + 1; j < n_backends; j++) {
            स्वतः sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            अगर (sess_i->phys_idx == sess_j->phys_idx) {
                वापस nullptr;
            }
        }
    }

    स्वतः * ctx = नया ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    अटल ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (अस्थिर uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].बफर = &sess_i->fence_buf->backend_बफर;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].तथ्य   = (व्योम *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = माप(int32_t);
        ctx->fence_tensors[i].nb[1]  = माप(int32_t);
        ctx->fence_tensors[i].nb[2]  = माप(int32_t);
        ctx->fence_tensors[i].nb[3]  = माप(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    वापस ctx;
}

अटल व्योम ggml_backend_hexagon_comm_free(व्योम * comm_ctx_v) {
    अगर (!comm_ctx_v) वापस;
    स्वतः * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    क्रम (माप_प्रकार i = 0; i < ctx->n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((व्योम *) ctx->fence_slots[i], 1);
    }
    मिटाओ ctx;
}

अटल bool ggml_backend_hexagon_comm_allreduce_tensor(व्योम * comm_ctx_v, काष्ठा ggml_tensor ** tensors) {
    अगर (opt_ar_select == 0 || !comm_ctx_v) वापस false;
    स्वतः * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    स्थिर माप_प्रकार n_backends = comm_ctx->n_backends;

    अगर (n_backends < 2 || n_backends > 4) वापस false;

    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        क्रम (माप_प्रकार j = i + 1; j < n_backends; j++) {
            स्वतः sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            अगर (sess_i->phys_idx == sess_j->phys_idx) {
                वापस false;
            }
        }
    }

    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        अगर (!tensors[i] || !tensors[i]->बफर || !ggml_backend_बफर_is_hexagon(tensors[i]->बफर)) {
            वापस false;
        }
        अगर (tensors[i]->type != tensors[0]->type) {
            वापस false;
        }
        अगर (!ggml_is_contiguous(tensors[i])) {
            वापस false;
        }
        अगर (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            वापस false;
        }
    }

    अगर (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        वापस false;
    }

    क्रम (माप_प्रकार r = 0; r < n_backends; r++) {
        स्वतः sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        काष्ठा htp_allreduce_kernel_params kparams;
        अगर (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, false, false, /*is_shard_ok=*/ false, &kparams)) {
            वापस false;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    क्रम (माप_प्रकार i = 1; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        अगर ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    अगर (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    अगर (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<स्थिर ggml_tensor *> तथ्य_tensors(n_backends);
    std::vector<स्थिर ggml_tensor *> sync_tensors(n_backends);
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        तथ्य_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    क्रम (माप_प्रकार r = 0; r < n_backends; r++) {
        स्वतः sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], तथ्य_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        क्रम (माप_प्रकार j = 0; j < n_backends; j++) {
            अगर (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    वापस true;
}

अटल ggml_backend_बफर_type_t ggml_backend_hexagon_split_बफर_type(पूर्णांक main_device, स्थिर भग्न * tensor_split) {
    GGML_UNUSED(tensor_split);
    स्वतः reg = ggml_backend_hexagon_reg();
    स्वतः dev = ggml_backend_reg_dev_get(reg, main_device);
    अगर (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    अगर (!dev) वापस nullptr;
    स्वतः dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    वापस &dev_ctx->बफर_type;
}

अटल व्योम * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    GGML_UNUSED(reg);
    अगर (म_भेद(name, "ggml_backend_split_buffer_type") == 0) {
        वापस (व्योम *) ggml_backend_hexagon_split_बफर_type;
    }
    अगर (म_भेद(name, "ggml_backend_comm_init") == 0) {
        वापस (व्योम *) ggml_backend_hexagon_comm_init;
    }
    अगर (म_भेद(name, "ggml_backend_comm_free") == 0) {
        वापस (व्योम *) ggml_backend_hexagon_comm_free;
    }
    अगर (म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        वापस (व्योम *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    वापस शून्य;
}

ढाँचा<typename T> std::vector<T> str_to_vec(स्थिर अक्षर* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    जबतक (std::लो_पंक्ति(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    वापस v;
}

ढाँचा<typename T, पूर्णांक BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::रखो_आधार(BASE) << std::लिखो_आधार;
    क्रम (स्वतः i : v) { ss << i << ','; }
    स्वतः str = ss.str(); str.pop_back(); // drop last comma
    वापस str;
}

अटल व्योम ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, bool discovery_supported, स्थिर std::unordered_map<पूर्णांक, fastrpc_domain> & cdsp_map) {
    अगर (discovery_supported) {
        स्वतः it = cdsp_map.find(cfg.physical_idx);
        अगर (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } अन्यथा {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } अन्यथा {
        चयन (cfg.physical_idx) {
            हाल 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                अवरोध;
            हाल 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                अवरोध;
            शेष:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                अवरोध;
        }
    }
    क्रम (स्वतः & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
अटल व्योम ggml_hexagon_discover_devices() {
    std::unordered_map<पूर्णांक, fastrpc_domain> cdsp_map;
    bool discovery_supported = false;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    पूर्णांक err = remote_system_request(&domain_info);
    अगर (err == AEE_SUCCESS && domain_info.sys.अंक_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.अंक_domains);
        domain_info.sys.domains     = domains.तथ्य();
        domain_info.sys.max_domains = (पूर्णांक) domains.size();

        err = remote_system_request(&domain_info);
        अगर (err == AEE_SUCCESS) {
            discovery_supported = true;
            स्थिर पूर्णांक n_domains = std::min(domain_info.sys.अंक_domains, (पूर्णांक) domains.size());
            क्रम (पूर्णांक i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (पूर्णांक) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                अगर (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (पूर्णांक) domains[i].type);
                    जारी;
                }
                अगर (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    जारी;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } अन्यथा {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (अचिन्हित) err);
        }
    } अन्यथा अगर (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (अचिन्हित) err);
    }

    // Populate domain IDs and names for all configured devices
    क्रम (माप_प्रकार i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

अटल व्योम ggml_hexagon_init(ggml_backend_reg * reg) {
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

    स्थिर अक्षर * str_verbose  = दो_पर्या("GGML_HEXAGON_VERBOSE");
    स्थिर अक्षर * str_opbatch  = दो_पर्या("GGML_HEXAGON_OPBATCH");
    स्थिर अक्षर * str_opqueue  = दो_पर्या("GGML_HEXAGON_OPQUEUE");
    स्थिर अक्षर * str_oppoll   = दो_पर्या("GGML_HEXAGON_OPPOLL");
    स्थिर अक्षर * str_opfusion = दो_पर्या("GGML_HEXAGON_OPFUSION");
    स्थिर अक्षर * str_rt_repack = दो_पर्या("GGML_HEXAGON_RT_REPACK");
    स्थिर अक्षर * str_opfilter = दो_पर्या("GGML_HEXAGON_OPFILTER");
    स्थिर अक्षर * str_profile  = दो_पर्या("GGML_HEXAGON_PROFILE");
    स्थिर अक्षर * str_etm      = दो_पर्या("GGML_HEXAGON_ETM");
    स्थिर अक्षर * str_nhvx     = दो_पर्या("GGML_HEXAGON_NHVX");
    स्थिर अक्षर * str_nhmx     = दो_पर्या("GGML_HEXAGON_NHMX");
    स्थिर अक्षर * str_mm_select = दो_पर्या("GGML_HEXAGON_MM_SELECT");
    स्थिर अक्षर * str_fa_select = दो_पर्या("GGML_HEXAGON_FA_SELECT");
    स्थिर अक्षर * str_fa_शीर्ष_split = दो_पर्या("GGML_HEXAGON_FA_HEAD_SPLIT");
    स्थिर अक्षर * str_gdn_select = दो_पर्या("GGML_HEXAGON_GDN_SELECT");
    स्थिर अक्षर * str_ar_select = दो_पर्या("GGML_HEXAGON_AR_SELECT");
    स्थिर अक्षर * str_ar_scatter = दो_पर्या("GGML_HEXAGON_AR_SCATTER");
    स्थिर अक्षर * str_ndev     = दो_पर्या("GGML_HEXAGON_NDEV");
    स्थिर अक्षर * str_arch     = दो_पर्या("GGML_HEXAGON_ARCH");
    स्थिर अक्षर * str_vmem     = दो_पर्या("GGML_HEXAGON_VMEM");
    स्थिर अक्षर * str_mbuf     = दो_पर्या("GGML_HEXAGON_MBUF");
    स्थिर अक्षर * str_optrace  = दो_पर्या("GGML_HEXAGON_OPTRACE");
    स्थिर अक्षर * str_hostbuf  = दो_पर्या("GGML_HEXAGON_HOSTBUF");
    स्थिर अक्षर * str_dma64    = दो_पर्या("GGML_HEXAGON_DMA64");
    स्थिर अक्षर * str_power_mode = दो_पर्या("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    अगर (!str_arch) {
        पूर्णांक err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        अगर (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } अन्यथा {
            अगर (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } अन्यथा अगर (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } अन्यथा {
        अगर (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = म_से_अदीर्घ(str_arch, शून्य, 0);
    }

    माप_प्रकार MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || म_से_प(str_dma64) != 0);

    स्वतः RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? नया std::regex(str_opfilter, RE_ICASE) : शून्य;
    opt_verbose   = str_verbose  ? म_से_प(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? म_से_अदीर्घ(str_opbatch, शून्य, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? म_से_अदीर्घ(str_opqueue, शून्य, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? म_से_अदीर्घ(str_optrace, शून्य, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? म_से_अदीर्घ(str_oppoll,  शून्य, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? म_से_प(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? म_से_प(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? म_से_प(str_profile)                      : 0;
    opt_etm       = str_etm      ? म_से_प(str_etm)                          : 0;
    अगर (str_power_mode) {
        पूर्णांक m = htp_power_mode_from_str(str_power_mode);
        अगर (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } अन्यथा {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? म_से_अदीर्घ(str_nhvx, शून्य, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? म_से_प(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? म_से_प(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? म_से_प(str_fa_select)                   : opt_fa_select;
    opt_fa_शीर्ष_split = str_fa_शीर्ष_split ? म_से_प(str_fa_शीर्ष_split)       : opt_fa_शीर्ष_split;
    opt_gdn_select    = str_gdn_select    ? म_से_प(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? म_से_प(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? म_से_प(str_ar_scatter)          : opt_ar_scatter;

    अगर (str_mbuf) {
        स्थिर अक्षर * p = str_mbuf;
        क्रम (पूर्णांक idx = 0; idx < 3 && p && *p; idx++) {
            जबतक (*p == ' ') p++;
            स्थिर अक्षर * comma = म_अक्षर(p, ',');
            माप_प्रकार len = comma ? (माप_प्रकार)(comma - p) : म_माप(p);
            जबतक (len > 0 && p[len - 1] == ' ') len--;
            अगर (len > 0) {
                std::string token(p, len);
                अगर (idx == 0) opt_mbuf_dyn    = parse_आकार(token.c_str());
                अगर (idx == 1) opt_mbuf_static = parse_आकार(token.c_str());
                अगर (idx == 2) opt_mbuf_योग  = parse_आकार(token.c_str());
            }
            अगर (!comma) अवरोध;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_आकार(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? म_से_प(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    स्थिर अक्षर * str_devices  = दो_पर्या("GGML_HEXAGON_DEVICES");
    अगर (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    अगर (str_devices && str_devices[0] != '\0') {
        bool is_single_संख्या = true;
        क्रम (पूर्णांक i = 0; str_devices[i] != '\0'; i++) {
            अगर (!है_अंक((अचिन्हित अक्षर)str_devices[i])) {
                is_single_संख्या = false;
                अवरोध;
            }
        }
        अगर (is_single_संख्या) {
            पूर्णांक n = म_से_प(str_devices);
            अगर (n < 1) n = 1;
            अगर (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            क्रम (माप_प्रकार i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (पूर्णांक)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.सफा();
            }
        } अन्यथा {
            std::string s_devices(str_devices);
            std::vector<std::string> वस्तुएँ;
            std::string curr_item;
            पूर्णांक bracket_depth = 0;
            क्रम (अक्षर ch : s_devices) {
                अगर (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } अन्यथा अगर (ch == ']') {
                    अगर (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } अन्यथा अगर (ch == ',' && bracket_depth == 0) {
                    माप_प्रकार s = curr_item.find_first_not_of(" \t\r\n");
                    माप_प्रकार e = curr_item.find_last_not_of(" \t\r\n");
                    अगर (s != std::string::npos) {
                        वस्तुएँ.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.सफा();
                } अन्यथा {
                    curr_item += ch;
                }
            }
            माप_प्रकार s = curr_item.find_first_not_of(" \t\r\n");
            माप_प्रकार e = curr_item.find_last_not_of(" \t\r\n");
            अगर (s != std::string::npos) {
                वस्तुएँ.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            क्रम (स्थिर स्वतः & item : वस्तुएँ) {
                माप_प्रकार b_open  = item.find('[');
                माप_प्रकार b_close = item.rfind(']');

                अगर (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    पूर्णांक virt = 0;
                    std::string phys_spec = content;
                    माप_प्रकार colon_pos = content.find(':');
                    अगर (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        प्रयास {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } पकड़ो (...) {
                            virt = 0;
                        }
                    } अन्यथा {
                        माप_प्रकार dev_colon = dev_name.find(':');
                        अगर (dev_colon != std::string::npos) {
                            प्रयास {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } पकड़ो (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<पूर्णांक> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    जबतक (std::लो_पंक्ति(pss, p_part, ',')) {
                        माप_प्रकार ps = p_part.find_first_not_of(" \t\r\n");
                        माप_प्रकार pe = p_part.find_last_not_of(" \t\r\n");
                        अगर (ps == std::string::npos) जारी;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        माप_प्रकार dash_pos = p_part.find('-');
                        अगर (dash_pos != std::string::npos) {
                            प्रयास {
                                पूर्णांक p_start = std::stoi(p_part.substr(0, dash_pos));
                                पूर्णांक p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                क्रम (पूर्णांक p = p_start; p <= p_end; p++) {
                                    अगर (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } पकड़ो (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } अन्यथा {
                            प्रयास {
                                पूर्णांक p = std::stoi(p_part);
                                अगर (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } पकड़ो (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    अगर (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    अगर (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        स्वतः & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.सफा();

                        क्रम (माप_प्रकार k = 1; k < phys_list.size(); k++) {
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
                } अन्यथा अगर (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    माप_प्रकार colon_pos = rest.find(':');
                    पूर्णांक phys = 0;
                    पूर्णांक virt = 0;
                    प्रयास {
                        अगर (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } अन्यथा {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } पकड़ो (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        जारी;
                    }

                    अगर (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.सफा();
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
        opt_device_configs[0].mdev_group.सफा();
    }

#अगर defined(__ANDROID__)
    अगर (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#पूर्ण_अगर

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    अगर (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            स्वतः v  = str_to_vec<uint32_t>(str_profile);
            चयन (v.size()) {
                हाल 1:  opt_profile = v[0]; वापस opt_pmu_evt; // mode with default pmu events
                हाल 8:  opt_profile = 2;    वापस v;           // mode with custom  pmu events
                शेष: opt_profile = 0;    वापस {};          // garbage input
            }}();
        अगर (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = नया ggml_hexagon_registry(reg);
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(व्योम) {
    अटल bool initialized = false;

    अटल ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ शून्य };

    {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        अगर (!initialized) {
            स्वतः nErr = htpdrv_init();
            अगर (nErr != AEE_SUCCESS) {
                वापस शून्य;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    वापस &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
