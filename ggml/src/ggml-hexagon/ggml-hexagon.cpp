#تعریف_کړه USE_MATH_تعریفوي

#ضمن <assert.h>
#ضمن <inttypes.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#ضمن <time.h>

#ضمن <atomic>
#ضمن <memory>
#ضمن <chrono>
#ضمن <mutex>
#ضمن <thread>
#ضمن <cstddef>
#ضمن <stdexcept>
#ضمن <string>
#ضمن <sstream>
#ضمن <iomanip>
#ضمن <unordered_set>
#ضمن <unordered_map>
#ضمن <regex>
#ضمن <queue>
#ضمن <deque>
#ضمن <algorithm>
#ضمن <cmath>
#ضمن <initializer_list>

#که_نه_تعریف_شوی M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#د_که_پای

#که_تعریف_شوی _WIN32
#    define WIN32_LEAN_AND_منځنۍ
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#وإلا
#    include <semaphore.h>
#    include <unistd.h>
#د_که_پای

#لارښوونه clang diagnostic ignored "-Wnested-anon-types"
#لارښوونه clang diagnostic ignored "-Wlanguage-extension-token"
#لارښوونه clang diagnostic ignored "-Wgnu-anonymous-struct"
#لارښوونه clang diagnostic ignored "-Wmicrosoft-enum-value"

#ضمن <AEEStdErr.h>
#ضمن <dspqueue.h>
#ضمن <rpcmem.h>

#تعریف_کړه GGML_COMMON_IMPL_CPP
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-common.h"
#ضمن "ggml-hexagon.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-quants.h"
#ضمن "htp-opnode.h"
#ضمن "htp-ops.h"
#ضمن "htp/matmul-ops.h"
#ضمن "htp/binary-ops.h"
#ضمن "htp/flash-attn-ops.h"
#ضمن "htp/unary-ops.h"
#ضمن "htp/get-rows-ops.h"
#ضمن "htp/set-rows-ops.h"
#ضمن "htp/softmax-ops.h"
#ضمن "htp/pool-ops.h"
#ضمن "htp/rope-ops.h"
#ضمن "htp/ssm-conv.h"
#ضمن "htp/gated-delta-net-ops.h"
#ضمن "htp/argsort-ops.h"
#ضمن "htp/concat-ops.h"
#ضمن "htp/cpy-ops.h"
#ضمن "htp/im2col-ops.h"
#ضمن "htp_iface.h"
#ضمن "htp-drv.h"
#ضمن "htp-repack.h"

کارول intvec  = std::vector<صحيح>;
کارول uintvec = std::vector<بینښه صحيح>;
کارول u32vec  = std::vector<uint32_t>;

#تعریف_کړه GGML_HEXAGON_MAX_SESSIONS          16

#تعریف_کړه GGML_HEXAGON_FENCE_خونه_اندازه       128

بنية ggml_hexagon_device_config {
    صحيح         physical_idx = 0;
    صحيح         virtual_idx  = 0;
    صحيح         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_ډله;
};

ساكن ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ساكن صحيح    opt_arch    = 0; // autodetect
ساكن size_t opt_ndev    = 1;
ساكن size_t opt_nhvx    = 0; // use all
ساكن صحيح    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ساكن size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ساكن صحيح    opt_etm     = 0;
ساكن صحيح    opt_verbose = 0;
ساكن صحيح    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ساكن منطقي   opt_hostbuf = خطأ;
ساكن منطقي   opt_dma64   = خطأ;
ساكن صحيح    opt_power_ډېرپېښ = HTP_POWER_ډېرپېښ_موده; // DCVS/HMX power profile requested at session start

ساكن ثابت std::pair<ثابت حرف *, صحيح> htp_power_ډېرپېښ_نومونه[] = {
    { "low_power_saver",            HTP_POWER_ډېرپېښ_ټیټ_POWER_SAVER },
    { "power_saver",                HTP_POWER_ډېرپېښ_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_ډېرپېښ_لوړ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_ډېرپېښ_ټیټ_متوازن },
    { "balanced",                   HTP_POWER_ډېرپېښ_متوازن },
    { "high_performance",           HTP_POWER_ډېرپېښ_لوړ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_ډېرپېښ_SUSTAINED_لوړ_PERFORMANCE },
    { "burst",                      HTP_POWER_ډېرپېښ_موده },
    { "default",                    HTP_POWER_ډېرپېښ_موده }, // flip this to change the default mode
};

ساكن صحيح htp_power_ډېرپېښ_from_str(ثابت حرف * s) {
    لكل (خپلکاری & e : htp_power_ډېرپېښ_نومونه) {
#إذا defined(_MSC_VER)
        إذا (_stricmp(s, e.لومړی) == 0) {
#وإلا
        إذا (strcasecmp(s, e.لومړی) == 0) {
#د_که_پای
            أرجع e.second;
        }
    }
    أرجع -1;
}

ساكن ثابت حرف * htp_power_ډېرپېښ_منزل_str(صحيح mode) {
    لكل (خپلکاری & e : htp_power_ډېرپېښ_نومونه) {
        إذا (e.second == mode) {
            أرجع e.لومړی;
        }
    }
    أرجع "unknown";
}

ساكن size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ساكن size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ساكن size_t opt_mbuf_ټول  = 0;                        // total buffer space limit (0 = unconstrained)

ساكن صحيح    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ساكن صحيح    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ساكن صحيح    opt_fa_سر_وېشل = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ساكن صحيح    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ساكن صحيح    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ساكن صحيح    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ساكن u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ساكن صحيح opt_opbatch  = 1280; // max number of ops in a batch
ساكن صحيح opt_opqueue  = 8;   // max number of pending batches
ساكن صحيح opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ساكن صحيح opt_oppoll   = 0;    // polling for batch completions
ساكن صحيح opt_opfusion  = 1;    // enable/disable op fusion
ساكن صحيح opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

تعداد ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_جمع_کړه = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_جمع_کړه   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ساكن مضمن منطقي ggml_hexagon_is_fusion_enabled(صحيح flag) {
    إذا (opt_opfusion <= 0) أرجع خطأ;
    إذا (opt_opfusion == 1) أرجع ریښتیا; // 1 enables all
    أرجع (opt_opfusion & flag) != 0;
}

ساكن std::regex* opt_opfilter = عدم; // regex of ops to not claim

#تعریف_کړه HEX_VERBOSE(...) \
    إذا (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ساكن ثابت حرف * حالت_منزل_str(uint32_t حالت) {
    بدل (حالت) {
        حالة HTP_حالت_OK:
            أرجع "OK";
        حالة HTP_حالت_NO_SUPPORT:
            أرجع "NO-SUPPORT";
        حالة HTP_حالت_INVAL_PARAMS:
            أرجع "INVAL-PARAMS";
        حالة HTP_حالت_VTCM_TOO_SMALL:
            أرجع "VTCM-TOO-SMALL";
        حالة HTP_حالت_دننی_ERR:
            أرجع "INTERNAL-ERROR";
        افتراضي:
            أرجع "UNKNOWN";
    }
}

// ** debug helpers

ساكن فراغ ggml_hexagon_dump_op_exec(ثابت std::string &sess_name, ثابت htp_opnode & غوټه, ثابت uint32_t req_flags) {
    إذا (!opt_verbose) أرجع;

    htp_opformat fmt(غوټه);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                غوټه.op_name().c_str(), fmt.نومونه, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ساكن فراغ ggml_hexagon_dump_op_supp(ثابت std::string &sess_name, ثابت بنية ggml_tensor * op, منطقي supp) {
    إذا (!opt_verbose) أرجع;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.نومونه, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ساكن ثابت حرف * htp_event_name(uint16_t id) {
    بدل (id) {
        حالة HTP_TRACE_EVT_DMA:            أرجع "DMA";
        حالة HTP_TRACE_EVT_HVX_COMP:       أرجع "HVX_COMP";
        حالة HTP_TRACE_EVT_HVX_A_QUANT:    أرجع "HVX_A_QUANT";
        حالة HTP_TRACE_EVT_HVX_A_PREP:     أرجع "HVX_A_PREP";
        حالة HTP_TRACE_EVT_HVX_W_DEQUANT:  أرجع "HVX_W_DEQUANT";
        حالة HTP_TRACE_EVT_HVX_W_PREP:     أرجع "HVX_W_PREP";
        حالة HTP_TRACE_EVT_HVX_O_PROC:     أرجع "HVX_O_PROC";
        حالة HTP_TRACE_EVT_HVX_FA_QK:      أرجع "HVX_QK_FA";
        حالة HTP_TRACE_EVT_HVX_FA_SFM:     أرجع "HVX_SFM_FA";
        حالة HTP_TRACE_EVT_HVX_FA_Q_PREP:  أرجع "HVX_Q_PREP";
        حالة HTP_TRACE_EVT_HVX_FA_K_PREP:  أرجع "HVX_K_PREP";
        حالة HTP_TRACE_EVT_HVX_FA_V_PREP:  أرجع "HVX_V_PREP";
        حالة HTP_TRACE_EVT_HVX_GDN_PREP:   أرجع "HVX_GDN_PREP";
        حالة HTP_TRACE_EVT_HVX_GDN_SOLVE:  أرجع "HVX_GDN_SOLVE";
        حالة HTP_TRACE_EVT_HVX_GDN_V_PREP: أرجع "HVX_GDN_V_PREP";
        حالة HTP_TRACE_EVT_HVX_GDN_D_PREP: أرجع "HVX_GDN_D_PREP";
        حالة HTP_TRACE_EVT_HVX_GDN_بهر:    أرجع "HVX_GDN_OUT";
        حالة HTP_TRACE_EVT_HVX_GDN_STATE:  أرجع "HVX_GDN_STATE";
        حالة HTP_TRACE_EVT_HVX_GDN_REM:    أرجع "HVX_GDN_REM";
        حالة HTP_TRACE_EVT_HMX_COMP:       أرجع "HMX_COMP";
        حالة HTP_TRACE_EVT_L2FLUSH:        أرجع "L2FLUSH";
        حالة HTP_TRACE_EVT_INIT:           أرجع "INIT";
        حالة HTP_TRACE_EVT_BUFF:           أرجع "BUFF";
        حالة HTP_TRACE_EVT_FENCE:          أرجع "FENCE";
        افتراضي:                           أرجع "UNKNOWN";
    }
}

ساكن فراغ ggml_hexagon_dump_op_prof(ثابت std::string &sess_name, ثابت htp_opnode & غوټه, ثابت htp_prof_desc & pd) {
    إذا (!opt_profile) أرجع;

    uint32_t op_usec = pd.usecs;
    uint32_t op_دورې = pd.دورې_stop - pd.دورې_پیل;
    ثابت uint32_t * pmu = pd.pmu;

    حرف pmu_str[256] = "";
    إذا (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, حجم(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(غوټه);
    عائم mhz = op_usec > 0 ? (عائم) op_دورې / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            غوټه.op_name().c_str(), fmt.نومونه, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_دورې, pd.دورې_پیل, mhz, pmu_str);
}

ساكن فراغ ggml_hexagon_dump_batch_prof(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp) {
    uint64_t batch_دورې = rsp.دورې_stop - rsp.دورې_پیل;
    عائم batch_mhz = rsp.usecs > 0 ? (عائم) batch_دورې / rsp.usecs : 0.0f;

    حرف evt_str[256] = "----";
    إذا (opt_profile == 3) {
        snprintf(evt_str, حجم(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (بینښه اوږد اوږد) batch_دورې, (بینښه اوږد اوږد) rsp.دورې_پیل, batch_mhz);
}

ساكن فراغ ggml_hexagon_dump_trace_events(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp,
                                           ثابت htp_trace_desc * trace_events, uint32_t n_traces) {
    إذا (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        لكل (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        لكل (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            لكل (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ثابت خپلکاری & e = trace_events[t * n_traces + idx];
                منطقي is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.دورې);
            }
        }
    }
}

تعداد ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_وزن    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

بنية ggml_hexagon_session;

ساكن فراغ ggml_hexagon_precompute_matmul_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_mm_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_fused_matmul_جمع_کړه_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * src2,
    ثابت بنية ggml_tensor * dst,
    بنية htp_mm_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_unary_params(
    ثابت بنية ggml_hexagon_session * sess,
    uint32_t op,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_unary_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_precompute_دوګونی_params(
    ثابت بنية ggml_hexagon_session * sess,
    uint32_t op,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_دوګونی_kernel_params * kparams,
    ثابت بنية ggml_tensor * src2 = تش_اشاره
);

ساكن فراغ ggml_hexagon_precompute_get_لیکې_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_get_لیکې_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_سېټ_لیکې_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_سېټ_لیکې_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_softmax_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_softmax_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_rope_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_rope_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_ssm_conv_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_ssm_conv_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_gated_delta_net_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_gdn_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_ترتیبول_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    منطقي is_سر_k,
    بنية htp_ترتیبول_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_pool_2d_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * dst,
    بنية htp_pool_2d_kernel_params * kparams,
    منطقي is_pool_1d
);

ساكن منطقي ggml_hexagon_precompute_concat_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_concat_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_precompute_cpy_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_copy_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_precompute_im2col_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_im2col_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_precompute_im2col_3d_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_im2col_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_fused_mmnx_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * act,
    int32_t n_weights,
    بنية htp_mm_kernel_params * kparams
);

ساكن فراغ ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * act,
    ثابت بنية ggml_tensor * dst,
    int32_t n_weights,
    بنية htp_mm_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_precompute_allreduce_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقي has_جمع_کړه,
    منطقي is_لیکه_bcast,
    منطقي is_shard_ok,
    بنية htp_allreduce_kernel_params * kparams
);

ساكن منطقي ggml_hexagon_لیکې_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride);
ساكن منطقي ggml_hexagon_matmul_can_collapse(ثابت بنية ggml_tensor * src0, ثابت بنية ggml_tensor * src1, ثابت بنية ggml_tensor * dst);
ساكن ggml_tensor ggml_hexagon_tensor_collapse_لیکې(ثابت بنية ggml_tensor * t);

ساكن منطقي mm_is_hmx_eligible(ثابت ggml_tensor * t);
ساكن htp_op_کوډ op_remap_منزل_htp(ثابت ggml_tensor * t);
ساكن منطقي is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت بنية htp_mm_kernel_params * kparams);
ساكن منطقي is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت بنية htp_mm_kernel_params * kparams);
ساكن منطقي is_mergeable_mul_mat(ثابت ggml_tensor * t);
ساكن منطقي is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);
ساكن منطقي is_mergeable_mul_mat_id(ثابت ggml_tensor * t);
ساكن منطقي is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);

// ** backend sessions

بنية ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_اندازه { 0 };
    uint32_t             flags { 0 };
};

ساكن مضمن منطقي ggml_hexagon_tensor_is_fuseable(ثابت بنية ggml_tensor * t) {
    إذا (!t->extra) أرجع خطأ;
    خپلکاری extra = (ثابت بنية ggml_hexagon_tensor_extra *) t->extra;
    أرجع (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ساكن مضمن منطقي ggml_hexagon_can_rt_repack(ثابت بنية ggml_tensor * t) {
    إذا (opt_rt_repack == 0 || !htp_is_tiled_وزن_tensor(t)) {
        أرجع خطأ;
    }
    إذا (t->بفر && ggml_backend_buft_is_host(t->بفر->buft)) {
        أرجع خطأ;
    }
    أرجع opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ساكن مضمن منطقي ggml_hexagon_tensor_is_tiled(ثابت بنية ggml_tensor * t) {
    ثابت خپلکاری * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    إذا (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        أرجع ریښتیا;
    }
    أرجع ggml_hexagon_can_rt_repack(t);
}

ساكن مضمن منطقي ggml_hexagon_needs_repack(ثابت بنية ggml_tensor * t) {
    ثابت خپلکاری * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    إذا (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        أرجع خطأ;
    }
    أرجع ggml_hexagon_can_rt_repack(t);
}

ساكن مضمن منطقي ggml_hexagon_tensors_overlap(ثابت بنية ggml_tensor * a, ثابت بنية ggml_tensor * b) {
    ثابت uintptr_t a0 = (uintptr_t) a->معلومات;
    ثابت uintptr_t b0 = (uintptr_t) b->معلومات;
    ثابت uintptr_t a1 = a0 + ggml_nbytes(a);
    ثابت uintptr_t b1 = b0 + ggml_nbytes(b);

    أرجع a0 < b1 && b0 < a1;
}

ساكن مضمن منطقي ggml_hexagon_can_لیکه_برخه_کول(ثابت بنية ggml_tensor * t) {
    إذا (t->ne[1] > 1 && (t->nb[1] & 127) != 0) أرجع خطأ;
    إذا (t->ne[2] > 1 && (t->nb[2] & 127) != 0) أرجع خطأ;
    إذا (t->ne[3] > 1 && (t->nb[3] & 127) != 0) أرجع خطأ;
    أرجع ریښتیا;
}

بنية htp_opnode;

بنية ggml_hexagon_opbatch;
بنية ggml_hexagon_opqueue;
بنية ggml_hexagon_shared_بفر;
بنية ggml_hexagon_fence_بفر;
بنية ggml_hexagon_session;
بنية ggml_backend_hexagon_device_context;

بنية ggml_hexagon_mdev_ډله {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

بنية ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    متطاير uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

بنية ggml_hexagon_event {
    ggml_hexagon_session * sess         = تش_اشاره;
    ggml_hexagon_session * fence_sess   = تش_اشاره;
    متطاير uint32_t *    fence_خونه   = تش_اشاره;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

بنية ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         کتار_id;
    صحيح              phys_idx;
    صحيح              virt_idx;
    منطقي             valid_session;
    منطقي             valid_handle;
    منطقي             valid_کتار;
    منطقي             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_کتار;

    std::unordered_map<صحيح, std::unique_ptr<ggml_hexagon_shared_بفر>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_تارونه   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_اندازه   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> وروستی_error{HTP_حالت_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_ډله                mdev;
    ggml_backend_dev_t                     dev       = تش_اشاره;
    ggml_backend_hexagon_device_context *  dev_ctx   = تش_اشاره;
    ggml_hexagon_fence_بفر *            fence_buf = تش_اشاره;

    ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev = تش_اشاره, uint32_t mdev_idx = 0, uint32_t mdev_شمېر = 0) noexcept(خطأ);
    ~ggml_hexagon_session() noexcept(ریښتیا);

    ثابت حرف* c_name() ثابت { أرجع name.c_str(); }

    فراغ allocate(ثابت ggml_hexagon_device_config & config) noexcept(خطأ);
    فراغ release() noexcept(ریښتیا);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    فراغ      free_fence(فراغ * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_خونه = تش_اشاره;
    std::unordered_map<uint64_t, متطاير uint32_t *> cpy_fence_slots;

    فراغ کتار_ته_ورزیاتول_mdev_ډله();
    فراغ کتار_ته_ورزیاتول_op(ثابت htp_opnode & غوټه);
    فراغ کتار_ته_ورزیاتول_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor = تش_اشاره, uint32_t fence_seq = 0);
    فراغ کتار_ته_ورزیاتول_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq = 0, منطقي wait = ریښتیا);
    فراغ کتار_ته_ورزیاتول_allreduce(ثابت ggml_tensor * dst, ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
                           ثابت std::vector<ثابت ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_ثبت = 0, uint32_t fence_seq_exit = 0);

    فراغ پیل_batch();
    فراغ flush_sync(منطقي all = ریښتیا);
    فراغ flush_async();
    فراغ flush_batch(size_t min_ops = 1);
    فراغ flush_peers();
    فراغ flush_pending(منطقي all = ریښتیا);

    ggml_hexagon_shared_بفر * mmap_tensor(ثابت ggml_tensor * t);
    منطقي clone_بفر(ثابت ggml_hexagon_shared_بفر*);
    فراغ release_بفر(ثابت ggml_hexagon_shared_بفر*);
    فراغ unclone_بفر(ثابت ggml_hexagon_shared_بفر*);

    فراغ جمع_کړه_peer(ggml_hexagon_session * peer) {
        إذا (هذا->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } وإلا {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

بنية ggml_backend_hexagon_device_context {
    صحيح                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = تش_اشاره;

    ggml_backend_بفر_type بفر_type       = {};
    ggml_backend_بفر_type host_بفر_type  = {};
    ggml_backend_بفر_type fence_بفر_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(صحيح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ثابت حرف * c_name() ثابت { أرجع config.name.c_str(); }

    ggml_hexagon_session * session() {
        إذا (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        أرجع sess.get();
    }
};

بنية ggml_backend_hexagon_بفر_type_context {
    ggml_backend_hexagon_بفر_type_context(ثابت std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        هذا->dev_ctx = dev_ctx;
        هذا->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

بنية ggml_hexagon_rpcmem_بلاک {
    uint8_t * بنسټ = تش_اشاره;
    صحيح       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_بلاک(size_t size) {
        بنسټ = (uint8_t *) rpcmem_alloc2(RPCMEM_ډېرۍ_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        إذا (!بنسټ) {
            ارم std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_منزل_fd(بنسټ);
        إذا (fd < 0) {
            rpcmem_free(بنسټ);
            ارم std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        هذا->size = size;
    }

    ~ggml_hexagon_rpcmem_بلاک() {
        إذا (بنسټ) {
            rpcmem_free(بنسټ);
        }
    }
};

بنية ggml_hexagon_shared_بفر {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_بلاک> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    منطقي     mapped;
    منطقي     pinned;
    منطقي     extended;

    ثابت حرف * c_name() ثابت { أرجع sess->c_name(); }
    uint8_t *    بنسټ()   ثابت { أرجع mem ? mem->بنسټ : تش_اشاره; }
    size_t       size()   ثابت { أرجع mem ? mem->size : 0;  }
    صحيح          fd()     ثابت { أرجع mem ? mem->fd   : -1; }

    فراغ mmap(منطقي extended = خطأ) {
        إذا (!هذا->mem)   أرجع;
        إذا (هذا->mapped) أرجع;

        GGML_ASSERT(!هذا->pinned || !extended);

        هذا->extended = extended;
        fastrpc_map_flags flags = هذا->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        صحيح err = fastrpc_mmap(sess->domain_id, fd(), (فراغ *) بنسټ(), 0, size(), flags);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (بینښه) err);
            ارم std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (فراغ *) بنسټ(), size(), fd(), pinned, extended);

        هذا->mapped = ریښتیا;
    }

    فراغ unmap() {
        إذا (!هذا->mapped) أرجع;

        إذا (!هذا->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        إذا (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (فراغ *) بنسټ(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (فراغ *) بنسټ(), size(), fd());

        هذا->mapped = خطأ;
    }

    فراغ alloc(size_t size) {
        إذا (هذا->mem) أرجع;

        هذا->mem = std::make_shared<ggml_hexagon_rpcmem_بلاک>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (فراغ *) بنسټ(), هذا->size(), fd(), (صحيح) pinned);
        إذا (هذا->pinned) {
            mmap();
        }
    }

    فراغ حرر() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (فراغ *) بنسټ(), size(), fd());
        هذا->mem  = تش_اشاره;
    }

    ggml_hexagon_shared_بفر(ggml_hexagon_session * sess, size_t size, منطقي pinned = خطأ) {
        هذا->sess     = sess;
        هذا->mapped   = خطأ;
        هذا->pinned   = pinned;
        هذا->extended = خطأ;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_دننه_واټن = (size + 4095) & ~4095;
        size_t ټول_اندازه   = guard_دننه_واټن + 4096;
        إذا (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            ټول_اندازه = (ټول_اندازه + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(ټول_اندازه);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_بفر(ggml_hexagon_session * sess, ثابت ggml_hexagon_shared_بفر & other) {
        هذا->sess   = sess;
        هذا->mem    = other.mem;
        هذا->mapped = خطأ;
        هذا->pinned = other.pinned;
        هذا->extended = other.extended;
    }

    ~ggml_hexagon_shared_بفر() {
        حرر();
        لكل (خپلکاری * extra : tensor_extra) {
            ړنګول extra;
        }
    }
};

بنية ggml_hexagon_fence_بفر : عام ggml_hexagon_shared_بفر {
    uint32_t              خونه_شمېر = 0;
    uint32_t              خونه_سر  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_بفر   backend_بفر{};

    ggml_hexagon_fence_بفر(ggml_hexagon_session * sess, ggml_backend_بفر_type_t buft, size_t size)
        : ggml_hexagon_shared_بفر(sess, size, خطأ /* pinned */),
          خونه_شمېر(size / GGML_HEXAGON_FENCE_خونه_اندازه),
          خونه_سر(0) {
        backend_بفر.buft    = buft;
        backend_بفر.context = static_cast<ggml_hexagon_shared_بفر *>(هذا);
        backend_بفر.size    = size;
        mmap(خطأ);
    }

    uint8_t * alloc_خونه(uint32_t n_slots = 1) {
        uint8_t * ptr = تش_اشاره;
        إذا (n_slots == 1 && !free_slots.تشه()) {
            uint32_t خونه = free_slots.back();
            free_slots.pop_back();
            ptr = بنسټ() + (size_t) خونه * GGML_HEXAGON_FENCE_خونه_اندازه;
        } وإلا إذا (خونه_سر + n_slots <= خونه_شمېر) {
            uint32_t خونه = خونه_سر;
            خونه_سر += n_slots;
            ptr = بنسټ() + (size_t) خونه * GGML_HEXAGON_FENCE_خونه_اندازه;
        }
        إذا (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_خونه_اندازه);
        }
        أرجع ptr;
    }

    فراغ free_خونه(فراغ * ptr, uint32_t n_slots = 1) {
        إذا (!ptr) أرجع;
        uint32_t خونه = ((uint8_t *) ptr - بنسټ()) / GGML_HEXAGON_FENCE_خونه_اندازه;
        لكل (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(خونه + i);
        }
    }
};

مضمن uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_خونه(n_slots);
    GGML_ASSERT(ptr);
    أرجع ptr;
}

مضمن فراغ ggml_hexagon_session::free_fence(فراغ * ptr, uint32_t n_slots) {
    إذا (fence_buf) {
        fence_buf->free_خونه(ptr, n_slots);
    }
}

ساكن فراغ ggml_backend_hexagon_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    sbuf->sess->unclone_بفر(sbuf);
    ړنګول sbuf;
}

ساكن فراغ * ggml_backend_hexagon_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    أرجع sbuf->بنسټ();
}

ساكن تعداد ggml_حالت ggml_backend_hexagon_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    خپلکاری sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (فراغ *) sbuf->بنسټ(), tensor->معلومات, ggml_nbytes(tensor));

    خپلکاری extra = جديد ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_hexagon_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                                   ggml_tensor *         tensor,
                                                   ثابت فراغ *          معلومات,
                                                   size_t                دننه_واټن,
                                                   size_t                size) {
    خپلکاری extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خپلکاری sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خپلکاری sess  = sbuf->sess;

    إذا (ggml_backend_بفر_get_usage(بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_وزن;
        إذا (htp_is_tiled_وزن_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, معلومات, دننه_واټن, size, (صحيح) بفر->usage, extra->flags);

    إذا ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        حافظه_نقل((حرف *) tensor->معلومات + دننه_واټن, معلومات, size);
        أرجع;
    }

    إذا (دننه_واټن == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.تشه()) {
        htp_repack_tensor_tiled(tensor, معلومات, size);
        أرجع;
    }

    إذا (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    حافظه_نقل(extra->shadow_buf.معلومات() + دننه_واټن, معلومات, size);
    extra->shadow_اندازه += size;

    إذا (extra->shadow_اندازه >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.معلومات(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_اندازه = 0;
    }
}

ساكن فراغ ggml_backend_hexagon_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                   ثابت ggml_tensor *   tensor,
                                                   فراغ *                معلومات,
                                                   size_t                دننه_واټن,
                                                   size_t                size) {
    خپلکاری extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خپلکاری sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خپلکاری sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, معلومات, دننه_واټن, size, (صحيح) بفر->usage, extra->flags);

    إذا ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        حافظه_نقل(معلومات, (ثابت حرف *) tensor->معلومات + دننه_واټن, size);
        أرجع;
    }

    htp_unpack_tensor_tiled(معلومات, tensor, دننه_واټن, size);
}

ساكن منطقي ggml_backend_hexagon_بفر_cpy_tensor(ggml_backend_بفر_t      بفر,
                                                   ثابت بنية ggml_tensor * src,
                                                   بنية ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    أرجع خطأ;

    GGML_ناکارول_شوی(بفر);
    GGML_ناکارول_شوی(src);
    GGML_ناکارول_شوی(dst);
}

ساكن فراغ ggml_backend_hexagon_بفر_سېټ_tensor_2d(ggml_backend_بفر_t بفر,
                                                          ggml_tensor *         tensor,
                                                          ثابت فراغ *          معلومات,
                                                          size_t                دننه_واټن,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_معلومات) {
    خپلکاری extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خپلکاری sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خپلکاری sess  = sbuf->sess;

    إذا (ggml_backend_بفر_get_usage(بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_وزن;
        إذا (htp_is_tiled_وزن_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات, (صحيح) بفر->usage, extra->flags);

    إذا ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        لكل (size_t i = 0; i < n_copies; i++) {
            حافظه_نقل((uint8_t *) tensor->معلومات + دننه_واټن + i * stride_tensor, (ثابت uint8_t *) معلومات + i * stride_معلومات, size);
        }
        أرجع;
    }

    إذا (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    لكل (size_t i = 0; i < n_copies; i++) {
        حافظه_نقل(extra->shadow_buf.معلومات() + دننه_واټن + i * stride_tensor, (ثابت uint8_t *) معلومات + i * stride_معلومات, size);
    }
    extra->shadow_اندازه += n_copies * size;

    إذا (extra->shadow_اندازه >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.معلومات(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_اندازه = 0;
    }
}

ساكن فراغ ggml_backend_hexagon_بفر_get_tensor_2d(ggml_backend_بفر_t بفر,
                                                      ثابت ggml_tensor *   tensor,
                                                      فراغ *                معلومات,
                                                      size_t                دننه_واټن,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_معلومات) {
    خپلکاری extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خپلکاری sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خپلکاری sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات, (صحيح) بفر->usage);

    إذا ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        لكل (size_t i = 0; i < n_copies; i++) {
            حافظه_نقل((uint8_t *)معلومات + i * stride_معلومات, (ثابت uint8_t *)tensor->معلومات + دننه_واټن + i * stride_tensor, size);
        }
        أرجع;
    }

    size_t لنډمهاله_اندازه      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t ټوټه_اندازه     = tensor->ne[1] * ggml_لیکه_اندازه(tensor->type, tensor->ne[0]);
    size_t ټوټه_دننه_واټن   = دننه_واټن % ټوټه_اندازه;
    size_t لیکه_اندازه_بایټونه = ggml_لیکه_اندازه(tensor->type, tensor->ne[0]);

    GGML_ASSERT((ټوټه_دننه_واټن % لیکه_اندازه_بایټونه) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((لنډمهاله_اندازه % لیکه_اندازه_بایټونه)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((دننه_واټن + لنډمهاله_اندازه) <= ggml_nbytes(tensor));

    std::vector<uint8_t> لنډمهاله_buf(لنډمهاله_اندازه);
    htp_unpack_tensor_tiled(لنډمهاله_buf.معلومات(), tensor, دننه_واټن, لنډمهاله_اندازه);

    لكل (size_t i = 0; i < n_copies; i++) {
        حافظه_نقل((uint8_t *) معلومات + i * stride_معلومات, لنډمهاله_buf.معلومات() + i * stride_tensor, size);
    }
}

ساكن فراغ ggml_backend_hexagon_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    خپلکاری sbuf = (ggml_hexagon_shared_بفر *) بفر->context;
    خپلکاری sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (فراغ *) sbuf->بنسټ(), sbuf->size());
    memset(sbuf->بنسټ(), value, sbuf->size());
}

ساكن ggml_backend_بفر_i ggml_backend_hexagon_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_hexagon_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_hexagon_بفر_init_tensor,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_hexagon_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_بفر_سېټ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_بفر_clear,
    /* .reset           = */ عدم,
};

// ** backend buffer type

ساكن فراغ ggml_backend_hexagon_host_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                                        ggml_tensor *         tensor,
                                                        ثابت فراغ *          معلومات,
                                                        size_t                دننه_واټن,
                                                        size_t                size) {
    حافظه_نقل((حرف *) tensor->معلومات + دننه_واټن, معلومات, size);
    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_hexagon_host_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                        ثابت ggml_tensor *   tensor,
                                                        فراغ *                معلومات,
                                                        size_t                دننه_واټن,
                                                        size_t                size) {
    حافظه_نقل(معلومات, (ثابت حرف *) tensor->معلومات + دننه_واټن, size);
    GGML_ناکارول_شوی(بفر);
}

ساكن ggml_backend_بفر_i ggml_backend_hexagon_host_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_hexagon_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_hexagon_بفر_init_tensor,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_hexagon_host_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_hexagon_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_بفر_clear,
    /* .reset           = */ عدم,
};

// ** backend buffer type

ساكن ثابت حرف * ggml_backend_hexagon_بفر_type_name(ggml_backend_بفر_type_t بفر_type) {
    أرجع static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_hexagon_بفر_type_alloc_بفر(
            ggml_backend_بفر_type_t بفر_type, size_t size) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->dev_ctx;
    خپلکاری sess    = dev_ctx->session();
    إذا (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        أرجع تش_اشاره;
    }
    حاول {
        ggml_hexagon_shared_بفر * sbuf = جديد ggml_hexagon_shared_بفر(sess, size, خطأ);
        أرجع ggml_backend_بفر_init(بفر_type, ggml_backend_hexagon_بفر_interface, sbuf, size);
    } امسك (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        أرجع تش_اشاره;
    }
}

ساكن ggml_backend_بفر_t ggml_backend_hexagon_host_بفر_type_alloc_بفر(
            ggml_backend_بفر_type_t بفر_type, size_t size) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->dev_ctx;
    خپلکاری sess    = dev_ctx->session();
    إذا (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        أرجع تش_اشاره;
    }
    حاول {
        ggml_hexagon_shared_بفر * sbuf = جديد ggml_hexagon_shared_بفر(sess, size, خطأ);
        أرجع ggml_backend_بفر_init(بفر_type, ggml_backend_hexagon_host_بفر_interface, sbuf, size);
    } امسك (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        أرجع تش_اشاره;
    }
}

ساكن size_t ggml_backend_hexagon_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 128;  // HVX alignment
    GGML_ناکارول_شوی(buft);
}

ساكن size_t ggml_backend_hexagon_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت بنية ggml_tensor * t) {
    إذا (!ggml_backend_buft_is_host(buft) && htp_is_tiled_وزن_tensor(t)) {
        int64_t ne0 = hex_پړاو_پورته(t->ne[0], 32);
        int64_t ne1 = hex_پړاو_پورته(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_اندازه = htp_tiled_لیکه_اندازه(t->type, ne0) * ne1 * ne2 * ne3;
        أرجع std::max(tiled_اندازه, ggml_nbytes(t));
    }
    أرجع ggml_nbytes(t);
}

ساكن size_t parse_اندازه(ثابت حرف * str, size_t default_unit = 1024 * 1024) {
    إذا (!str || str[0] == '\0') {
        أرجع 0;
    }
    حرف * end = عدم;
    مزدوج val = strtod(str, &end);
    إذا (val < 0) {
        أرجع 0;
    }
    إذا (end && *end) {
        طالما (*end == ' ') end++;
        إذا (*end == 'k' || *end == 'K') {
            أرجع (size_t) (val * 1024);
        } وإلا إذا (*end == 'm' || *end == 'M') {
            أرجع (size_t) (val * 1024 * 1024);
        } وإلا إذا (*end == 'g' || *end == 'G') {
            أرجع (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    أرجع (size_t) (val * default_unit);
}

ساكن size_t ggml_backend_hexagon_بفر_type_get_max_اندازه(ggml_backend_بفر_type_t buft) {
    أرجع opt_mbuf_dyn;
    GGML_ناکارول_شوی(buft);
}

ساكن منطقي ggml_backend_hexagon_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;
    GGML_ناکارول_شوی(buft);
}

ساكن منطقي ggml_backend_hexagon_host_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع ریښتیا;
    GGML_ناکارول_شوی(buft);
}

بنية ggml_backend_hexagon_alloc_بفر_n_plan_توکی {
    size_t size;
    صحيح    لومړی;
    صحيح    وروستی;
};

کارول ggml_backend_hexagon_alloc_بفر_n_plan_t = std::vector<ggml_backend_hexagon_alloc_بفر_n_plan_توکی>;

ساكن ثابت حرف * ggml_hexagon_kv_layer_suffix(ثابت بنية ggml_tensor * t) {
    إذا (مزي_پرتله(t->name, "cache_", 6) != 0) {
        أرجع عدم;
    }
    ثابت حرف * p = مزي_پلټنه(t->name, "_l");
    إذا (!p || !که_عدد((بینښه حرف)p[2])) {
        أرجع عدم;
    }
    أرجع p;
}

بنية ggml_backend_hexagon_alloc_unit {
    size_t size;
    صحيح    لومړی;
    صحيح    وروستی;
};

ساكن ggml_backend_hexagon_alloc_بفر_n_plan_t ggml_backend_hexagon_alloc_بفر_n_plan(
        ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ggml_backend_hexagon_alloc_بفر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    صحيح i = 0;
    طالما (i < n_tensors) {
        بنية ggml_tensor * t = tensors[i];
        size_t unit_اندازه = 0;
        صحيح unit_لومړی = i;
        صحيح unit_وروستی = i + 1;

        إذا (t->معلومات == عدم && t->view_src == عدم) {
            unit_اندازه += GGML_PAD(ggml_backend_buft_get_alloc_اندازه(buft, t), alignment);
        }

        ثابت حرف * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        طالما (unit_وروستی < n_tensors) {
            بنية ggml_tensor * بل = tensors[unit_وروستی];

            إذا (بل->view_src != عدم) {
                unit_وروستی++;
                استمر;
            }

            إذا (layer_suffix != عدم) {
                ثابت حرف * بل_suffix = ggml_hexagon_kv_layer_suffix(بل);
                إذا (بل_suffix != عدم && strcmp(layer_suffix, بل_suffix) == 0) {
                    إذا (بل->معلومات == عدم) {
                        unit_اندازه += GGML_PAD(ggml_backend_buft_get_alloc_اندازه(buft, بل), alignment);
                    }
                    unit_وروستی++;
                    استمر;
                }
            }

            اكسر;
        }

        units.push_back({ unit_اندازه, unit_لومړی, unit_وروستی });
        i = unit_وروستی;
    }

    size_t cur_buf_اندازه  = 0;
    صحيح    cur_buf_لومړی = 0;

    لكل (ثابت خپلکاری & unit : units) {
        إذا (unit.size == 0) {
            استمر;
        }

        إذا (cur_buf_اندازه > 0 && (cur_buf_اندازه + unit.size) > max_size) {
            plan.push_back({ cur_buf_اندازه, cur_buf_لومړی, unit.لومړی });
            cur_buf_اندازه  = 0;
            cur_buf_لومړی = unit.لومړی;
        }

        cur_buf_اندازه += unit.size;
    }

    إذا (cur_buf_اندازه > 0) {
        plan.push_back({ cur_buf_اندازه, cur_buf_لومړی, n_tensors });
    }

    أرجع plan;
}

ساكن ggml_backend_بفر_t ggml_backend_hexagon_بفر_type_alloc_بفر_n(
        ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_بفر_n_plan_t plan = ggml_backend_hexagon_alloc_بفر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_بفر_t> buffers;
    buffers.reserve(plan.size());

    لكل (ثابت خپلکاری & توکی : plan) {
        ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, توکی.size);
        إذا (بفر == عدم) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), توکی.size);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            أرجع عدم;
        }

        بنية ggml_tallocr tallocr = ggml_tallocr_new(بفر);

        بنية ggml_tensor * t_failed = عدم;
        لكل (صحيح j = توکی.لومړی; j < توکی.وروستی; j++) {
            بنية ggml_tensor * t = tensors[j];
            إذا (t->معلومات == عدم) {
                إذا (t->view_src == عدم) {
                    إذا (ggml_tallocr_alloc(&tallocr, t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                } وإلا إذا (t->بفر == عدم) {
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            } وإلا {
                إذا (t->view_src != عدم && t->بفر == عدم) {
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            }
        }
        إذا (t_failed != عدم) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            ggml_backend_بفر_free(بفر);
            أرجع عدم;
        }

        buffers.push_back(بفر);
    }

    إذا (buffers.تشه()) {
        أرجع عدم;
    }

    إذا (buffers.size() == 1) {
        أرجع buffers[0];
    }

    أرجع ggml_backend_multi_بفر_alloc_بفر(buffers.معلومات(), buffers.size());
}

ساكن size_t ggml_backend_hexagon_بفر_type_get_alloc_اندازه_n(
        ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_بفر_n_plan_t plan = ggml_backend_hexagon_alloc_بفر_n_plan(buft, tensors, n_tensors);

    size_t ټول = 0;
    لكل (ثابت خپلکاری & توکی : plan) {
        ټول += توکی.size;
    }
    أرجع ټول;
}

ساكن ggml_backend_بفر_type_i ggml_backend_hexagon_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_hexagon_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_بفر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_hexagon_بفر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_بفر_type_get_alloc_اندازه_n,
    /* .is_host             = */ ggml_backend_hexagon_بفر_type_is_host,
};

ساكن ggml_backend_بفر_type_i ggml_backend_hexagon_host_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_hexagon_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_بفر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_hexagon_بفر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_بفر_type_get_alloc_اندازه_n,
    /* .is_host             = */ ggml_backend_hexagon_host_بفر_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(صحيح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    بفر_type.device  = dev;
    بفر_type.iface   = ggml_backend_hexagon_بفر_type_interface;
    بفر_type.context = جديد ggml_backend_hexagon_بفر_type_context(config.name, هذا);

    host_بفر_type.device  = dev;
    host_بفر_type.iface   = ggml_backend_hexagon_host_بفر_type_interface;
    host_بفر_type.context = جديد ggml_backend_hexagon_بفر_type_context(config.name + "-HOST", هذا);

    fence_بفر_type.device  = dev;
    fence_بفر_type.iface   = ggml_backend_hexagon_بفر_type_interface;
    fence_بفر_type.context = جديد ggml_backend_hexagon_بفر_type_context(config.name + "-FENCE", هذا);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    ړنګول static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type.context);
    ړنګول static_cast<ggml_backend_hexagon_بفر_type_context *>(host_بفر_type.context);
    ړنګول static_cast<ggml_backend_hexagon_بفر_type_context *>(fence_بفر_type.context);
}

ساكن منطقي ggml_backend_بفر_is_hexagon(ثابت بنية ggml_backend_بفر * b) {
    أرجع b->buft->iface.get_alignment == ggml_backend_hexagon_بفر_type_get_alignment;
}

بنية ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<صحيح, صحيح>                b_map; // buffer fd   to index
    std::unordered_map<ثابت ggml_tensor*, صحيح> t_map; // tensor ptr  to index
    std::unordered_multimap<فراغ*, صحيح>         d_map; // tensor data to index

    بینښه صحيح n_bufs;     // num buffers in the batch
    بینښه صحيح n_tens;     // num tensors ...
    بینښه صحيح n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    بینښه صحيح n_bufs_max;
    بینښه صحيح n_tens_max;
    بینښه صحيح n_ops_max;
    size_t       b_vmem_max;

    فراغ reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_اندازه, size_t max_vmem) {
        هذا->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_اندازه;
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

    منطقي تشه() ثابت { أرجع n_ops == 0; }

    // add buffer and return its index
    صحيح جمع_کړه_بفر(ggml_hexagon_shared_بفر * sbuf) {
        // Lookup by fd
        خپلکاری it = b_map.find(sbuf->fd());
        إذا (it != b_map.end()) { أرجع it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        صحيح bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.بنسټ  = (uint64_t) sbuf->بنسټ();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        إذا (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (فراغ*) sbuf->بنسټ(), (size_t) b.size, b_vmem);

        أرجع bi;
    }

    منطقي same_shape(ثابت htp_tensor * h, ثابت ggml_tensor * t) ثابت {
        خپلکاری extra = (ggml_hexagon_tensor_extra *) t->extra;

        ثابت منطقي is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ثابت int64_t ne0_padded = hex_پړاو_پورته(t->ne[0], 32);
        ثابت int64_t ne1_padded = hex_پړاو_پورته(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_لیکه_اندازه(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? حجم(ggml_fp16_t) : t->nb[0];
        تعداد ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        أرجع (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    صحيح جمع_کړه_tensor(ثابت ggml_tensor * t) {
        خپلکاری extra = (ggml_hexagon_tensor_extra *) t->extra;
        خپلکاری sbuf  = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);

        // First lookup by tensor data
        خپلکاری range = d_map.equal_range(t->معلومات);
        لكل (خپلکاری it = range.لومړی; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            إذا (same_shape(h, t)) { أرجع it->second; }
        }

        // Lookup by tensor ptr
        خپلکاری it = t_map.find(t);
        إذا (it != t_map.end()) { أرجع it->second; }

        // Add new tensor to the batch
        صحيح ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->معلومات, ti});

        uint64_t t_دننه_واټن = (uint8_t *) t->معلومات - sbuf->بنسټ();
        size_t   t_اندازه   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = جمع_کړه_بفر(sbuf);
        h.ti    = ti;
        h.معلومات  = t_دننه_واټن;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ثابت منطقي is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        إذا (is_tiled) {
            إذا (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = حجم(ggml_fp16_t);
            } وإلا {
                h.nb[0] = t->nb[0];
            }
            ثابت int64_t ne0_padded = hex_پړاو_پورته(t->ne[0], 32);
            ثابت int64_t ne1_padded = hex_پړاو_پورته(t->ne[1], 32);
            h.nb[1] = htp_tiled_لیکه_اندازه(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_اندازه  = h.size;
        } وإلا {
            h.size  = t_اندازه;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        إذا ((extra->flags & GGML_HEXAGON_TENSOR_وزن) != 0) {
            h.flags |= HTP_TENSOR_وزن;
        }
        إذا ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        إذا ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (فراغ*) t->معلومات, (size_t) t_دننه_واټن, t_اندازه, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        أرجع ti;
    }

    منطقي fit_op(ثابت htp_opnode & غوټه) ثابت {
        إذا (n_ops >= n_ops_max) أرجع خطأ;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        صحيح کتل_شوی_bufs[HTP_OP_MAX_BUFS];
        صحيح n_کتل_شوی_bufs = 0;

        خپلکاری fit_tensor = [&](ثابت ggml_tensor *t) {
            إذا (!t) أرجع;
            إذا (!t_map.count(t)) {
                extra_tens++;

                خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
                صحيح fd = sbuf->fd();
                إذا (!b_map.count(fd)) {
                    لكل (صحيح i = 0; i < n_کتل_شوی_bufs; i++) {
                        إذا (کتل_شوی_bufs[i] == fd) أرجع;
                    }
                    إذا (n_کتل_شوی_bufs < HTP_OP_MAX_BUFS) {
                        کتل_شوی_bufs[n_کتل_شوی_bufs++] = fd;
                    }
                    إذا (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        لكل (ثابت خپلکاری * src : غوټه.get_inputs()) {
            fit_tensor(src);
        }
        لكل (ثابت خپلکاری * output : غوټه.get_outputs()) {
            fit_tensor(output);
        }

        إذا ((extra_bufs + n_bufs) > n_bufs_max) أرجع خطأ;
        إذا ((extra_tens + n_tens) > n_tens_max) أرجع خطأ;
        إذا ((extra_vmem + b_vmem) > b_vmem_max) أرجع خطأ;

        أرجع ریښتیا;
    }

    // assumes that fit_op() was called first and returned true
    فراغ جمع_کړه_op(ثابت htp_opnode & غوټه) {
        // Add new op

        بینښه صحيح n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = غوټه;

        htp_op_desc &o = h_ops[n];
        حافظه_نقل(o.params,        غوټه.غوټه->op_params, حجم(غوټه.غوټه->op_params));
        حافظه_نقل(o.kernel_params, غوټه.kernel_params,   حجم(o.kernel_params));
        o.opcode = غوټه.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        خپلکاری inputs = غوټه.get_inputs();
        لكل (بینښه صحيح i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? جمع_کړه_tensor(inputs[i]) : 0xffff;
        }

        خپلکاری outputs = غوټه.get_outputs();
        لكل (بینښه صحيح i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? جمع_کړه_tensor(outputs[i]) : 0xffff;
        }
    }

    فراغ ترتیبول_buffers() {
        إذا (n_bufs <= 1) أرجع;

        std::vector<صحيح> ترتیب(n_bufs);
        لكل (بینښه صحيح i = 0; i < n_bufs; i++) { ترتیب[i] = (صحيح) i; }

        std::stable_sort(ترتیب.begin(), ترتیب.end(), [&](صحيح a, صحيح b) {
            أرجع h_bufs[a].size > h_bufs[b].size;
        });

        منطقي already_sorted = ریښتیا;
        لكل (بینښه صحيح i = 0; i < n_bufs; i++) {
            إذا (ترتیب[i] != (صحيح) i) {
                already_sorted = خطأ;
                اكسر;
            }
        }
        إذا (already_sorted) أرجع;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        لكل (بینښه صحيح new_bi = 0; new_bi < n_bufs; new_bi++) {
            صحيح old_bi = ترتیب[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        لكل (بینښه صحيح i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        لكل (بینښه صحيح i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    فراغ update_mdev_ډله(uint32_t mdev_idx) {
        إذا (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_ډله) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    منطقي try_fuse_common(std::initializer_list<ثابت ggml_tensor *> tensors) ثابت {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        صحيح کتل_شوی_bufs[HTP_OP_MAX_BUFS];
        صحيح n_کتل_شوی_bufs = 0;

        لكل (ثابت خپلکاری * t : tensors) {
            إذا (!t || t_map.count(t)) {
                استمر;
            }
            extra_tens++;
            خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
            صحيح fd = sbuf->fd();
            إذا (!b_map.count(fd)) {
                منطقي وموندل = خطأ;
                لكل (صحيح i = 0; i < n_کتل_شوی_bufs; i++) {
                    إذا (کتل_شوی_bufs[i] == fd) {
                        وموندل = ریښتیا;
                        اكسر;
                    }
                }
                إذا (!وموندل) {
                    إذا (n_کتل_شوی_bufs < HTP_OP_MAX_BUFS) {
                        کتل_شوی_bufs[n_کتل_شوی_bufs++] = fd;
                    }
                    إذا (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        إذا ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            أرجع خطأ;
        }

        أرجع ریښتیا;
    }

    منطقي try_fuse_common(ثابت ggml_tensor * t1, ثابت ggml_tensor * t2) ثابت {
        أرجع try_fuse_common({t1, t2});
    }

    منطقي try_fuse_allreduce_جمع_کړه(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0 || opt_ar_select != 2) أرجع خطأ;
        إذا (غوټه.opcode != HTP_OP_جمع_کړه) أرجع خطأ;

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];
        إذا (وروستی_غوټه.opcode != HTP_OP_ALLREDUCE) أرجع خطأ;

        خپلکاری * ar_kparams = (بنية htp_allreduce_kernel_params *) وروستی_غوټه.kernel_params;
        ثابت uint32_t rank    = (uint32_t) ar_kparams->rank;
        ثابت uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ثابت ggml_tensor * ar_local = وروستی_غوټه.inputs[rank];
        ثابت ggml_tensor * جمع_کړه_src0 = غوټه.src0();
        ثابت ggml_tensor * جمع_کړه_src1 = غوټه.src1();
        ثابت ggml_tensor * جمع_کړه_dst  = غوټه.dst();

        إذا (!ggml_hexagon_tensor_is_fuseable(ar_local)) أرجع خطأ;

        ثابت ggml_tensor * res_tensor;
        إذا (جمع_کړه_src0 == ar_local || جمع_کړه_src0->معلومات == ar_local->معلومات) {
            res_tensor = جمع_کړه_src1;
        } وإلا إذا (جمع_کړه_src1 == ar_local || جمع_کړه_src1->معلومات == ar_local->معلومات) {
            res_tensor = جمع_کړه_src0;
        } وإلا {
            أرجع خطأ;
        }

        إذا (ar_local->type != res_tensor->type) أرجع خطأ;

        ثابت منطقي is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ثابت منطقي is_لیکه_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        إذا (!is_same_shape && !is_لیکه_bcast) أرجع خطأ;

        إذا (is_same_shape) {
            إذا (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                أرجع خطأ;
            }
            إذا (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                أرجع خطأ;
            }
        }
        إذا (ggml_is_contiguous(ar_local) != ggml_is_contiguous(جمع_کړه_dst)) {
            أرجع خطأ;
        }

        // scatter is only valid for in-place add within max outputs
        ثابت منطقي is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && جمع_کړه_dst->معلومات == ar_local->معلومات;

        بنية htp_allreduce_kernel_params new_kparams;
        إذا (!ggml_hexagon_precompute_allreduce_params(
            sess, جمع_کړه_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, ریښتیا, is_لیکه_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            أرجع خطأ;
        }

        ثابت منطقي scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ډک;

        لكل (uint32_t r = 0; r < n_ranks; r++) {
            ثابت ggml_tensor * ar_src = وروستی_غوټه.inputs[r];
            إذا (!ggml_hexagon_tensors_overlap(جمع_کړه_dst, ar_src)) {
                استمر;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            إذا (scatter_ok && r == rank) {
                استمر;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            أرجع خطأ;
        }

        إذا (!try_fuse_common(res_tensor, جمع_کړه_dst)) {
            أرجع خطأ;
        }

        وروستی_غوټه.opcode = HTP_OP_ALLREDUCE_جمع_کړه;
        وروستی_غوټه.name   = "ALLREDUCE+ADD";
        وروستی_غوټه.inputs.push_back(res_tensor);
        وروستی_غوټه.outputs.clear();
        وروستی_غوټه.outputs.push_back(جمع_کړه_dst);
        وروستی_غوټه.fused.push_back(غوټه.غوټه);
        حافظه_نقل(وروستی_غوټه.kernel_params, &new_kparams, حجم(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_جمع_کړه;
        حافظه_نقل(o.kernel_params, &new_kparams, حجم(new_kparams));

        o.src[2 * n_ranks] = جمع_کړه_tensor(res_tensor);
        إذا (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            لكل (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            لكل (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } وإلا {
            o.dst[0] = جمع_کړه_tensor(جمع_کړه_dst);
            لكل (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (صحيح) new_kparams.mode, (صحيح) new_kparams.n_dsts);
        أرجع ریښتیا;
    }

    منطقي try_fuse_rms_norm_mul(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0) أرجع خطأ;
        إذا (غوټه.opcode != HTP_OP_MUL) أرجع خطأ;

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];
        إذا (وروستی_غوټه.opcode != HTP_OP_RMS_NORM) أرجع خطأ;

        ثابت ggml_tensor * mul_src0 = غوټه.src0();
        ثابت ggml_tensor * mul_src1 = غوټه.src1();
        ثابت ggml_tensor * rms_بهر  = وروستی_غوټه.dst();

        إذا (!ggml_hexagon_tensor_is_fuseable(rms_بهر)) أرجع خطأ;

        ثابت ggml_tensor * وزن;
        إذا (mul_src0 == rms_بهر || mul_src0->معلومات == rms_بهر->معلومات) {
            وزن = mul_src1;
        } وإلا إذا (mul_src1 == rms_بهر || mul_src1->معلومات == rms_بهر->معلومات) {
            وزن = mul_src0;
        } وإلا {
            أرجع خطأ;
        }

        ثابت ggml_tensor * src0 = وروستی_غوټه.src0();

        إذا (src0->ne[0] != وزن->ne[0] || src0->ne[0] != غوټه.dst()->ne[0]) {
            أرجع خطأ;
        }

        ثابت منطقي is_لیکه_bcast = (وزن->ne[1] == 1 && وزن->ne[2] == 1 && وزن->ne[3] == 1);
        ثابت منطقي is_same_shape = (src0->ne[0] == وزن->ne[0] && src0->ne[1] == وزن->ne[1] &&
                                    src0->ne[2] == وزن->ne[2] && src0->ne[3] == وزن->ne[3]);
        إذا (!is_لیکه_bcast && !is_same_shape) أرجع خطأ;

        إذا (!ggml_are_same_shape(src0, غوټه.dst())) {
            أرجع خطأ;
        }
        إذا (ggml_is_contiguous(src0) != ggml_is_contiguous(غوټه.dst())) {
            أرجع خطأ;
        }

        بنية htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, وزن, غوټه.dst(), &new_kparams
        );

        إذا ((size_t) new_kparams.vtcm_اندازه > sess->vtcm_اندازه) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_اندازه, sess->vtcm_اندازه);
            أرجع خطأ;
        }

        إذا (!try_fuse_common(وزن, غوټه.dst())) {
            أرجع خطأ;
        }

        وروستی_غوټه.opcode = HTP_OP_RMS_NORM_MUL;
        وروستی_غوټه.name   = "RMS_NORM+MUL";
        وروستی_غوټه.inputs.clear();
        وروستی_غوټه.inputs.push_back(src0);
        وروستی_غوټه.inputs.push_back(وزن);
        وروستی_غوټه.outputs.clear();
        وروستی_غوټه.outputs.push_back(غوټه.dst());
        وروستی_غوټه.fused.push_back(غوټه.غوټه);
        حافظه_نقل(وروستی_غوټه.kernel_params, &new_kparams, حجم(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        حافظه_نقل(o.kernel_params, &new_kparams, حجم(new_kparams));

        o.src[0] = جمع_کړه_tensor(src0);
        o.src[1] = جمع_کړه_tensor(وزن);
        لكل (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = جمع_کړه_tensor(غوټه.dst());
        لكل (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        أرجع ریښتیا;
    }

    منطقي try_fuse_mul_mat_جمع_کړه(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0) أرجع خطأ;
        إذا (غوټه.opcode != HTP_OP_جمع_کړه) أرجع خطأ;

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];
        إذا (وروستی_غوټه.opcode != HTP_OP_MUL_MAT) أرجع خطأ;

        ثابت ggml_tensor * جمع_کړه_src0 = غوټه.src0();
        ثابت ggml_tensor * جمع_کړه_src1 = غوټه.src1();
        ثابت ggml_tensor * mm_بهر   = وروستی_غوټه.dst();

        إذا (!ggml_hexagon_tensor_is_fuseable(mm_بهر)) أرجع خطأ;

        ثابت ggml_tensor * src2;
        إذا (جمع_کړه_src0 == mm_بهر || جمع_کړه_src0->معلومات == mm_بهر->معلومات) {
            src2 = جمع_کړه_src1;
        } وإلا إذا (جمع_کړه_src1 == mm_بهر || جمع_کړه_src1->معلومات == mm_بهر->معلومات) {
            src2 = جمع_کړه_src0;
        } وإلا {
            أرجع خطأ;
        }

        ثابت ggml_tensor * src0 = وروستی_غوټه.src0();
        ثابت ggml_tensor * src1 = وروستی_غوټه.src1();

        إذا (src2->type != GGML_TYPE_F32) أرجع خطأ;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        إذا (!ggml_are_same_shape(mm_بهر, غوټه.dst())) أرجع خطأ;

        ثابت بنية htp_mm_kernel_params * orig_kparams = (ثابت بنية htp_mm_kernel_params *) وروستی_غوټه.kernel_params;
        بنية htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_جمع_کړه_params(sess, src0, src1, src2, غوټه.dst(), &kparams);
        إذا (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            أرجع خطأ;
        }

        إذا (orig_kparams->collapse != kparams.collapse) {
            أرجع خطأ;
        }

        ثابت صحيح src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ثابت منطقي can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        إذا (!can_fuse) أرجع خطأ;

        إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
            أرجع خطأ;
        }

        إذا (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            إذا (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_تارونه < orig_kparams->n_act_تارونه) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_تارونه, kparams.n_act_تارونه);
                أرجع خطأ;
            }
        }

        إذا (!try_fuse_common(src2, غوټه.dst())) {
            أرجع خطأ;
        }

        وروستی_غوټه.opcode = HTP_OP_MUL_MAT_جمع_کړه;
        وروستی_غوټه.name   = "MUL_MAT+ADD";
        وروستی_غوټه.inputs.clear();
        وروستی_غوټه.inputs.push_back(src0);
        وروستی_غوټه.inputs.push_back(src1);
        وروستی_غوټه.inputs.push_back(src2);
        وروستی_غوټه.outputs.clear();
        وروستی_غوټه.outputs.push_back(غوټه.dst());
        وروستی_غوټه.fused.push_back(غوټه.غوټه);
        حافظه_نقل(وروستی_غوټه.kernel_params, &kparams, حجم(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_جمع_کړه;
        حافظه_نقل(o.kernel_params, &kparams, حجم(kparams));

        o.src[0] = جمع_کړه_tensor(src0);
        o.src[1] = جمع_کړه_tensor(src1);
        o.src[2] = جمع_کړه_tensor(src2);
        لكل (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = جمع_کړه_tensor(غوټه.dst());
        لكل (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        أرجع ریښتیا;
    }

    منطقي try_fuse_mul_mat_nx(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0 || غوټه.opcode != HTP_OP_MUL_MAT) أرجع خطأ;
        إذا (!is_mergeable_mul_mat(غوټه.غوټه)) أرجع خطأ;

        ثابت ggml_tensor * w_دننه = غوټه.src0();
        ثابت ggml_tensor * x_دننه = غوټه.src1();
        ثابت ggml_tensor * d_دننه = غوټه.dst();

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        إذا (وروستی_غوټه.opcode == HTP_OP_MUL_MAT_NX) {
            ثابت uint32_t curr_n = (uint32_t) وروستی_غوټه.outputs.size();
            إذا (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                أرجع خطأ;
            }

            ثابت ggml_tensor * w0 = وروستی_غوټه.inputs[0];
            ثابت ggml_tensor * x  = وروستی_غوټه.inputs[curr_n];

            إذا (x_دننه != x || w_دننه->type != w0->type || w_دننه->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_دننه) != ggml_hexagon_tensor_is_tiled(w0)) {
                أرجع خطأ;
            }
            إذا (!وروستی_غوټه.fused.تشه() && (mm_is_hmx_eligible(وروستی_غوټه.fused[0]) != mm_is_hmx_eligible(غوټه.غوټه))) {
                أرجع خطأ;
            }

            ثابت بنية htp_mm_kernel_params * orig_kparams = (ثابت بنية htp_mm_kernel_params *) وروستی_غوټه.kernel_params;
            ثابت منطقي collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_دننه, x, d_دننه);
            إذا (orig_kparams->collapse && !collapse) {
                أرجع خطأ;
            }

            بنية htp_mm_kernel_params kparams;
            إذا (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_لیکې(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } وإلا {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            إذا (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                أرجع خطأ;
            }
            إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
                أرجع خطأ;
            }

            إذا (!try_fuse_common(w_دننه, d_دننه)) {
                أرجع خطأ;
            }

            وروستی_غوټه.inputs[curr_n] = w_دننه;
            وروستی_غوټه.inputs.push_back(x);
            وروستی_غوټه.outputs.push_back(d_دننه);
            وروستی_غوټه.fused.push_back(غوټه.غوټه);
            حافظه_نقل(وروستی_غوټه.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            حافظه_نقل(o.kernel_params, &kparams, حجم(kparams));

            لكل (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = جمع_کړه_tensor(وروستی_غوټه.inputs[s]);
            }
            لكل (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            لكل (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = جمع_کړه_tensor(وروستی_غوټه.outputs[d]);
            }
            لكل (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            أرجع ریښتیا;
        }

        // Case 2: last_node is single MUL_MAT
        إذا (وروستی_غوټه.opcode == HTP_OP_MUL_MAT) {
            إذا (!is_mergeable_mul_mat_pair(وروستی_غوټه.غوټه, غوټه.غوټه)) {
                أرجع خطأ;
            }

            ثابت ggml_tensor * w0 = وروستی_غوټه.src0();
            ثابت ggml_tensor * x  = وروستی_غوټه.src1();
            ثابت ggml_tensor * w1 = غوټه.src0();
            ثابت ggml_tensor * dst_0 = وروستی_غوټه.dst();
            ثابت ggml_tensor * dst_1 = غوټه.dst();

            ثابت بنية htp_mm_kernel_params * orig_kparams = (ثابت بنية htp_mm_kernel_params *) وروستی_غوټه.kernel_params;
            ثابت منطقي collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            إذا (orig_kparams->collapse && !collapse) {
                أرجع خطأ;
            }

            بنية htp_mm_kernel_params kparams;
            إذا (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_لیکې(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } وإلا {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            إذا (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                أرجع خطأ;
            }
            إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
                أرجع خطأ;
            }

            إذا (!try_fuse_common(w1, غوټه.dst())) {
                أرجع خطأ;
            }

            وروستی_غوټه.opcode = HTP_OP_MUL_MAT_NX;
            وروستی_غوټه.name   = "MUL_MAT_NX";
            وروستی_غوټه.inputs.clear();
            وروستی_غوټه.inputs.push_back(w0);
            وروستی_غوټه.inputs.push_back(w1);
            وروستی_غوټه.inputs.push_back(x);
            وروستی_غوټه.outputs.clear();
            وروستی_غوټه.outputs.push_back(dst_0);
            وروستی_غوټه.outputs.push_back(dst_1);
            وروستی_غوټه.fused.push_back(غوټه.غوټه);
            حافظه_نقل(وروستی_غوټه.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            حافظه_نقل(o.kernel_params, &kparams, حجم(kparams));

            o.src[0] = جمع_کړه_tensor(w0);
            o.src[1] = جمع_کړه_tensor(w1);
            o.src[2] = جمع_کړه_tensor(x);
            لكل (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = جمع_کړه_tensor(dst_0);
            o.dst[1] = جمع_کړه_tensor(dst_1);
            لكل (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            أرجع ریښتیا;
        }

        أرجع خطأ;
    }

    منطقي try_fuse_mul_mat_id_nx(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0 || غوټه.opcode != HTP_OP_MUL_MAT_ID) أرجع خطأ;
        إذا (!is_mergeable_mul_mat_id(غوټه.غوټه)) أرجع خطأ;

        ثابت ggml_tensor * w_دننه   = غوټه.src0();
        ثابت ggml_tensor * x_دننه   = غوټه.src1();
        ثابت ggml_tensor * ids_دننه = غوټه.غوټه->src[2];
        ثابت ggml_tensor * d_دننه   = غوټه.dst();

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        إذا (وروستی_غوټه.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ثابت uint32_t curr_n = (uint32_t) وروستی_غوټه.outputs.size();
            إذا (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                أرجع خطأ;
            }

            ثابت ggml_tensor * w0  = وروستی_غوټه.inputs[0];
            ثابت ggml_tensor * x   = وروستی_غوټه.inputs[curr_n];
            ثابت ggml_tensor * ids = وروستی_غوټه.inputs[curr_n + 1];

            إذا (x_دننه != x || ids_دننه != ids || w_دننه->type != w0->type || w_دننه->ne[0] != w0->ne[0] || w_دننه->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_دننه) != ggml_hexagon_tensor_is_tiled(w0)) {
                أرجع خطأ;
            }
            إذا (!وروستی_غوټه.fused.تشه() && (mm_is_hmx_eligible(وروستی_غوټه.fused[0]) != mm_is_hmx_eligible(غوټه.غوټه))) {
                أرجع خطأ;
            }

            بنية htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_دننه, curr_n + 1, &kparams);
            إذا (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                أرجع خطأ;
            }
            إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
                أرجع خطأ;
            }

            إذا (!try_fuse_common(w_دننه, d_دننه)) {
                أرجع خطأ;
            }

            وروستی_غوټه.inputs[curr_n] = w_دننه;
            وروستی_غوټه.inputs[curr_n + 1] = x;
            وروستی_غوټه.inputs.push_back(ids);
            وروستی_غوټه.outputs.push_back(d_دننه);
            وروستی_غوټه.fused.push_back(غوټه.غوټه);
            حافظه_نقل(وروستی_غوټه.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            حافظه_نقل(o.kernel_params, &kparams, حجم(kparams));

            لكل (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = جمع_کړه_tensor(وروستی_غوټه.inputs[s]);
            }
            لكل (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            لكل (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = جمع_کړه_tensor(وروستی_غوټه.outputs[d]);
            }
            لكل (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            أرجع ریښتیا;
        }

        // Case 2: last_node is single MUL_MAT_ID
        إذا (وروستی_غوټه.opcode == HTP_OP_MUL_MAT_ID) {
            إذا (!is_mergeable_mul_mat_id_pair(وروستی_غوټه.غوټه, غوټه.غوټه)) {
                أرجع خطأ;
            }

            ثابت ggml_tensor * w0  = وروستی_غوټه.src0();
            ثابت ggml_tensor * x   = وروستی_غوټه.src1();
            ثابت ggml_tensor * ids = وروستی_غوټه.غوټه->src[2];
            ثابت ggml_tensor * w1  = غوټه.src0();

            بنية htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, غوټه.dst(), 2, &kparams);
            إذا (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                أرجع خطأ;
            }
            إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
                أرجع خطأ;
            }

            إذا (!try_fuse_common(w1, غوټه.dst())) {
                أرجع خطأ;
            }

            ثابت ggml_tensor * dst_0 = وروستی_غوټه.dst();
            ثابت ggml_tensor * dst_1 = غوټه.dst();

            وروستی_غوټه.opcode = HTP_OP_MUL_MAT_ID_NX;
            وروستی_غوټه.name   = "MUL_MAT_ID_NX";
            وروستی_غوټه.inputs.clear();
            وروستی_غوټه.inputs.push_back(w0);
            وروستی_غوټه.inputs.push_back(w1);
            وروستی_غوټه.inputs.push_back(x);
            وروستی_غوټه.inputs.push_back(ids);
            وروستی_غوټه.outputs.clear();
            وروستی_غوټه.outputs.push_back(dst_0);
            وروستی_غوټه.outputs.push_back(dst_1);
            وروستی_غوټه.fused.push_back(غوټه.غوټه);
            حافظه_نقل(وروستی_غوټه.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            حافظه_نقل(o.kernel_params, &kparams, حجم(kparams));

            o.src[0] = جمع_کړه_tensor(w0);
            o.src[1] = جمع_کړه_tensor(w1);
            o.src[2] = جمع_کړه_tensor(x);
            o.src[3] = جمع_کړه_tensor(ids);
            لكل (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = جمع_کړه_tensor(dst_0);
            o.dst[1] = جمع_کړه_tensor(dst_1);
            لكل (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            أرجع ریښتیا;
        }

        أرجع خطأ;
    }

    منطقي try_fuse_gdn_cpy(ثابت htp_opnode & غوټه) {
        إذا (n_ops == 0 || غوټه.opcode != HTP_OP_CPY) أرجع خطأ;

        htp_opnode & وروستی_غوټه = ops[n_ops - 1];
        إذا (وروستی_غوټه.opcode != HTP_OP_GATED_DELTA_NET) أرجع خطأ;
        إذا (وروستی_غوټه.outputs.size() != 1) أرجع خطأ;

        ثابت ggml_tensor * gdn_بهر  = وروستی_غوټه.dst();
        ثابت ggml_tensor * cpy_غوټه = غوټه.غوټه;
        ثابت ggml_tensor * cpy_src  = غوټه.src0();
        ثابت ggml_tensor * cpy_dst  = غوټه.dst();

        إذا (!cpy_src || !cpy_dst || !cpy_dst->معلومات) أرجع خطأ;
        إذا (gdn_بهر->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) أرجع خطأ;
        إذا ((gdn_بهر->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_غوټه->flags & GGML_TENSOR_FLAG_OUTPUT)) أرجع خطأ;

        ثابت ggml_tensor * v = وروستی_غوټه.غوټه->src[2];
        إذا (!v) أرجع خطأ;

        ثابت int64_t S_v      = v->ne[0];
        ثابت int64_t H        = v->ne[1];
        ثابت int64_t n_نښې = v->ne[2];
        ثابت int64_t n_seqs   = v->ne[3];
        ثابت int64_t K        = ggml_get_op_params_i32(وروستی_غوټه.غوټه, 0);
        ثابت size_t  لکۍ_off = (size_t) S_v * H * n_نښې * n_seqs * حجم(عائم);

        ثابت int64_t D         = S_v * S_v * H;
        ثابت int64_t n_written = std::min<int64_t>(n_نښې, K);

        إذا (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_بهر && cpy_src->view_src->معلومات != gdn_بهر->معلومات) ||
            cpy_src->view_offs != لکۍ_off || !ggml_is_contiguous(cpy_src)) {
            أرجع خطأ;
        }

        إذا (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != حجم(عائم)) {
            أرجع خطأ;
        }
        إذا (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * حجم(عائم)) {
            أرجع خطأ;
        }
        إذا (n_written > 1) {
            إذا (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * حجم(عائم)) {
                أرجع خطأ;
            }
        }

        إذا (!try_fuse_common({cpy_dst})) {
            أرجع خطأ;
        }

        وروستی_غوټه.name += "+CPY";
        وروستی_غوټه.outputs.push_back(cpy_dst);
        وروستی_غوټه.fused.push_back(غوټه.غوټه);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = جمع_کړه_tensor(cpy_dst);
        لكل (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        أرجع ریښتیا;
    }

    منطقي try_fuse(ثابت htp_opnode & غوټه) {
        إذا (!opt_opfusion) أرجع خطأ;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_جمع_کړه) && try_fuse_allreduce_جمع_کړه(غوټه)) أرجع ریښتیا;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(غوټه))  أرجع ریښتیا;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_جمع_کړه)   && try_fuse_mul_mat_جمع_کړه(غوټه))   أرجع ریښتیا;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(غوټه))    أرجع ریښتیا;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(غوټه)) أرجع ریښتیا;
        إذا (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(غوټه))       أرجع ریښتیا;
        أرجع خطأ;
    }
};

بنية ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

بنية ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_بفر *shm_buf;
    size_t                      shm_blk_اندازه;
    size_t                      ژوروالی;

    کارول opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       پیل_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_اندازه, size_t ژوروالی) : ژوروالی(ژوروالی) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_اندازه;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_اندازه = 0;
        إذا (opt_profile == 3) {
            tr_اندازه = (HTP_MAX_NTHREADS + 1) * opt_optrace * حجم(htp_trace_desc);
        }

        shm_blk_اندازه = حجم(htp_buf_desc)  * n_bufs    +
                       حجم(htp_tensor)    * n_tensors +
                       حجم(htp_op_desc)   * n_ops     +
                       حجم(htp_prof_desc) * n_ops     +
                       tr_اندازه;

        shm_buf = جديد ggml_hexagon_shared_بفر(sess, shm_blk_اندازه * ژوروالی, ریښتیا /* pinned */);

        op_cache.resize(ژوروالی);
        پیل_usec.resize(ژوروالی, 0);

        إذا (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_اندازه, ژوروالی, shm_buf->size(), shm_blk_اندازه);
        }
    }

    ~ggml_hexagon_opqueue() {
        ړنګول shm_buf;
    }

    size_t shm_اندازه() ثابت { أرجع shm_buf ? shm_buf->size() : 0; }

    // push new batch
    منطقي ټېل(htp_opbatch_req& req, dspqueue_بفر& dbuf, ثابت ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(حجم(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(حجم(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(حجم(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(حجم(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(حجم(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(حجم(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        إذا (seq - shm_buf->sess->batch_rsp_seq > ژوروالی) { أرجع خطأ; }

        ثابت uint32_t خونه = (uint32_t) ((seq - 1) % ژوروالی);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[خونه]   = op_batch->ops;
        پیل_usec[خونه] = ggml_وخت_us();

        ثابت size_t b_اندازه = حجم(htp_buf_desc)  * req.n_bufs;
        ثابت size_t t_اندازه = حجم(htp_tensor)    * req.n_tensors;
        ثابت size_t o_اندازه = حجم(htp_op_desc)   * req.n_ops;
        ثابت size_t p_اندازه = حجم(htp_prof_desc) * req.n_ops;

        size_t tr_اندازه = 0;
        إذا (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_اندازه = (HTP_MAX_NTHREADS + 1) * req.n_traces * حجم(htp_trace_desc);
        } وإلا {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->بنسټ() + ((size_t) خونه * shm_blk_اندازه);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_بفر_FLAG_FLUSH_SENDER | DSPQUEUE_بفر_FLAG_INVALIDATE_RECIPIENT;
        dbuf.دننه_واټن   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->بنسټ();
        dbuf.size     = b_اندازه + t_اندازه + o_اندازه + p_اندازه + tr_اندازه;

        GGML_ASSERT(dbuf.size <= shm_blk_اندازه);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_اندازه;
        uint8_t * t_ptr = m_ptr; m_ptr += t_اندازه;
        uint8_t * o_ptr = m_ptr;

        حافظه_نقل(b_ptr, (فراغ *) op_batch->h_bufs.معلومات(), b_اندازه);
        حافظه_نقل(t_ptr, (فراغ *) op_batch->h_tens.معلومات(), t_اندازه);
        حافظه_نقل(o_ptr, (فراغ *) op_batch->h_ops.معلومات(),  o_اندازه);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (بینښه اوږد اوږد) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_اندازه, t_اندازه, o_اندازه, (size_t) dbuf.size);

        إذا (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            لكل (بینښه صحيح i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (فراغ *) b[i].بنسټ, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            لكل (بینښه صحيح i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (بینښه اوږد اوږد) t[i].معلومات, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        أرجع ریښتیا;
    }

    فراغ راوباسه(htp_opbatch_rsp rsp, dspqueue_بفر dbuf) {
        ثابت uint32_t خونه = (uint32_t) ((rsp.seq - 1) % ژوروالی);

        ثابت size_t b_اندازه = حجم(htp_buf_desc)  * rsp.n_bufs;
        ثابت size_t t_اندازه = حجم(htp_tensor)    * rsp.n_tensors;
        ثابت size_t o_اندازه = حجم(htp_op_desc)   * rsp.n_ops;
        ثابت size_t p_اندازه = حجم(htp_prof_desc) * rsp.n_ops;

        size_t tr_اندازه = 0;
        uint32_t n_traces = 0;
        إذا (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_اندازه = (HTP_MAX_NTHREADS + 1) * n_traces * حجم(htp_trace_desc);
        }

        ثابت size_t m_اندازه = b_اندازه + t_اندازه + o_اندازه + p_اندازه + tr_اندازه;
        GGML_ASSERT(m_اندازه <= shm_blk_اندازه);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (بینښه اوږد اوږد) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_اندازه, t_اندازه, o_اندازه);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_اندازه + t_اندازه + o_اندازه);

        إذا (rsp.n_ops > 0) {
            خپلکاری & ops = op_cache[خونه];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ثابت htp_prof_desc * pd = (ثابت htp_prof_desc *) p_ptr;
            ثابت htp_trace_desc * trace_events = تش_اشاره;
            إذا (opt_profile == 3) {
                trace_events = (ثابت htp_trace_desc *) (p_ptr + p_اندازه);
            }

            إذا (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            لكل (uint32_t i = 0; i < rsp.n_ops; i++) {
                إذا (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            إذا (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

فراغ ggml_hexagon_session::flush_peers() {
    خپلکاری vpeers = std::move(virt_peers);
    virt_peers.clear();
    لكل (خپلکاری * peer : vpeers) {
        peer->flush_sync();
    }

    خپلکاری ppeers = std::move(phys_peers);
    phys_peers.clear();
    لكل (خپلکاری * peer : ppeers) {
        peer->flush_async();
    }

    لكل (خپلکاری & sub : هذا->mdev.sessions) {
        sub->flush_peers();
    }
}

فراغ ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

فراغ ggml_hexagon_session::flush_pending(منطقي all) {
    لكل (خپلکاری & sub : هذا->mdev.sessions) {
        sub->flush_pending(all);
        إذا (sub->وروستی_error > HTP_حالت_OK) {
            هذا->وروستی_error = sub->وروستی_error.load();
        }
    }

    طالما (هذا->batch_rsp_seq < هذا->batch_req_seq) {
        بنية htp_opbatch_rsp rsp;
        uint32_t               rsp_اندازه;
        uint32_t               flags;

        بنية dspqueue_بفر dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ثابت uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        صحيح err = dspqueue_read(هذا->queue, &flags, 1, &n_dbufs, &dbuf, حجم(rsp), &rsp_اندازه, (uint8_t *) &rsp, timeo);
        إذا (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            استمر;
        }

        إذا (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (بینښه) err);
        }

        // Basic sanity checks
        إذا (rsp_اندازه != حجم(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", هذا->c_name(), rsp_اندازه, n_dbufs);
        }

        إذا (rsp.حالت > HTP_حالت_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", هذا->c_name(), حالت_منزل_str(rsp.حالت));
            هذا->وروستی_error = rsp.حالت;
            لكل (خپلکاری & sub : هذا->mdev.sessions) {
                sub->وروستی_error = rsp.حالت;
            }
        }

        op_کتار->راوباسه(rsp, dbuf);

        GGML_ASSERT(rsp.seq == هذا->batch_rsp_seq + 1);
        هذا->batch_rsp_seq = rsp.seq;

        إذا (!all) اكسر;
    }
}

فراغ ggml_hexagon_session::flush_sync(منطقي all) {
    flush_async();
    flush_pending(all);
}

فراغ ggml_hexagon_session::پیل_batch() {
    إذا (هذا->mdev.count > 1) {
        کتار_ته_ورزیاتول_mdev_ډله();
    }
}

فراغ ggml_hexagon_session::flush_batch(size_t min_ops) {
    إذا (op_batch->n_ops < min_ops) { أرجع; }

    op_batch->ترتیبول_buffers();

    htp_opbatch_req req {};
    dspqueue_بفر dbuf{};

    ثابت uint64_t seq = ++هذا->batch_req_seq;

    op_batch->update_mdev_ډله(هذا->mdev.idx);

    إذا (!op_کتار->ټېل(req, dbuf, op_batch, seq)) {
        flush_pending(خطأ);
        op_کتار->ټېل(req, dbuf, op_batch, seq);
    }

    لكل (خپلکاری & sub : هذا->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_بفر sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_ډله(sub->mdev.idx);

        إذا (!sub->op_کتار->ټېل(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(خطأ);
            sub->op_کتار->ټېل(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        صحيح err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, حجم(sub_req), (ثابت uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        إذا (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (بینښه) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", هذا->c_name(), dbuf.ptr, dbuf.size);

    صحيح err = dspqueue_write(هذا->queue, 0, 1, &dbuf, حجم(req), (ثابت uint8_t*) &req, DSPQUEUE_TIMEOUT);
    إذا (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", هذا->c_name(), (بینښه) err);
    }

    op_batch->reset();
}

فراغ ggml_hexagon_session::کتار_ته_ورزیاتول_op(ثابت htp_opnode & غوټه) {
    خپلکاری clone_tensor_بفر = [هذا](ثابت ggml_tensor * t) {
        خپلکاری sbuf = هذا->mmap_tensor(t);
        إذا (!sbuf) أرجع;
        إذا (sbuf->sess != هذا) {
            هذا->clone_بفر(sbuf);
        }
        لكل (خپلکاری & sub : هذا->mdev.sessions) {
            sub->clone_بفر(sbuf);
        }
    };

    لكل (خپلکاری t : غوټه.get_inputs()) {
        clone_tensor_بفر(t);
    }
    لكل (خپلکاری t : غوټه.get_outputs()) {
        clone_tensor_بفر(t);
    }

    إذا (opt_opfusion && op_batch->try_fuse(غوټه)) {
        أرجع;
    }

    إذا (!op_batch->fit_op(غوټه)) {
        flush_async();
    }

    إذا (op_batch->تشه()) {
        پیل_batch();
    }

    إذا (!op_batch->fit_op(غوټه)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->جمع_کړه_op(غوټه);
}

فراغ ggml_hexagon_session::کتار_ته_ورزیاتول_mdev_ډله() {
    htp_opnode ډله_غوټه(HTP_OP_MDEV_ډله);

    uint8_t * fence_خونه = هذا->mdev_fence_خونه;

    ساكن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.بفر       = &هذا->fence_buf->backend_بفر;
    dummy_t.extra        = &fence_extra;
    dummy_t.معلومات         = (فراغ *) fence_خونه;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_خونه_اندازه;
    dummy_t.ne[1]        = (int64_t) هذا->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_خونه_اندازه;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) هذا->mdev.idx;

    ggml_tensor * غوټه = ډله_غوټه.جمع_کړه_dummy(dummy_t);
    غوټه->src[0] = غوټه;
    ډله_غوټه.init(غوټه);
    ډله_غوټه.outputs.clear();
    ډله_غوټه.name = "MDEV_GROUP";

    لكل (خپلکاری & sub : هذا->mdev.sessions) {
        sub->clone_بفر(هذا->fence_buf);
    }

    op_batch->جمع_کړه_op(ډله_غوټه);
}

فراغ ggml_hexagon_session::کتار_ته_ورزیاتول_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ثابت منطقي with_fence = sync_tensor != تش_اشاره;
    htp_opnode cpy_غوټه(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* غوټه = cpy_غوټه.جمع_کړه_dummy(*dst);
    غوټه->op     = GGML_OP_CPY;
    غوټه->src[0] = const_cast<ggml_tensor *>(src);
    غوټه->src[1] = with_fence ? cpy_غوټه.جمع_کړه_dummy(*sync_tensor) : تش_اشاره;
    إذا (with_fence) {
        غوټه->op_params[0] = (int32_t) fence_seq;
    }

    cpy_غوټه.init(غوټه);
    إذا (with_fence) {
        cpy_غوټه.name = "CPY+FENCE";
    }
    ثابت منطقي ok = ggml_hexagon_precompute_cpy_params(هذا, غوټه, (بنية htp_copy_kernel_params *) cpy_غوټه.kernel_params);
    ثابت خپلکاری * kparams = (ثابت بنية htp_copy_kernel_params *) cpy_غوټه.kernel_params;
    إذا (ok && !with_fence && kparams->ټول_elems == 0) {
        أرجع;
    }
    هذا->کتار_ته_ورزیاتول_op(cpy_غوټه);
}

فراغ ggml_hexagon_session::کتار_ته_ورزیاتول_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq, منطقي wait) {
    htp_opnode sync_غوټه(HTP_OP_FENCE);

    ggml_tensor* غوټه = sync_غوټه.جمع_کړه_dummy(*sync_tensor);
    غوټه->op           = GGML_OP_NONE;
    غوټه->src[0]       = غوټه;
    غوټه->op_params[0] = (int32_t) fence_seq;
    غوټه->op_params[1] = wait ? 0 : 1;

    sync_غوټه.init(غوټه);
    sync_غوټه.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    هذا->کتار_ته_ورزیاتول_op(sync_غوټه);
}

ساكن منطقي ggml_hexagon_precompute_allreduce_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقي has_جمع_کړه,
    منطقي is_لیکه_bcast,
    منطقي is_shard_ok,
    بنية htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_لیکه_bcast = (has_جمع_کړه && is_لیکه_bcast) ? 1 : 0;

    ثابت uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ثابت uint32_t elem_اندازه = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(عائم);
    ثابت منطقي is_contiguous = ggml_is_contiguous(dst);

    ثابت uint32_t ne0 = (uint32_t) dst->ne[0];
    ثابت uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ثابت منطقي use_1d = is_contiguous && !(has_جمع_کړه && is_لیکه_bcast && ne1 > 1);

    إذا (has_جمع_کړه) {
        // sharded reduce-scatter for contiguous in-place add
        إذا (opt_ar_scatter && is_shard_ok && use_1d && !is_لیکه_bcast) {
            ثابت uint32_t rank_chunk_elems = hex_پړاو_پورته((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_پیل  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_پای_ټکی    = (std::min)(rank_elem_پیل + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_پیل = (int32_t) rank_elem_پیل;
            kparams->rank_nelem      = (int32_t) (rank_elem_پای_ټکی - rank_elem_پیل);
        } وإلا إذا (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_پیل = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } وإلا {
            kparams->n_dsts          = 1;
            kparams->rank_elem_پیل = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } وإلا {
        kparams->n_dsts = (int32_t) n_ranks;
        إذا (use_1d) {
            ثابت uint32_t rank_chunk_elems = hex_پړاو_پورته((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_پیل  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_پای_ټکی    = (std::min)(rank_elem_پیل + rank_chunk_elems, nelem);
            ثابت uint32_t rank_nelem       = rank_elem_پای_ټکی - rank_elem_پیل;
            kparams->rank_elem_پیل        = (int32_t) rank_elem_پیل;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } وإلا {
            ثابت uint32_t rank_chunk_لیکې = (ne1 + n_ranks - 1) / n_ranks;
            ثابت uint32_t rank_r0         = (std::min)(rank * rank_chunk_لیکې, ne1);
            ثابت uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_لیکې, ne1);
            ثابت uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_پیل       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ډک;

    إذا (use_1d) {
        ثابت uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, (std::max)(1u, rank_nelem / 128));
        kparams->n_تارونه = n_تارونه;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_شمېر(n_ranks, n_تارونه, has_جمع_کړه, is_لیکه_bcast);

        uint32_t بلاک_elems = 65536;
        إذا (بلاک_elems > rank_nelem / n_تارونه && rank_nelem / n_تارونه > 128) {
            بلاک_elems = hex_پړاو_پورته(rank_nelem / (n_تارونه * 2), 128);
        }
        بلاک_elems = (std::max)(128u, بلاک_elems);

        kparams->بلاک_elems          = بلاک_elems;
        kparams->vtcm_اندازه_فی_thread = 2 * بلاک_elems * elem_اندازه;
        kparams->vtcm_اندازه            = n_vtcm_buffers * kparams->vtcm_اندازه_فی_thread;

        طالما ((size_t) kparams->vtcm_اندازه > sess->vtcm_اندازه && بلاک_elems > 128) {
            ثابت size_t max_بایټونه_فی_buf = sess->vtcm_اندازه / (n_vtcm_buffers * 2);
            بلاک_elems = (uint32_t) hex_align_ښکته((size_t) (max_بایټونه_فی_buf / elem_اندازه), 128);
            إذا (بلاک_elems < 128) اكسر;
            kparams->بلاک_elems          = بلاک_elems;
            kparams->vtcm_اندازه_فی_thread = 2 * بلاک_elems * elem_اندازه;
            kparams->vtcm_اندازه            = n_vtcm_buffers * kparams->vtcm_اندازه_فی_thread;
        }

        إذا (sess->vtcm_اندازه < (size_t) kparams->vtcm_اندازه || بلاک_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_اندازه, sess->vtcm_اندازه);
            أرجع خطأ;
        }

        kparams->elems_فی_thread = hex_پړاو_پورته((rank_nelem + n_تارونه - 1) / n_تارونه, بلاک_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        أرجع ریښتیا;
    } وإلا {
        ثابت uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, (std::max)(1u, rank_nrows));
        kparams->n_تارونه = n_تارونه;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_شمېر(n_ranks, n_تارونه, has_جمع_کړه, is_لیکه_bcast);

        ثابت uint32_t لیکه_بایټونه = ne0 * elem_اندازه;
        ثابت uint32_t لیکه_اندازه_aligned = (uint32_t) hex_align_پورته(لیکه_بایټونه, 128);
        kparams->لیکه_اندازه_aligned = لیکه_اندازه_aligned;

        ثابت uint32_t nrows_فی_thread = (rank_nrows + n_تارونه - 1) / n_تارونه;
        uint32_t بلاک_لیکې = (std::min)(128u, nrows_فی_thread);
        بلاک_لیکې = (std::max)(1u, بلاک_لیکې);
        kparams->بلاک_elems = بلاک_لیکې;

        kparams->vtcm_اندازه_فی_thread = 2 * (بلاک_لیکې * لیکه_اندازه_aligned);
        kparams->vtcm_اندازه            = n_vtcm_buffers * kparams->vtcm_اندازه_فی_thread;

        طالما ((size_t) kparams->vtcm_اندازه > sess->vtcm_اندازه && بلاک_لیکې > 1) {
            ثابت size_t max_لیکې_فی_buf = sess->vtcm_اندازه / (n_vtcm_buffers * 2 * لیکه_اندازه_aligned);
            بلاک_لیکې = (std::max)(1u, (uint32_t) max_لیکې_فی_buf);
            kparams->بلاک_elems          = بلاک_لیکې;
            kparams->vtcm_اندازه_فی_thread = 2 * (بلاک_لیکې * لیکه_اندازه_aligned);
            kparams->vtcm_اندازه            = n_vtcm_buffers * kparams->vtcm_اندازه_فی_thread;
            إذا (max_لیکې_فی_buf == 0) اكسر;
        }

        إذا (sess->vtcm_اندازه < (size_t) kparams->vtcm_اندازه || بلاک_لیکې < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_اندازه, sess->vtcm_اندازه);
            أرجع خطأ;
        }

        kparams->elems_فی_thread = nrows_فی_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        أرجع ریښتیا;
    }
}

فراغ ggml_hexagon_session::کتار_ته_ورزیاتول_allreduce(
    ثابت ggml_tensor * dst,
    ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
    ثابت std::vector<ثابت ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_ثبت,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_غوټه(HTP_OP_ALLREDUCE);

    ggml_tensor* غوټه = ar_غوټه.جمع_کړه_dummy(*dst);
    غوټه->op           = GGML_OP_NONE;
    غوټه->op_params[0] = (int32_t) fence_seq_ثبت;
    غوټه->op_params[1] = (int32_t) fence_seq_exit;

    ar_غوټه.init(غوټه);

    ar_غوټه.inputs.clear();
    لكل (size_t i = 0; i < src_tensors.size(); i++) {
        ar_غوټه.inputs.push_back(src_tensors[i]);
    }
    لكل (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_غوټه.inputs.push_back(ar_غوټه.جمع_کړه_dummy(*sync_tensors[i]));
    }

    ar_غوټه.outputs.clear();
    لكل (size_t i = 0; i < src_tensors.size(); i++) {
        ar_غوټه.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        هذا, dst, rank, n_ranks, خطأ, خطأ, /*is_shard_ok=*/ خطأ,
        (بنية htp_allreduce_kernel_params *) ar_غوټه.kernel_params);

    ar_غوټه.name = "ALLREDUCE";
    هذا->کتار_ته_ورزیاتول_op(ar_غوټه);
}

ggml_hexagon_shared_بفر * ggml_hexagon_session::mmap_tensor(ثابت ggml_tensor * t) {
    إذا (!t) أرجع تش_اشاره;

    خپلکاری sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
    إذا (!sbuf->mapped) {
        ثابت منطقي is_وزن = ggml_backend_بفر_get_usage(t->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        ثابت منطقي extended  = opt_dma64 && is_وزن;
        sbuf->mmap(extended);
    }
    أرجع sbuf;
}

منطقي ggml_hexagon_session::clone_بفر(ثابت ggml_hexagon_shared_بفر * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    إذا (sbuf->sess == هذا) أرجع ریښتیا;

    خپلکاری mem = sbuf->mem;
    صحيح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    إذا (هذا->cloned_buffers.find(fd) != هذا->cloned_buffers.end()) أرجع ریښتیا;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", هذا->name.c_str(),
                sbuf->c_name(), sbuf->بنسټ(), sbuf->size(), fd);

    خپلکاری clone = std::make_unique<ggml_hexagon_shared_بفر>(هذا, *sbuf);
    حاول {
        clone->mmap(sbuf->extended);
    } امسك (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", هذا->c_name(), exc.what());
        أرجع خطأ;
    }

    هذا->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(هذا);
    أرجع ریښتیا;
}

فراغ ggml_hexagon_session::release_بفر(ثابت ggml_hexagon_shared_بفر * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    خپلکاری mem = sbuf->mem;
    صحيح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    خپلکاری it = هذا->cloned_buffers.find(fd);
    إذا (it != هذا->cloned_buffers.end()) {
        خپلکاری clone = std::move(it->second);
        هذا->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(هذا);
}

فراغ ggml_hexagon_session::unclone_بفر(ثابت ggml_hexagon_shared_بفر * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    خپلکاری mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    لكل (خپلکاری * sess : sessions) {
        sess->release_بفر(sbuf);
    }
}

ساكن size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_بفر *> sbufs;

    ثابت size_t MiB = 1024 * 1024;
    ثابت size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t ګام = 256u * MiB;

    حاول {
        sbufs.push_back(جديد ggml_hexagon_shared_بفر(sess, GiB, ریښتیا)); vmem += GiB;
        sbufs.push_back(جديد ggml_hexagon_shared_بفر(sess, GiB, ریښتیا)); vmem += GiB;
        sbufs.push_back(جديد ggml_hexagon_shared_بفر(sess, GiB, ریښتیا)); vmem += GiB;

        طالما (1) {
            sbufs.push_back(جديد ggml_hexagon_shared_بفر(sess, ګام, ریښتیا));
            vmem += ګام;
        }
    } امسك (...) { }

    لكل (خپلکاری b : sbufs) { ړنګول b; }

    أرجع vmem - ګام; // backoff to account for overhead from internal mappings
}

فراغ ggml_hexagon_session::allocate(ثابت ggml_hexagon_device_config & config) noexcept(خطأ) {
    صحيح phys_idx = config.physical_idx;
    صحيح virt_idx = config.virtual_idx;

    هذا->valid_session = خطأ;
    هذا->valid_handle  = خطأ;
    هذا->valid_کتار   = خطأ;
    هذا->valid_iface   = خطأ;

    هذا->name          = config.name;
    هذا->phys_idx      = phys_idx;
    هذا->virt_idx      = virt_idx;
    هذا->domain_id     = config.domain_id;
    هذا->session_id    = 0;
    هذا->batch_req_seq = 0;
    هذا->batch_rsp_seq = 0;
    هذا->وروستی_error    = HTP_حالت_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", هذا->name.c_str(), هذا->domain_id, phys_idx, virt_idx);

    إذا (config.domain_id < 0 || config.domain_name.تشه()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ارم std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ثابت std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    إذا (virt_idx > 0) {
        بنية remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<حرف *>(dom_name.c_str());
        n.session_name     = const_cast<حرف *>(هذا->name.c_str());
        n.session_name_len = هذا->name.size();
        n.session_id       = virt_idx;

        صحيح err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (فراغ *) &n, حجم(n));
        إذا (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           هذا->c_name(), phys_idx, virt_idx, err);
            ارم std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        هذا->session_id    = n.session_id;
        هذا->domain_id     = n.effective_domain_id;
        هذا->valid_session = ریښتیا;
    } وإلا {
        بنية remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<حرف *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        صحيح err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (فراغ *) &eff, حجم(eff));
        إذا (err == AEE_SUCCESS) {
            هذا->domain_id = eff.effective_domain_id;
        } وإلا {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           هذا->name.c_str(), err, هذا->domain_id);
        }
    }

    // Enable unsigned modules
    {
        بنية remote_rpc_control_unsigned_module u;
        u.domain = هذا->domain_id;
        u.enable = 1;
        صحيح err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (فراغ *) &u, حجم(u));
        إذا (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", هذا->c_name(), err);
            ارم std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    حرف session_uri[256];
    {
        حرف htp_uri[256];
        snprintf(htp_uri, حجم(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        بنية remote_rpc_get_uri u = {};
        u.session_id      = هذا->session_id;
        u.domain_name     = const_cast<حرف *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<حرف *>(htp_uri);
        u.module_uri_len  = مزي_اوږدوالی(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = حجم(session_uri);

        صحيح err = remote_session_control(FASTRPC_GET_URI, (فراغ *) &u, حجم(u));
        إذا (err != AEE_SUCCESS) {
            snprintf(session_uri, حجم(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), هذا->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          هذا->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    صحيح err = htp_iface_open(session_uri, &هذا->handle);
    إذا (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", هذا->c_name(), session_uri, err);
        ارم std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    هذا->valid_handle = ریښتیا;

    // Query HW info and resolve session options
    {
        بینښه صحيح hw_n_تارونه = 0;
        بینښه صحيح hw_n_hvx     = 0;
        بینښه صحيح hw_n_hmx     = 0;
        بینښه اوږد اوږد hw_vtcm_اندازه = 0;
        صحيح hw_err = htp_iface_hwinfo(هذا->handle, &hw_n_تارونه, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_اندازه);
        إذا (hw_err == 0) {
            ثابت uint32_t max_n_تارونه = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_تارونه);
            هذا->n_تارونه = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_تارونه) : max_n_تارونه;
            هذا->n_hvx     = هذا->n_تارونه;
            هذا->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            هذا->vtcm_اندازه = (uint64_t)hw_vtcm_اندازه;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          هذا->c_name(), هذا->n_تارونه, هذا->n_hvx, هذا->n_hmx,
                          (بینښه اوږد اوږد)(هذا->vtcm_اندازه / (1024 * 1024)));
        } وإلا {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", هذا->c_name(), hw_err);
            ثابت uint32_t default_n_تارونه = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            هذا->n_تارونه = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_تارونه;
            هذا->n_hvx     = هذا->n_تارونه;
            هذا->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            هذا->vtcm_اندازه = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        بنية remote_rpc_control_latency l;
        l.enable = 1;

        صحيح err = remote_handle64_control(هذا->handle, DSPRPC_CONTROL_LATENCY, (فراغ *) &l, حجم(l));
        إذا (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (بینښه) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", هذا->c_name(),
                  هذا->session_id, هذا->domain_id, session_uri, (بینښه اوږد) هذا->handle);

    ثابت size_t req_q_اندازه = (حجم(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ثابت size_t rsp_q_اندازه = (حجم(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(هذا->domain_id,
                          0,              // Flags
                          req_q_اندازه,     // Request  queue size (in bytes)
                          rsp_q_اندازه,     // Response queue size (in bytes)
                          تش_اشاره,        // Read packet callback (we handle reads explicitly)
                          تش_اشاره,        // Error callback (we handle errors during reads)
                          (فراغ *) هذا,  // Callback context
                          &queue);
    إذا (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", هذا->name.c_str(), (بینښه) err);
        ارم std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    هذا->valid_کتار = ریښتیا;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &هذا->کتار_id);
    إذا (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (بینښه) err);
        ارم std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    إذا (opt_etm) {
        err = htp_iface_etm(هذا->handle, 1);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (بینښه) err);
        }
    }

    // Allocate buffers and state for op batching
    هذا->op_کتار = جديد ggml_hexagon_opqueue(هذا, opt_opbatch, opt_opqueue);

    هذا->fence_buf = جديد ggml_hexagon_fence_بفر(هذا, &dev_ctx->fence_بفر_type, 64 * 1024);
    إذا (هذا->mdev.count > 1) {
        هذا->mdev_fence_خونه = هذا->alloc_fence(هذا->mdev.count);
    }

    إذا (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(هذا);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", هذا->c_name(), opt_vmem);
    }
    ثابت size_t shm_اندازه = هذا->op_کتار->shm_اندازه();
    هذا->max_vmem = (opt_vmem > shm_اندازه) ? (opt_vmem - shm_اندازه) : opt_vmem;

    هذا->op_batch = جديد ggml_hexagon_opbatch(هذا, opt_opbatch, هذا->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_پیل(هذا->handle, هذا->session_id, هذا->کتار_id, هذا->n_تارونه, opt_nhmx, هذا->max_vmem, (uint32_t) opt_power_ډېرپېښ);
    إذا (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", هذا->c_name(), (بینښه) err);
        ارم std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    هذا->valid_iface = ریښتیا;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", هذا->c_name(), htp_power_ډېرپېښ_منزل_str(opt_power_ډېرپېښ));

    إذا (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(هذا->handle, opt_profile, &pmu_conf);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (بینښه) err);
        }
    }
}

فراغ ggml_hexagon_session::release() noexcept(ریښتیا) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", هذا->name.c_str());

    هذا->mdev.sessions.clear();

    صحيح err;

    إذا (هذا->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(هذا->handle);
        إذا (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (بینښه) err);
        }
    }

    ړنګول هذا->op_batch;
    ړنګول هذا->op_کتار;
    لكل (خپلکاری & it : هذا->cpy_fence_slots) {
        free_fence((فراغ *) it.second, 1);
    }
    هذا->cpy_fence_slots.clear();

    إذا (هذا->fence_buf) {
        unclone_بفر(هذا->fence_buf);
        ړنګول هذا->fence_buf;
        هذا->fence_buf = تش_اشاره;
    }
    طالما (!هذا->cloned_buffers.تشه()) {
        release_بفر(هذا->cloned_buffers.begin()->second.get());
    }

    إذا (opt_etm) {
        err = htp_iface_etm(هذا->handle, 0);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (بینښه) err);
        }
    }

    إذا (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(هذا->handle, 0, &pmu_conf);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (بینښه) err);
        }
    }

    إذا (هذا->valid_کتار) {
        err = dspqueue_close(queue);
        إذا (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (بینښه) err);
        }
    }

    إذا (هذا->valid_handle) {
        htp_iface_close(هذا->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_شمېر) noexcept(خطأ) {
    هذا->dev        = dev;
    هذا->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    هذا->mdev.idx   = mdev_idx;
    هذا->mdev.count = mdev_شمېر > 0 ? mdev_شمېر : (uint32_t) (1 + config.mdev_ډله.size());
    op_batch         = تش_اشاره;
    op_کتار         = تش_اشاره;
    fence_buf        = تش_اشاره;
    fence_seq        = ((uintptr_t)هذا) & 0xFFFF;

    حاول {
        allocate(config);
        إذا (هذا->mdev.idx == 0 && !config.mdev_ډله.تشه()) {
            لكل (size_t i = 0; i < config.mdev_ډله.size(); i++) {
                هذا->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_ډله[i], هذا->dev, (uint32_t) (i + 1), هذا->mdev.count));
            }
        }
    } امسك (ثابت std::exception & exc) {
        release();
        ارم;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(ریښتیا) {
    release();
}

// ** backend interface

ساكن منطقي ggml_hexagon_flash_attn_is_hmx_eligible(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * q,
    ثابت بنية ggml_tensor * k,
    ثابت بنية ggml_tensor * v,
    ثابت بنية ggml_tensor * sinks
) {
    إذا (sess->n_hmx == 0) {
        أرجع خطأ;
    }

    إذا (opt_fa_select < 2) {
        أرجع خطأ;
    }

    إذا ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        أرجع خطأ;
    }

    ثابت uint32_t DK = q->ne[0];
    ثابت uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    إذا (DK % 8 != 0 || DV % 8 != 0) {
        أرجع خطأ;
    }

    GGML_ناکارول_شوی(sinks);

    // Explicit force mode
    إذا (opt_fa_select > 2) {
        أرجع ریښتیا;
    }

    ثابت uint32_t M = q->ne[1];

    // Prefill or batched decode
    إذا (M > 1) {
        أرجع ریښتیا;
    }

    ثابت uint32_t n_سر = q->ne[2];
    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G = n_kv_heads > 0 ? n_سر / n_kv_heads : 1;
    ثابت uint32_t S = k->ne[1];
    ثابت uint32_t ټول_تارونه = sess->n_تارونه * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    إذا (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < ټول_تارونه))) {
        أرجع ریښتیا;
    }

    // Tile alignment for 32-row HMX tiles
    ثابت منطقي is_tile_aligned = (G > 0 && (32 % G == 0));
    إذا (!is_tile_aligned) {
        إذا (sess->n_تارونه >= 6) {
            أرجع خطأ;
        }
        أرجع S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    إذا (DK == 256) {
        s_cross = 4096;
    } وإلا إذا (DK <= 64) {
        s_cross = (sess->n_تارونه >= 8) ? 2048 : ((sess->n_تارونه >= 6) ? 768 : 512);
    } وإلا {
        s_cross = (sess->n_تارونه >= 8) ? 1024 : 512;
    }

    أرجع S >= s_cross;
}

ساكن منطقي ggml_hexagon_precompute_flash_attn_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_fa_kernel_params * kparams
) {
    إذا (opt_fa_select < 1) {
        أرجع خطأ;
    }

    memset(kparams, 0, حجم(*kparams));

    ثابت بنية ggml_tensor * q    = op->src[0];
    ثابت بنية ggml_tensor * k    = op->src[1];
    ثابت بنية ggml_tensor * v    = op->src[2];
    ثابت بنية ggml_tensor * ماسک = op->src[3];

    ثابت uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ثابت uint32_t neq1 = q->ne[1];  // n_tokens
    ثابت uint32_t neq2 = q->ne[2];  // n_heads

    ثابت uint32_t nek1 = k->ne[1];  // kv_len

    ثابت uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ثابت uint32_t DK = neq0;
    ثابت uint32_t DV = nev0;

    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G          = neq2 / n_kv_heads;

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;
    حافظه_نقل(&scale,         &op->op_params[0], حجم(عائم));
    حافظه_نقل(&max_bias,      &op->op_params[1], حجم(عائم));
    حافظه_نقل(&logit_softcap, &op->op_params[2], حجم(عائم));

    إذا (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->سر_وېشل = (opt_fa_سر_وېشل != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ثابت uint32_t n_سر = q->ne[2];
    kparams->n_سر_log2 = 1u << (uint32_t) std::floor(std::log2(n_سر));
    kparams->m0 = std::توان(2.0f, -(max_bias) / kparams->n_سر_log2);
    kparams->m1 = std::توان(2.0f, -(max_bias / 2.0f) / kparams->n_سر_log2);

    // Check HMX eligibility
    ثابت بنية ggml_tensor * sinks = op->src[4];
    إذا (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ثابت uint32_t DK_pad = hex_پړاو_پورته(DK, 64);
        ثابت uint32_t DV_pad = hex_پړاو_پورته(DV, 64);
        size_t Br = 0, Bc = 0;
        صحيح ret = hmx_fa_پیدا_کړه_chunk_اندازه(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_اندازه, sess->n_تارونه, (q->type == GGML_TYPE_F32), sinks != تش_اشاره, n_سر);
        إذا (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            إذا (logit_softcap == 0.0f) {
                kparams->scale = scale * (عائم) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } وإلا {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (عائم) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_بلاکونه = (nek1 + Bc - 1) / Bc;
            kparams->n_تارونه = (kparams->n_kv_بلاکونه >= 3 && sess->n_تارونه >= 2) ? sess->n_تارونه : 1;

            kparams->u.hmx.g_br = hex_align_پورته(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_بلاکونه >= 3 && sess->n_تارونه >= 2) ? 1 : 0;
            kparams->vtcm_اندازه = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_تارونه, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != تش_اشاره, n_سر);

            ثابت size_t لیکه_vec_بایټونه = hex_align_پورته(Bc * حجم(uint16_t), 256);
            kparams->u.hmx.لیکه_buf_stride = لیکه_vec_بایټونه / 128; // HVX vector is 128 bytes

            ثابت size_t m_کرښه_بایټونه = hex_align_پورته(Bc * حجم(uint16_t), 128);
            kparams->u.hmx.ماسک_buf_لیکه_stride = m_کرښه_بایټونه / حجم(uint16_t);
            kparams->u.hmx.ماسک_خپرونه = (ماسک != تش_اشاره && ماسک->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_ارزښتونه(G);
            إذا (ماسک) {
                kparams->src3_div2 = init_fastdiv_ارزښتونه(ماسک->ne[2]);
                kparams->src3_div3 = init_fastdiv_ارزښتونه(ماسک->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_فی_thread = 0;
            أرجع ریښتیا;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_بلاکونه = (k->ne[1] + 64 - 1) / 64;
    kparams->n_تارونه = sess->n_تارونه;

    ثابت size_t اندازه_q_لیکه_padded = hex_پړاو_پورته(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ثابت size_t اندازه_k_لیکه_padded = hex_پړاو_پورته(k->ne[0] * 2, 128);
    ثابت size_t اندازه_v_لیکه_padded = hex_پړاو_پورته(v->ne[0] * 2, 128);

    kparams->vtcm_اندازه = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), ماسک != تش_اشاره, sinks != تش_اشاره, n_سر, sess->n_تارونه);

    kparams->u.hvx.اندازه_q_لیکه_padded = اندازه_q_لیکه_padded;
    kparams->u.hvx.اندازه_k_لیکه_padded = اندازه_k_لیکه_padded;
    kparams->u.hvx.اندازه_v_لیکه_padded = اندازه_v_لیکه_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_ارزښتونه(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_ارزښتونه(q->ne[1]);
    kparams->خپرونه_rk2 = init_fastdiv_ارزښتونه(q->ne[2]/k->ne[2]);
    kparams->خپرونه_rk3 = init_fastdiv_ارزښتونه(q->ne[3]/k->ne[3]);
    kparams->خپرونه_rv2 = init_fastdiv_ارزښتونه(q->ne[2]/v->ne[2]);
    kparams->خپرونه_rv3 = init_fastdiv_ارزښتونه(q->ne[3]/v->ne[3]);
    إذا (ماسک) {
        kparams->src3_div2 = init_fastdiv_ارزښتونه(ماسک->ne[2]);
        kparams->src3_div3 = init_fastdiv_ارزښتونه(ماسک->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_فی_thread = (kparams->qrows + sess->n_تارونه - 1) / sess->n_تارونه;

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_flash_attn_ext(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * src3 = op->src[3];
    ثابت بنية ggml_tensor * src4 = op->src[4];
    ثابت بنية ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    إذا ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        أرجع خطأ;
    }

    إذا (src3 && src3->type != GGML_TYPE_F16) {  //ماسک
        أرجع خطأ;
    }

    إذا (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        أرجع خطأ;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    بنية htp_fa_kernel_params kparams;
    إذا (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        أرجع خطأ;
    }

    إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_اندازه, sess->vtcm_اندازه);
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_gated_delta_net(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    إذا (opt_gdn_select < 1) {
        أرجع خطأ;
    }

    ثابت بنية ggml_tensor * q     = op->src[0];
    ثابت بنية ggml_tensor * k     = op->src[1];
    ثابت بنية ggml_tensor * v     = op->src[2];
    ثابت بنية ggml_tensor * g     = op->src[3];
    ثابت بنية ggml_tensor * beta  = op->src[4];
    ثابت بنية ggml_tensor * state = op->src[5];
    ثابت بنية ggml_tensor * dst   = op;

    إذا (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous_لیکې(q) || !ggml_is_contiguous_لیکې(k) || !ggml_is_contiguous_لیکې(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    ثابت int64_t S_v      = v->ne[0];
    ثابت int64_t H        = v->ne[1];
    ثابت int64_t n_نښې = v->ne[2];
    ثابت int64_t n_seqs   = v->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(op, 0);

    إذا (S_v <= 0 || S_v > 128 || H <= 0 || n_نښې <= 0 || n_seqs <= 0) {
        أرجع خطأ;
    }
    إذا (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_نښې || k->ne[2] != n_نښې || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        أرجع خطأ;
    }
    إذا ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        أرجع خطأ;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    إذا (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        أرجع خطأ;
    }
    إذا (dst->ne[0] != S_v * H || dst->ne[1] != n_نښې * n_seqs + S_v * n_seqs * K) {
        أرجع خطأ;
    }

    ثابت uint32_t ټول_لیکې = (uint32_t) (H * n_seqs);
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, ټول_لیکې);

    ثابت منطقي can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_نښې >= HTP_GDN_MIN_نښې) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    إذا (can_use_hmx) {
        بنية htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        إذا (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_اندازه, ټول_لیکې, sess->vtcm_اندازه, n_تارونه, ریښتیا, &n_heads_batch)) {
            أرجع خطأ;
        }
    } وإلا {
        بنية htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_تارونه);
        إذا (layout.ټول_بایټونه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_matmul_is_hmx_eligible(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    منطقي is_matmul_id,
    منطقي is_batched
) {
    GGML_ناکارول_شوی(dst);

    إذا (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        أرجع خطأ;
    }

    ثابت صحيح ne00  = src0->ne[0];
    ثابت صحيح ne01  = src0->ne[1];
    ثابت صحيح ne11  = src1->ne[1];
    ثابت صحيح ne12  = src1->ne[2];
    ثابت صحيح wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    إذا (ne01 % HTP_TENSOR_TILE_N_لیکې != 0 && is_matmul_id) {
        أرجع خطأ;
    }

    // HMX supports F16, F32, and repack quantized types.
    إذا (!htp_is_hmx_وزن_type((ggml_type) wtype)) {
        أرجع خطأ;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    إذا (ne00 % HTP_TENSOR_TILE_N_ستون_شمېر != 0) {
        أرجع خطأ;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ثابت منطقي is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    إذا (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        أرجع خطأ;
    }
    إذا (!is_matmul_id && is_batched && is_tiled) {
        أرجع خطأ;
    }

    // HMX assumes contiguous row-major layout.
    إذا (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        أرجع خطأ;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ثابت صحيح m = is_matmul_id ? ne12 : ne11;
    إذا (m <= HTP_MM_HMX_MIN_NROWS) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(dst);
}

ساكن منطقي ggml_hexagon_precompute_hmx_mm_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * act,
    ثابت بنية ggml_tensor * dst,
    صحيح wtype,
    صحيح ne00_padded,
    صحيح ne01_padded,
    صحيح ne02,
    صحيح ne11,
    صحيح ne12,
    صحيح ne11_padded,
    منطقي is_matmul_id,
    منطقي is_batched,
    size_t src2_اندازه,
    size_t vtcm_budget,
    بنية htp_mm_kernel_params * kparams
) {
    ثابت منطقي is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحيح aligned_tile_اندازه = is_tiled ? htp_tensor_get_tile_padded_اندازه(wtype) : 0;
    ثابت صحيح n_تارونه = (صحيح)sess->n_تارونه;
    ثابت صحيح ne10 = act->ne[0];

    صحيح m_for_solver = ne11;
    صحيح m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    إذا (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ثابت منطقي dst_لیکه_وېشل = dst ? ggml_hexagon_can_لیکه_برخه_کول(dst) : ریښتیا;
        ثابت منطقي act_لیکه_وېشل = ggml_hexagon_can_لیکه_برخه_کول(act);
        إذا (dst_لیکه_وېشل && act_لیکه_وېشل) {
            m_for_solver = (ne11 + (صحيح) sess->mdev.count - 1) / (صحيح) sess->mdev.count;
            m_for_solver_padded = hex_پړاو_پورته(std::max(m_for_solver, 32), 32);
        }
    }
    ثابت منطقي pipeline = is_matmul_id ? خطأ : htp_mm_hmx_pipeline(m_for_solver);

    ثابت منطقي is_batched_val = is_matmul_id ? خطأ : is_batched;
    ثابت صحيح ډله_اندازه = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_اندازه = 0;
    منطقي use_grouped = خطأ;
    صحيح act_تارونه_selected = 0;

    إذا (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && ډله_اندازه > 1) {
        // Try grouped path first
        إذا (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, ډله_اندازه, n_تارونه, pipeline, src2_اندازه, vtcm_budget, &m_chunk, &n_chunk, &act_تارونه_selected, &vtcm_اندازه)) {
            use_grouped = ریښتیا;
        }
    }

    إذا (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        صحيح m_id_لیکې = 0;
        إذا (dst && is_matmul_id) {
            ثابت صحيح n_experts = ne02 > 0 ? ne02 : 1;
            ثابت size_t ټول_expert_لیکې = (size_t) dst->ne[1] * dst->ne[2];
            صحيح m_فی_expert = (صحيح) ((ټول_expert_لیکې + n_experts - 1) / n_experts);
            إذا (sess->mdev.count > 1 && ggml_hexagon_can_لیکه_برخه_کول(dst)) {
                m_فی_expert = (m_فی_expert + (صحيح) sess->mdev.count - 1) / (صحيح) sess->mdev.count;
            }
            m_id_لیکې = hex_پړاو_پورته(std::max(m_فی_expert, 32), 32);
        }
        ثابت uint32_t لګښت_m = is_matmul_id ? (uint32_t) m_id_لیکې : (uint32_t) m_for_solver;
        إذا (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_لیکې, ne01_padded, m_for_solver_padded, لګښت_m, n_تارونه, pipeline, is_matmul_id, aligned_tile_اندازه, src2_اندازه, vtcm_budget, &m_chunk, &n_chunk, &act_تارونه_selected, &vtcm_اندازه)) {
            أرجع خطأ;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_تارونه = n_تارونه;
    kparams->n_act_تارونه = act_تارونه_selected;
    kparams->tile_اندازه = is_tiled ? htp_tensor_get_tile_اندازه(wtype) : 0;
    kparams->aligned_tile_اندازه = aligned_tile_اندازه;
    kparams->act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
    kparams->vtcm_اندازه = vtcm_اندازه;
    kparams->vtcm_src0_اندازه = 0;
    kparams->div_n_act_تارونه = init_fastdiv_ارزښتونه(act_تارونه_selected);
    kparams->div_ne00_padded   = init_fastdiv_ارزښتونه(ne00_padded);
    kparams->vtcm_act_اندازه     = 0;
    kparams->vtcm_bias_اندازه    = (int32_t) src2_اندازه;
    kparams->vtcm_dst_اندازه     = 0;

    إذا (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } وإلا {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    أرجع ریښتیا;
}

ساكن فراغ ggml_hexagon_precompute_hvx_mm_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    صحيح wtype,
    منطقي is_tiled,
    صحيح ne02,
    صحيح ne03,
    صحيح ne10,
    صحيح ne11,
    صحيح ne12,
    صحيح ne13,
    منطقي is_matmul_id,
    ثابت size_t src2_لیکه_اندازه,
    size_t vtcm_budget,
    بنية htp_mm_kernel_params * kparams
) {
    إذا (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    }

    kparams->n_hmx = 0;
    kparams->n_تارونه = sess->n_تارونه;

    GGML_ناکارول_شوی(ne02);
    GGML_ناکارول_شوی(ne03);

    ثابت صحيح src1_nrows = ne11 * ne12 * ne13;

    إذا (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_اندازه = htp_tensor_get_tile_اندازه(wtype);
        kparams->aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);

        ثابت منطقي k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        إذا (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (صحيح) sess->n_تارونه)
                                  ? HTP_MM_KERNEL_HVX_QUANT_بلاک : HTP_MM_KERNEL_HVX_QUANT_لیکه;
            kparams->act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);

            بنية htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t غوره_n_prefetch = 2;
            لكل (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_تارونه,
                    0, src0->nb[1], kparams->act_لیکه_اندازه, 0, d, ریښتیا, خطأ
                );
                إذا (L.ټول_بایټونه <= vtcm_budget) {
                    غوره_n_prefetch = d;
                    اكسر;
                }
            }
            إذا (L.ټول_بایټونه > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                أرجع;
            }
            kparams->n_prefetch     = غوره_n_prefetch;
            kparams->vtcm_اندازه      = L.ټول_بایټونه;
            kparams->vtcm_src0_اندازه = L.src0_بایټونه;
            kparams->vtcm_act_اندازه  = L.act_بایټونه;
            kparams->vtcm_bias_اندازه = 0;
            kparams->vtcm_dst_اندازه  = L.dst_بایټونه;
            أرجع;
        } وإلا {
            منطقي try_tiled = (k_align && opt_mm_select >= 1);
            إذا (try_tiled) {
                kparams->act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
                إذا (src1_nrows < (صحيح) sess->n_تارونه) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_بلاک;
                } وإلا {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_لیکه;
                }

                بنية htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t غوره_n_prefetch = 2;
                لكل (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_تارونه,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_لیکه_اندازه, d, خطأ, خطأ
                    );
                    إذا (L.ټول_بایټونه <= vtcm_budget) {
                        غوره_n_prefetch = d;
                        اكسر;
                    }
                }

                uint32_t m_chunk = 0;
                إذا (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_تارونه,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_لیکه_اندازه, غوره_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = غوره_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_اندازه      = L.ټول_بایټونه;
                    kparams->vtcm_src0_اندازه = L.src0_بایټونه;
                    kparams->vtcm_act_اندازه  = L.act_بایټونه;
                    kparams->vtcm_bias_اندازه = L.bias_بایټونه;
                    kparams->vtcm_dst_اندازه  = L.dst_بایټونه;
                    أرجع;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            أرجع;
        }
    } وإلا إذا (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    } وإلا إذا (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        بنية htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        إذا (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_تارونه,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_لیکه_اندازه, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_لیکه_اندازه = hex_پړاو_پورته(ne10 * 2, 128);
            kparams->vtcm_اندازه = L.ټول_بایټونه;
            kparams->vtcm_src0_اندازه = L.src0_بایټونه;
            kparams->vtcm_act_اندازه  = L.act_بایټونه;
            kparams->vtcm_bias_اندازه = L.bias_بایټونه;
            kparams->vtcm_dst_اندازه = L.dst_بایټونه;
            kparams->n_prefetch = 16;
            أرجع;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    } وإلا إذا (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        بنية htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        إذا (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_تارونه,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_لیکه_اندازه, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_لیکه_اندازه = hex_پړاو_پورته(ne10 * 4, 128);
            kparams->vtcm_اندازه = L.ټول_بایټونه;
            kparams->vtcm_src0_اندازه = L.src0_بایټونه;
            kparams->vtcm_act_اندازه  = L.act_بایټونه;
            kparams->vtcm_bias_اندازه = L.bias_بایټونه;
            kparams->vtcm_dst_اندازه = L.dst_بایټونه;
            kparams->n_prefetch = 16;
            أرجع;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    } وإلا إذا (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        بنية htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        إذا (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_تارونه,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_لیکه_اندازه, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_لیکه_اندازه = hex_پړاو_پورته(ne10 * 4, 128);
            kparams->vtcm_اندازه = L.ټول_بایټونه;
            kparams->vtcm_src0_اندازه = L.src0_بایټونه;
            kparams->vtcm_act_اندازه  = L.act_بایټونه;
            kparams->vtcm_bias_اندازه = L.bias_بایټونه;
            kparams->vtcm_dst_اندازه = L.dst_بایټونه;
            kparams->n_prefetch = 16;
            أرجع;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    } وإلا {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    }
}

ساكن فراغ ggml_hexagon_precompute_matmul_params_impl(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    ثابت size_t src2_لیکه_اندازه,
    ثابت size_t src2_اندازه,
    بنية htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت صحيح ne00 = src0->ne[0];
    ثابت صحيح ne01 = src0->ne[1];
    ثابت صحيح ne02 = src0->ne[2];
    ثابت صحيح ne03 = src0->ne[3];

    ثابت صحيح ne10 = src1->ne[0];
    ثابت صحيح ne11 = src1->ne[1];
    ثابت صحيح ne12 = src1->ne[2];
    ثابت صحيح ne13 = src1->ne[3];

    ثابت منطقي is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحيح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحيح ne00_padded = is_tiled ? hex_پړاو_پورته(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ثابت صحيح ne11_padded = hex_پړاو_پورته(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ثابت صحيح ne01_tiled  = hex_پړاو_پورته(ne01, 32);

    ثابت منطقي is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقي is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ثابت size_t vtcm_budget = sess->vtcm_اندازه;

    // Check HMX eligibility and try precomputing HMX parameters
    منطقي hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    إذا (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        إذا (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_اندازه, vtcm_budget, kparams)) {
            اذهب finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_لیکه_اندازه, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ارزښتونه(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ارزښتونه(ne11);
    kparams->div_r2       = init_fastdiv_ارزښتونه(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ارزښتونه(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ارزښتونه(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ساكن منطقي ggml_hexagon_لیکې_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride) {
    size_t s = 0, بل = 0;
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        إذا (ne[i] == 1) استمر;
        إذا (s == 0) { s = nb[i]; بل = s * ne[i]; استمر; }
        إذا (nb[i] != بل) أرجع خطأ;
        بل *= ne[i];
    }
    *stride = s ? s : nb[1];
    أرجع ریښتیا;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ساكن منطقي ggml_hexagon_matmul_can_collapse(ثابت بنية ggml_tensor * src0, ثابت بنية ggml_tensor * src1, ثابت بنية ggml_tensor * dst) {
    size_t s1, sd;
    أرجع (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_جمع_کړه) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_اندازه(src1->type) && ggml_hexagon_لیکې_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_اندازه(dst->type) && ggml_hexagon_لیکې_stride(dst->ne, dst->nb, &sd);
}

ساكن منطقي ggml_hexagon_matmul_جمع_کړه_can_collapse(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * src2,
    ثابت بنية ggml_tensor * dst
) {
    إذا (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        أرجع خطأ;
    }
    إذا (!src2) {
        أرجع ریښتیا;
    }
    ثابت int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    إذا (src2_nrows == 1) {
        أرجع ریښتیا;
    }
    size_t s2;
    أرجع src2->nb[0] == ggml_type_اندازه(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_لیکې_stride(src2->ne, src2->nb, &s2);
}

ساكن ggml_tensor ggml_hexagon_tensor_collapse_لیکې(ثابت بنية ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_لیکې_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    أرجع c;
}

ساكن فراغ ggml_hexagon_precompute_matmul_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_mm_kernel_params * kparams
) {
    إذا (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_لیکې(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_لیکې(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        أرجع;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ساكن فراغ ggml_hexagon_precompute_fused_matmul_جمع_کړه_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * src2,
    ثابت بنية ggml_tensor * dst,
    بنية htp_mm_kernel_params * kparams
) {
    إذا (ggml_hexagon_matmul_جمع_کړه_can_collapse(src0, src1, src2, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_لیکې(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_لیکې(dst);
        ثابت ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_لیکې(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ثابت بنية ggml_tensor * p_src2 = src2 ? &src2_collapsed : تش_اشاره;
        ثابت size_t src2_اندازه = p_src2 ? hex_پړاو_پورته(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_اندازه, kparams);
        kparams->collapse = 1;
        أرجع;
    }
    ثابت size_t src2_اندازه = src2 ? hex_پړاو_پورته(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_اندازه, kparams);
}

ساكن منطقي ggml_hexagon_precompute_دوګونی_params(
    ثابت بنية ggml_hexagon_session * sess,
    uint32_t op,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_دوګونی_kernel_params * kparams,
    ثابت بنية ggml_tensor * src2
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت size_t elem_اندازه = ggml_type_اندازه(src0->type);
    ثابت size_t src0_لیکه_اندازه = src0->ne[0] * elem_اندازه;
    ثابت size_t src1_لیکه_اندازه = src1->ne[0] * elem_اندازه;
    ثابت size_t dst_لیکه_اندازه  = dst->ne[0]  * elem_اندازه;

    ثابت size_t src0_لیکه_اندازه_aligned = hex_پړاو_پورته(src0_لیکه_اندازه, 128);
    ثابت size_t src1_لیکه_اندازه_aligned = hex_پړاو_پورته(src1_لیکه_اندازه, 128);
    ثابت size_t dst_لیکه_اندازه_aligned  = hex_پړاو_پورته(dst_لیکه_اندازه,  128);

    إذا (!htp_دوګونی_strides_خوندي(src0, src1, dst, elem_اندازه)) {
        أرجع خطأ;
    }

    ثابت منطقي is_جمع_کړه_id = op == HTP_OP_جمع_کړه_ID;
    ثابت منطقي is_scalar = !is_جمع_کړه_id && src1->ne[0] == 1;
    ثابت منطقي is_لیکه_bcast = !is_جمع_کړه_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ثابت منطقي is_same_shape = !is_جمع_کړه_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ثابت منطقي is_complex   = !is_جمع_کړه_id && !is_scalar && !is_same_shape && !is_لیکه_bcast && (src1->ne[0] == src0->ne[0]);
    ثابت منطقي is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ثابت منطقي is_scalar_خپرونه = !is_جمع_کړه_id && (ggml_nelements(src1) == 1);

    إذا (!is_جمع_کړه_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_خپرونه)) {
        ثابت uint32_t ټول_elems = (uint32_t) ggml_nelements(src0);
        ثابت uint32_t n_تارونه = sess->n_تارونه;
        ثابت uint32_t max_chunk_elems = 32768 / elem_اندازه;
        ثابت uint32_t min_chunk_elems = 256;
        ثابت uint32_t موخه_chunk_elems = hex_پړاو_پورته((ټول_elems + (2 * n_تارونه) - 1) / (2 * n_تارونه), 32);
        ثابت uint32_t chunk_اندازه = (std::min)(max_chunk_elems, (std::max)(موخه_chunk_elems, min_chunk_elems));
        ثابت uint32_t chunk_بایټونه = hex_پړاو_پورته(chunk_اندازه * elem_اندازه, 128);

        kparams->kernel_type           = HTP_دوګونی_KERNEL_CHUNKED;
        kparams->n_تارونه             = n_تارونه;
        kparams->لیکې_فی_بفر       = 1;
        kparams->src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
        kparams->src1_لیکه_اندازه_aligned = is_scalar_خپرونه ? 0 : src1_لیکه_اندازه_aligned;
        kparams->dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;
        kparams->src1_اندازه             = 0;
        kparams->chunk_اندازه            = chunk_اندازه;
        kparams->chunk_بایټونه           = chunk_بایټونه;
        kparams->is_scalar             = is_scalar_خپرونه ? 1 : 0;

        بنية htp_دوګونی_vtcm_layout L;
        htp_دوګونی_vtcm_layout_build(&L, kparams, sess->vtcm_اندازه);
        إذا (L.ټول_بایټونه == 0 || L.ټول_بایټونه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }

        kparams->vtcm_اندازه = L.ټول_بایټونه;
        أرجع ریښتیا;
    }

    تعداد htp_دوګونی_kernel_type kernel_type;
    size_t src1_اندازه = 0;
    size_t src2_اندازه = 0;

    إذا (is_جمع_کړه_id) {
        GGML_ASSERT(src2 != تش_اشاره);
        kernel_type = HTP_دوګونی_KERNEL_جمع_کړه_ID;
        src1_اندازه = hex_پړاو_پورته(src1->ne[1] * src1_لیکه_اندازه_aligned, 128);
        src2_اندازه = hex_پړاو_پورته(ggml_nelements(src2) * حجم(int32_t), 128);
        إذا (src1_اندازه + src2_اندازه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }
    } وإلا إذا (is_لیکه_bcast) {
        kernel_type = HTP_دوګونی_KERNEL_لیکه_BCAST;
        src1_اندازه = src1_لیکه_اندازه_aligned;
    } وإلا إذا (is_scalar) {
        src1_اندازه = hex_پړاو_پورته(ggml_nelements(src1) * elem_اندازه, 128);
        إذا (src1_اندازه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }
        kernel_type = HTP_دوګونی_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } وإلا إذا (is_same_shape) {
        kernel_type = HTP_دوګونی_KERNEL_SAME_SHAPE;
    } وإلا إذا (is_complex) {
        src1_اندازه = hex_پړاو_پورته(ggml_nelements(src1) * elem_اندازه + 128, 128);
        إذا (src1_اندازه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }
        kernel_type = HTP_دوګونی_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } وإلا {
        src1_اندازه = hex_پړاو_پورته(ggml_nelements(src1) * elem_اندازه + 128, 128);
        إذا (src1_اندازه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }
        kernel_type = HTP_دوګونی_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_تارونه = sess->n_تارونه;
    kparams->src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
    kparams->src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned;
    kparams->dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;
    kparams->src1_اندازه             = src1_اندازه;
    kparams->src2_اندازه             = src2_اندازه;

    ثابت منطقي src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ثابت منطقي src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ثابت منطقي dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ثابت منطقي src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ثابت منطقي src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ثابت منطقي dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    منطقي وېشل_at_ne01 = خطأ;
    إذا (src0->ne[2] > 1) {
        إذا (kernel_type == HTP_دوګونی_KERNEL_SAME_SHAPE) {
            وېشل_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } وإلا إذا (kernel_type == HTP_دوګونی_KERNEL_لیکه_BCAST) {
            وېشل_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } وإلا {
            وېشل_at_ne01 = ریښتیا;
        }
    }
    kparams->وېشل_at_ne01 = وېشل_at_ne01 ? 1 : 0;

    منطقي وېشل_at_ne02 = خطأ;
    إذا (src0->ne[3] > 1) {
        إذا (kernel_type == HTP_دوګونی_KERNEL_SAME_SHAPE) {
            وېشل_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } وإلا إذا (kernel_type == HTP_دوګونی_KERNEL_لیکه_BCAST) {
            وېشل_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } وإلا {
            وېشل_at_ne02 = ریښتیا;
        }
    }
    kparams->وېشل_at_ne02 = وېشل_at_ne02 ? 1 : 0;

    بنية htp_دوګونی_vtcm_layout L;
    htp_دوګونی_vtcm_layout_build(&L, kparams, sess->vtcm_اندازه);
    إذا (L.لیکې_فی_بفر == 0 || L.ټول_بایټونه > sess->vtcm_اندازه) {
        إذا (!is_جمع_کړه_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_خپرونه)) {
            ثابت uint32_t ټول_elems = (uint32_t) ggml_nelements(src0);
            ثابت uint32_t n_تارونه = sess->n_تارونه;
            ثابت uint32_t max_chunk_elems = 32768 / elem_اندازه;
            ثابت uint32_t min_chunk_elems = 256;
            ثابت uint32_t موخه_chunk_elems = hex_پړاو_پورته((ټول_elems + (2 * n_تارونه) - 1) / (2 * n_تارونه), 32);
            ثابت uint32_t chunk_اندازه = (std::min)(max_chunk_elems, (std::max)(موخه_chunk_elems, min_chunk_elems));
            ثابت uint32_t chunk_بایټونه = hex_پړاو_پورته(chunk_اندازه * elem_اندازه, 128);

            kparams->kernel_type           = HTP_دوګونی_KERNEL_CHUNKED;
            kparams->n_تارونه             = n_تارونه;
            kparams->لیکې_فی_بفر       = 1;
            kparams->src1_لیکه_اندازه_aligned = is_scalar_خپرونه ? 0 : src1_لیکه_اندازه_aligned;
            kparams->src1_اندازه             = 0;
            kparams->chunk_اندازه            = chunk_اندازه;
            kparams->chunk_بایټونه           = chunk_بایټونه;
            kparams->is_scalar             = is_scalar_خپرونه ? 1 : 0;

            htp_دوګونی_vtcm_layout_build(&L, kparams, sess->vtcm_اندازه);
            إذا (L.ټول_بایټونه == 0 || L.ټول_بایټونه > sess->vtcm_اندازه) {
                أرجع خطأ;
            }

            kparams->vtcm_اندازه = L.ټول_بایټونه;
            أرجع ریښتیا;
        }
        أرجع خطأ;
    }

    kparams->لیکې_فی_بفر = L.لیکې_فی_بفر;
    kparams->vtcm_اندازه       = L.ټول_بایټونه;

    أرجع ریښتیا;
}

ساكن فراغ ggml_hexagon_precompute_unary_params(
    ثابت بنية ggml_hexagon_session * sess,
    uint32_t op,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t)sess->n_تارونه, src0_nrows);

    kparams->n_تارونه = n_تارونه;

    ثابت size_t elem_اندازه = ggml_type_اندازه(src0->type);

    ثابت size_t src0_معلومات_لیکه_اندازه = src0->ne[0] * elem_اندازه;
    ثابت size_t dst_معلومات_لیکه_اندازه  = dst->ne[0]  * ggml_type_اندازه(dst->type);

    ثابت size_t src0_لیکه_اندازه_aligned = hex_پړاو_پورته(src0_معلومات_لیکه_اندازه, 128);
    ثابت size_t dst_لیکه_اندازه_aligned  = hex_پړاو_پورته(dst_معلومات_لیکه_اندازه,  128);

    kparams->src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
    kparams->dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;

    size_t src1_معلومات_لیکه_اندازه = 0;
    size_t src1_لیکه_اندازه_aligned = 0;
    منطقي خپرونه_وزن = خطأ;

    إذا (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != تش_اشاره);
        src1_معلومات_لیکه_اندازه = src1->ne[0] * ggml_type_اندازه(src1->type);
        src1_لیکه_اندازه_aligned = hex_پړاو_پورته(src1_معلومات_لیکه_اندازه, 128);
        خپرونه_وزن = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned;
    kparams->خپرونه_وزن      = خپرونه_وزن;

    بنية htp_unary_vtcm_layout L;
    uint32_t ستون_tile = 0;
    uint32_t vtcm_لیکه_فی_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                خپرونه_وزن, n_تارونه, sess->vtcm_اندازه, elem_اندازه,
                                &ستون_tile, &vtcm_لیکه_فی_thread);

    kparams->ستون_tile = ستون_tile;
    kparams->vtcm_لیکه_فی_thread = vtcm_لیکه_فی_thread;
    kparams->vtcm_اندازه = L.ټول_بایټونه;

    kparams->vtcm_src0_اندازه_فی_thread = L.src0_بایټونه;
    kparams->vtcm_src1_اندازه_فی_thread = L.src1_بایټونه;
    kparams->vtcm_dst_اندازه_فی_thread  = L.dst_بایټونه;

    kparams->vtcm_src0_اندازه = L.src0_بایټونه * n_تارونه;
    kparams->vtcm_src1_اندازه = L.src1_بایټونه * n_تارونه;
    kparams->vtcm_dst_اندازه  = L.dst_بایټونه * n_تارونه;

    kparams->بلاک = ستون_tile ? 0 : ((L.src0_بایټونه / 2) / src0_لیکه_اندازه_aligned);

    ثابت uint32_t tiles_فی_لیکه = ستون_tile > 0 ? (src0->ne[0] + ستون_tile - 1) / ستون_tile : 1;
    kparams->div_ne01  = init_fastdiv_ارزښتونه(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_ارزښتونه(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_ارزښتونه(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_ارزښتونه(tiles_فی_لیکه);
}

ساكن فراغ ggml_hexagon_precompute_get_لیکې_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_get_لیکې_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    ثابت uint32_t ne10 = src1->ne[0];
    ثابت uint32_t ne11 = src1->ne[1];
    ثابت uint32_t ne12 = src1->ne[2];
    ثابت uint32_t nr = ne10 * ne11 * ne12;

    ثابت منطقي tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحيح wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    إذا (tiled) {
        kparams->kernel_type = HTP_GET_لیکې_KERNEL_TILED;
    } وإلا إذا (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_لیکې_KERNEL_SAMETYPE;
    } وإلا {
        kparams->kernel_type = HTP_GET_لیکې_KERNEL_FLAT;
    }

    ثابت uint32_t ټول_tasks = nr;

    kparams->n_تارونه = (std::min)((uint32_t)sess->n_تارونه, ټول_tasks);

    بنية htp_get_لیکې_vtcm_layout vtcm_layout = {};
    طالما (kparams->n_تارونه > 0) {
        htp_get_لیکې_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ټول_tasks, src1->type, kparams->n_تارونه);
        إذا (vtcm_layout.ټول_بایټونه <= sess->vtcm_اندازه) {
            اكسر;
        }
        --kparams->n_تارونه;
    }

    إذا (kparams->n_تارونه == 0 && ټول_tasks > 0) {
        htp_get_لیکې_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ټول_tasks, src1->type, 1);
    }

    kparams->vtcm_اندازه = (ټول_tasks == 0) ? 0 : vtcm_layout.ټول_بایټونه;
    kparams->tasks_فی_thread = kparams->n_تارونه > 0 ? (ټول_tasks + kparams->n_تارونه - 1) / kparams->n_تارونه : 0;
    kparams->ټول_tasks = ټول_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_ارزښتونه(ne10) : fastdiv_ارزښتونه{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_ارزښتونه(ne10 * ne11) : fastdiv_ارزښتونه{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_ارزښتونه(ne02) : fastdiv_ارزښتونه{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_ارزښتونه(ne03) : fastdiv_ارزښتونه{0, 0};
}

ساكن فراغ ggml_hexagon_precompute_سېټ_لیکې_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0, //ارزښتونه
    ثابت بنية ggml_tensor * src1, // indices
    ثابت بنية ggml_tensor * dst,  // destination
    بنية htp_سېټ_لیکې_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t nr = src0->ne[1];

    kparams->n_تارونه = (std::min)((uint32_t)sess->n_تارونه, nr);
    kparams->tasks_فی_thread = (nr + kparams->n_تارونه - 1) / kparams->n_تارونه;
    kparams->ټول_tasks = nr;

    kparams->div_ne11 = init_fastdiv_ارزښتونه(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_ارزښتونه(src1->ne[2]);
    kparams->div_tasks_فی_thread = init_fastdiv_ارزښتونه(kparams->tasks_فی_thread);
    kparams->div_ne02 = init_fastdiv_ارزښتونه(src0->ne[2]);

    بنية htp_سېټ_لیکې_vtcm_layout vtcm_layout;
    htp_سېټ_لیکې_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_تارونه);
    kparams->vtcm_اندازه = vtcm_layout.ټول_بایټونه;
}

ساكن فراغ ggml_hexagon_precompute_softmax_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, src0_nrows);

    عائم scale = 1.0f;
    عائم max_bias = 0.0f;
    حافظه_نقل(&scale,    &op->op_params[0], حجم(عائم));
    حافظه_نقل(&max_bias, &op->op_params[1], حجم(عائم));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ثابت uint32_t n_سر = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));
    kparams->n_سر      = n_سر;
    kparams->n_سر_log2 = n_سر_log2;

    إذا (max_bias > 0.0f && n_سر_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_سر_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);
    } وإلا {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != تش_اشاره) ? 1 : 0;
    kparams->use_f16  = (src1 != تش_اشاره && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne10 = src1 ? src1->ne[0] : 1;

    بنية htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_تارونه);

    kparams->n_تارونه                 = n_تارونه;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_فی_thread     = (src0_nrows + n_تارونه - 1) / n_تارونه;
    kparams->vtcm_اندازه                 = (uint32_t) layout.ټول_بایټونه;
    kparams->vtcm_src0_اندازه_فی_thread = (uint32_t) layout.src0_بایټونه_فی_thread;
    kparams->vtcm_src1_اندازه_فی_thread = (uint32_t) layout.src1_بایټونه_فی_thread;
    kparams->vtcm_dst_اندازه_فی_thread  = (uint32_t) layout.dst_بایټونه_فی_thread;
    kparams->src0_لیکه_اندازه_aligned     = (uint32_t) layout.src0_spad_half_اندازه;
    kparams->src1_لیکه_اندازه_aligned     = (uint32_t) layout.src1_spad_half_اندازه;
    kparams->dst_لیکه_اندازه_aligned      = (uint32_t) layout.dst_spad_half_اندازه;
    kparams->src0_spad_half_اندازه       = (uint32_t) layout.src0_spad_half_اندازه;
    kparams->src1_spad_half_اندازه       = (uint32_t) layout.src1_spad_half_اندازه;
    kparams->dst_spad_half_اندازه        = (uint32_t) layout.dst_spad_half_اندازه;
    إذا (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } وإلا إذا (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ماسک_F16;
    } وإلا {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ماسک_F32;
    }

    إذا (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_ارزښتونه(src0->ne[1]);
    إذا (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_ارزښتونه(src0->ne[2]);
    ثابت uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت uint32_t ne13 = src1 ? src1->ne[3] : 1;
    إذا (ne12 > 0) kparams->div_ne12 = init_fastdiv_ارزښتونه(ne12);
    إذا (ne13 > 0) kparams->div_ne13 = init_fastdiv_ارزښتونه(ne13);
}

ساكن فراغ ggml_hexagon_precompute_rope_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * dst  = op;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    بنية htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_تارونه, n_freq_factors);

    kparams->n_تارونه              = n_تارونه;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_فی_thread  = (src0_nrows + n_تارونه - 1) / n_تارونه;
    kparams->vtcm_اندازه              = (uint32_t) layout.ټول_بایټونه;
    kparams->spad_فی_thread        = (uint32_t) layout.بایټونه_فی_thread;
    kparams->theta_cache_دننه_واټن     = (uint32_t) layout.theta_cache_اندازه_aligned;
    kparams->src0_لیکه_اندازه_aligned  = (uint32_t) layout.src0_لیکه_اندازه_aligned;
    kparams->freq_factors_دننه_واټن    = (uint32_t) (layout.بایټونه_فی_thread * n_تارونه);
    kparams->freq_factors_اندازه      = (uint32_t) layout.freq_factors_اندازه_aligned;

    إذا (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_ارزښتونه(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_ارزښتونه(dst->ne[1]);
    }
}

ساكن فراغ ggml_hexagon_precompute_ssm_conv_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * dst,
    بنية htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t d_conv  = (uint32_t) src1->ne[0];
    ثابت uint32_t d_inner = (uint32_t) src0->ne[1];
    ثابت uint32_t n_t     = (uint32_t) dst->ne[1];
    ثابت uint32_t n_s     = (uint32_t) dst->ne[2];
    ثابت uint32_t ncs     = (uint32_t) src0->ne[0];

    ثابت uint32_t n_تارونه = (std::min)((uint32_t) sess->n_تارونه, (d_inner + 31) / 32);

    kparams->n_تارونه = n_تارونه;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ثابت uint32_t raw_rpt = (d_inner + n_تارونه - 1) / n_تارونه;
    ثابت uint32_t d_inner_فی_thread = hex_پړاو_پورته(raw_rpt, 32);

    ثابت uint32_t src1_raw_بایټونه = hex_پړاو_پورته(d_inner_فی_thread * d_conv * حجم(عائم), 128) + 128;
    ثابت uint32_t src1_T_بایټونه   = hex_پړاو_پورته(d_conv * d_inner_فی_thread * حجم(عائم), 128);
    ثابت uint32_t vtcm_src1_فی_thread = src1_raw_بایټونه + src1_T_بایټونه;

    uint32_t vtcm_src0_فی_thread = 0;
    uint32_t vtcm_dst_فی_thread  = 0;

    إذا (n_t == 1) {
        kparams->d_inner_tile = d_inner_فی_thread;

        ثابت uint32_t src0_tile_raw_بایټونه = hex_پړاو_پورته(d_inner_فی_thread * d_conv * حجم(عائم), 128);
        ثابت uint32_t src0_T_بایټونه        = hex_پړاو_پورته(d_conv * d_inner_فی_thread * حجم(عائم), 128);
        vtcm_src0_فی_thread = 2 * src0_tile_raw_بایټونه + src0_T_بایټونه;

        ثابت uint32_t dst_tile_بایټونه = hex_پړاو_پورته(d_inner_فی_thread * حجم(عائم), 128);
        vtcm_dst_فی_thread = 2 * dst_tile_بایټونه;
    } وإلا {
        ثابت size_t vtcm_budget = (sess->vtcm_اندازه > 0 ? sess->vtcm_اندازه / n_تارونه : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ثابت uint32_t src0_بلاک_T    = hex_پړاو_پورته(ncs * 32 * حجم(عائم), 128);
        ثابت size_t   fixed_بایټونه     = vtcm_src1_فی_thread + src0_بلاک_T;
        ثابت size_t   avail_for_tiles = vtcm_budget > fixed_بایټونه ? vtcm_budget - fixed_بایټونه : (128 * 1024);

        uint32_t موخه_max_tile = hex_پړاو_پورته((d_inner_فی_thread + 3) / 4, 32);
        موخه_max_tile = (std::max)(موخه_max_tile, 32u);
        موخه_max_tile = (std::min)(موخه_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * حجم(عائم)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        إذا (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        إذا (d_inner_tile > موخه_max_tile) {
            d_inner_tile = موخه_max_tile;
        }
        إذا (d_inner_tile > d_inner_فی_thread) {
            d_inner_tile = d_inner_فی_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ثابت uint32_t src0_tile_raw = hex_پړاو_پورته(d_inner_tile * ncs * حجم(عائم), 128);
        vtcm_src0_فی_thread = 2 * src0_tile_raw + src0_بلاک_T;

        vtcm_dst_فی_thread = 2 * hex_پړاو_پورته(d_inner_tile * n_t * حجم(عائم), 128);
    }

    kparams->vtcm_src0_اندازه_فی_thread = vtcm_src0_فی_thread;
    kparams->vtcm_src1_اندازه_فی_thread = vtcm_src1_فی_thread;
    kparams->vtcm_dst_اندازه_فی_thread  = vtcm_dst_فی_thread;

    kparams->vtcm_src0_اندازه = vtcm_src0_فی_thread * n_تارونه;
    kparams->vtcm_src1_اندازه = vtcm_src1_فی_thread * n_تارونه;
    kparams->vtcm_dst_اندازه  = vtcm_dst_فی_thread  * n_تارونه;
    kparams->vtcm_اندازه      = kparams->vtcm_src0_اندازه + kparams->vtcm_src1_اندازه + kparams->vtcm_dst_اندازه;
}

ساكن فراغ ggml_hexagon_precompute_gated_delta_net_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت بنية ggml_tensor * q     = op->src[0];
    ثابت بنية ggml_tensor * k     = op->src[1];
    ثابت بنية ggml_tensor * v     = op->src[2];
    ثابت بنية ggml_tensor * g     = op->src[3];
    ثابت بنية ggml_tensor * state = op->src[5];

    ثابت uint32_t S_v      = (uint32_t) v->ne[0];
    ثابت uint32_t H        = (uint32_t) v->ne[1];
    ثابت uint32_t n_نښې = (uint32_t) v->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) v->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ثابت uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ثابت uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ثابت uint32_t ټول_لیکې = H * n_seqs;
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, ټول_لیکې);

    ثابت منطقي can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_نښې >= HTP_GDN_MIN_نښې) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    بنية htp_gdn_hmx_vtcm_layout hmx_layout;
    بنية htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    إذا (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_اندازه, ټول_لیکې, sess->vtcm_اندازه, n_تارونه, ریښتیا, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_اندازه      = HTP_GDN_CHUNK_اندازه;
        kparams->n_chunks        = (n_نښې + HTP_GDN_CHUNK_اندازه - 1) / HTP_GDN_CHUNK_اندازه;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_اندازه       = (uint32_t) hmx_layout.ټول_بایټونه;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_بایټونه;
        kparams->vtcm_فی_thread = (uint32_t) (hmx_layout.ټول_بایټونه / (n_تارونه > 0 ? n_تارونه : 1));
    } وإلا {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_تارونه);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_فی_thread = (uint32_t) hvx_layout.بایټونه_فی_thread;
        kparams->vtcm_اندازه       = (uint32_t) hvx_layout.ټول_بایټونه;
    }

    kparams->n_تارونه           = n_تارونه;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_نښې            = n_نښې;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->ټول_لیکې          = ټول_لیکې;
    kparams->لیکې_فی_thread     = (ټول_لیکې + kparams->n_تارونه - 1) / kparams->n_تارونه;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / حجم(عائم));
    kparams->state_اندازه_فی_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((عائم) S_v);

    إذا (H > 0)                  kparams->div_H         = init_fastdiv_ارزښتونه(H);
    إذا (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_ارزښتونه((uint32_t) q->ne[1]);
    إذا (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_ارزښتونه((uint32_t) k->ne[1]);
    إذا (rq3 > 0)                kparams->div_rq3       = init_fastdiv_ارزښتونه(rq3);
    إذا (rk3 > 0)                kparams->div_rk3       = init_fastdiv_ارزښتونه(rk3);
    إذا (kparams->n_تارونه > 0) kparams->div_n_تارونه = init_fastdiv_ارزښتونه(kparams->n_تارونه);
}

ساكن فراغ ggml_hexagon_precompute_ترتیبول_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    منطقي is_سر_k,
    بنية htp_ترتیبول_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    ثابت uint32_t ټول_لیکې = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t ne00       = src0->ne[0];
    ثابت uint32_t k          = dst->ne[0];

    int32_t ترتیب = GGML_ترتیبول_ترتیب_DESC;
    إذا (!is_سر_k) {
        ترتیب = ((ثابت int32_t *) op->op_params)[0];
    }

    ثابت uint32_t n_تارونه_max = sess->n_تارونه > 0 ? sess->n_تارونه : 4;
    ثابت size_t vtcm_budget = sess->vtcm_اندازه > 0 ? sess->vtcm_اندازه : (8 * 1024 * 1024);

    بنية htp_ترتیبول_vtcm_layout layout;
    منطقي ok = htp_ترتیبول_solve_layout(&layout, ne00, ټول_لیکې, k, n_تارونه_max, vtcm_budget, is_سر_k);
    GGML_ASSERT(ok);

    kparams->n_تارونه         = (int32_t) layout.n_تارونه;
    kparams->ټول_لیکې        = (int32_t) ټول_لیکې;
    kparams->لیکه_پیل         = 0;
    kparams->لیکه_پای_ټکی           = (int32_t) ټول_لیکې;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->ترتیب             = ترتیب;
    kparams->is_سر_k          = is_سر_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_اندازه         = (int32_t) layout.ټول_بایټونه;
    kparams->phase1_خونه_اندازه  = (int32_t) layout.phase1_خونه_اندازه;
    kparams->یوځای_ارزښتونه_off  = (int32_t) layout.یوځای_ارزښتونه_off;
    kparams->یوځای_indices_off = (int32_t) layout.یوځای_indices_off;
    kparams->یوځای_elems       = (int32_t) layout.یوځای_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ساكن فراغ ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ثابت uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ثابت int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    إذا (numer_hi < 0) {
        *ox_hi = 0;
    } وإلا {
        ثابت uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    إذا (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ساكن منطقي ggml_hexagon_pool_shape_ځاییږي(
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * dst
) {
    أرجع ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ساكن فراغ ggml_hexagon_precompute_pool_2d_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * dst,
    بنية htp_pool_2d_kernel_params * kparams,
    منطقي is_pool_1d
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t src_x = (uint32_t) src0->ne[0];
    ثابت uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ثابت uint32_t dst_x = (uint32_t) dst->ne[0];
    ثابت uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ثابت uint32_t src_plane_بایټونه = src_x * src_y * حجم(عائم);
    ثابت uint32_t dst_plane_بایټونه = dst_x * dst_y * حجم(عائم);
    ثابت uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ثابت uint32_t n_تارونه = (std::min)((uint32_t) sess->n_تارونه, planes > 0 ? planes : 1);

    بنية htp_pool_vtcm_layout layout;
    ثابت منطقي ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_تارونه, sess->vtcm_اندازه);
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
    kparams->src_plane_بایټونه = src_plane_بایټونه;
    kparams->dst_plane_بایټونه = dst_plane_بایټونه;
    kparams->src_plane_بایټونه_aligned = (uint32_t) layout.src_spad_half_اندازه;
    kparams->dst_plane_بایټونه_aligned = (uint32_t) layout.dst_spad_half_اندازه;
    kparams->n_تارونه = n_تارونه;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_لار = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_لار = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->بلاک_لار = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_شمېر = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ثابت منطقي narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_لار = (kparams->fast_لار && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (عائم) (kparams->kernel_x * kparams->kernel_y);
}

ساكن منطقي ggml_hexagon_precompute_concat_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!src0 || !src1 || !dst) {
        أرجع خطأ;
    }

    صحيح dim = ((ثابت int32_t *) op->op_params)[0];
    إذا (dim < 0 || dim >= GGML_MAX_DIMS) {
        أرجع خطأ;
    }
    kparams->dim = dim;

    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }
    إذا (src0->type != dst->type || src1->type != dst->type) {
        أرجع خطأ;
    }

    ثابت uint32_t type_اندازه = ggml_type_اندازه(dst->type);

    لكل (صحيح d = 0; d < GGML_MAX_DIMS; d++) {
        ثابت int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        إذا (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            أرجع خطأ;
        }
    }

    ثابت uint32_t max_gather_ne0 = 128 / type_اندازه;
    ثابت uint32_t ټول_لیکې = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ثابت منطقي b_dense = (src1->nb[0] == type_اندازه);
    ثابت منطقي gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_اندازه &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_اندازه
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_اندازه);

    إذا (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_اندازه && dst->nb[1] == (size_t) dst->ne[0] * type_اندازه &&
        src0->nb[0] == type_اندازه &&
        (src1->nb[0] == type_اندازه || src1->nb[1] == type_اندازه) &&
        gather_strides_ok &&
        htp_gather_لیکې_tensor_outer_contiguous(dst) &&
        htp_gather_لیکې_tensor_outer_contiguous(src0) &&
        htp_gather_لیکې_tensor_outer_contiguous(src1)) {

        إذا (htp_gather_لیکې_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_اندازه, ټول_لیکې,
                                         sess->n_تارونه, sess->mdev.count, sess->vtcm_اندازه)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_لیکې;
            kparams->n_تارونه   = (uint8_t) sess->n_تارونه;
            kparams->vtcm_اندازه   = kparams->u.gather.ټوټه_2 * sess->n_تارونه;
            أرجع ریښتیا;
        }
    }

    ثابت منطقي dma_strides_ok = (src0->nb[0] == type_اندازه && src1->nb[0] == type_اندازه && dst->nb[0] == type_اندازه);

    إذا (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_تارونه   = 1;
        أرجع ریښتیا;
    }

    ثابت منطقي is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ثابت منطقي is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ثابت منطقي transposed_لیکې_ok = (src0->nb[0] == type_اندازه && src1->nb[1] == type_اندازه && dst->nb[0] == type_اندازه);

    إذا (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_لیکې_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ثابت uint32_t n_تارونه = sess->n_تارونه > 0 ? (uint32_t) sess->n_تارونه : 8;
        بنية htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_اندازه, n_تارونه);

        إذا (sess->vtcm_اندازه > 0 && layout.ټول_بایټونه > sess->vtcm_اندازه) {
            أرجع خطأ;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_تارونه                         = n_تارونه;
        kparams->vtcm_اندازه                         = layout.ټول_بایټونه;
        kparams->u.transposed.spad0_اندازه_فی_thread = layout.src0_spad_اندازه_فی_thread;
        kparams->u.transposed.spad1_اندازه_فی_thread = layout.src1_spad_اندازه_فی_thread;
        أرجع ریښتیا;
    }

    أرجع خطأ;
}

ساكن منطقي ggml_hexagon_precompute_cpy_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!src0 || !dst) {
        أرجع خطأ;
    }

    إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }
    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    ثابت int64_t nelem_src = ggml_nelements(src0);
    ثابت int64_t nelem_dst = ggml_nelements(dst);
    إذا (nelem_src != nelem_dst || nelem_src < 0) {
        أرجع خطأ;
    }

    ثابت uint32_t src_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    kparams->src0_type_اندازه = (uint8_t) src_type_اندازه;
    kparams->dst_type_اندازه  = (uint8_t) dst_type_اندازه;
    kparams->ټول_elems    = (uint32_t) nelem_src;

    إذا (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        أرجع ریښتیا;
    }

    إذا (nelem_src == 1) {
        إذا (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            أرجع ریښتیا;
        }
        إذا ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            أرجع ریښتیا;
        }
        أرجع خطأ;
    }

    ثابت منطقي sametype = (src0->type == dst->type);

    منطقي same_extents = ریښتیا;
    لكل (صحيح d = 0; d < GGML_MAX_DIMS; d++) {
        إذا (src0->ne[d] != dst->ne[d]) {
            same_extents = خطأ;
            اكسر;
        }
    }

    ثابت منطقي transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_اندازه) || (dst->nb[0] != dst_type_اندازه) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_اندازه) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_اندازه);
    ثابت منطقي sameshape  = same_extents && !transposed;

    ثابت منطقي src_is_contiguous = ggml_is_contiguous(src0);
    ثابت منطقي dst_is_contiguous = ggml_is_contiguous(dst);

    إذا (sametype) {
        إذا (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            أرجع ریښتیا;
        }

        ثابت uint32_t max_gather_ne0 = 128 / src_type_اندازه;
        ثابت uint32_t ټول_لیکې = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        إذا ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_اندازه && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_اندازه &&
            ټول_لیکې > 1 &&
            htp_gather_لیکې_tensor_outer_contiguous(src0)) {

            إذا (htp_gather_لیکې_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, خطأ,
                                             src_type_اندازه, ټول_لیکې,
                                             sess->n_تارونه, sess->mdev.count, sess->vtcm_اندازه)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_لیکې;
                kparams->n_تارونه   = (uint8_t) sess->n_تارونه;
                kparams->vtcm_اندازه   = kparams->u.gather.ټوټه_2 * sess->n_تارونه;
                أرجع ریښتیا;
            }
        }

        إذا (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->ټول_لیکې  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            أرجع ریښتیا;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_تارونه   = sess->n_تارونه > 0 ? (uint8_t) sess->n_تارونه : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_ارزښتونه((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_ارزښتونه((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_ارزښتونه((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_ارزښتونه((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_ارزښتونه((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_ارزښتونه((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        أرجع ریښتیا;
    }

    إذا (!sameshape) {
        أرجع خطأ;
    }

    ثابت منطقي valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    إذا (!valid_conversion) {
        أرجع خطأ;
    }

    ثابت uint32_t n_تارونه = sess->n_تارونه > 0 ? (uint32_t) sess->n_تارونه : 4;
    بنية htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_اندازه, dst_type_اندازه, n_تارونه);

    إذا (sess->vtcm_اندازه > 0 && layout.ټول_بایټونه > sess->vtcm_اندازه) {
        أرجع خطأ;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->ټول_لیکې              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_تارونه               = (uint8_t) n_تارونه;
    kparams->vtcm_اندازه               = layout.ټول_بایټونه;
    kparams->u.convert.src0_buf_اندازه = layout.src0_buf_اندازه;
    kparams->u.convert.dst_buf_اندازه  = layout.dst_buf_اندازه;
    kparams->u.convert.spad0_اندازه_فی_thread = layout.spad0_اندازه_فی_thread;
    kparams->u.convert.spad1_اندازه_فی_thread = layout.spad1_اندازه_فی_thread;
    kparams->u.convert.src0_لیکه_stride = layout.src0_لیکه_stride;
    kparams->u.convert.dst_لیکه_stride  = layout.dst_لیکه_stride;
    kparams->u.convert.blk_لیکې        = layout.blk_لیکې;
    kparams->u.convert.div_ne01      = init_fastdiv_ارزښتونه((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_ارزښتونه((uint32_t) (src0->ne[2] * src0->ne[1]));

    أرجع ریښتیا;
}

ساكن فراغ ggml_hexagon_precompute_fused_mmnx_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0, // W0
    ثابت بنية ggml_tensor * act,  // x
    int32_t n_weights,
    بنية htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->n_تارونه = sess->n_تارونه;

    ثابت صحيح ne00 = src0->ne[0];
    ثابت صحيح ne01 = src0->ne[1];
    ثابت صحيح ne02 = src0->ne[2];
    ثابت صحيح ne03 = src0->ne[3];

    ثابت صحيح ne10 = act->ne[0];
    ثابت صحيح ne11 = act->ne[1];
    ثابت صحيح ne12 = act->ne[2];
    ثابت صحيح ne13 = act->ne[3];

    ثابت منطقي is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحيح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحيح ne00_padded = is_tiled ? hex_پړاو_پورته(ne00, 32) : ne00;
    ثابت صحيح ne11_padded = hex_پړاو_پورته(ne11, 32);
    ثابت صحيح ne01_tiled  = hex_پړاو_پورته(ne01, 32);

    ثابت size_t vtcm_budget = sess->vtcm_اندازه;
    ثابت منطقي is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    منطقي hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    إذا (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, تش_اشاره, خطأ, is_batched)) {
        إذا (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, تش_اشاره, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, خطأ, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            اذهب finalize;
        }
    }

    إذا (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        أرجع;
    }

    {
        ثابت صحيح act_nrows = ne11 * ne12 * ne13;
        ثابت size_t act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
        ثابت size_t src0_لیکه_اندازه = src0->nb[1];

        ثابت uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t غوره_n_prefetch = 2;
        لكل (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            بنية htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_لیکه, wtype, ne10, act_nrows, sess->n_تارونه,
                0, src0_لیکه_اندازه, act_لیکه_اندازه, 0, d, خطأ, ریښتیا
            );
            إذا (L.ټول_بایټونه <= sess->vtcm_اندازه) {
                غوره_n_prefetch = d;
                اكسر;
            }
        }

        بنية htp_mm_hvx_vtcm_layout L;
        منطقي try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_لیکه, wtype, ne10, act_nrows, sess->n_تارونه,
            0, src0_لیکه_اندازه, act_لیکه_اندازه, 0, غوره_n_prefetch, خطأ, ریښتیا
        );

        إذا (try_tiled && L.ټول_بایټونه <= sess->vtcm_اندازه) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_لیکه;
            kparams->tile_اندازه         = htp_tensor_get_tile_اندازه(wtype);
            kparams->aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
            kparams->act_لیکه_اندازه      = act_لیکه_اندازه;
            kparams->vtcm_src0_اندازه    = L.src0_بایټونه;
            kparams->vtcm_act_اندازه     = L.act_بایټونه;
            kparams->vtcm_bias_اندازه    = 0;
            kparams->vtcm_dst_اندازه     = L.dst_بایټونه;
            kparams->vtcm_اندازه         = L.ټول_بایټونه;
            kparams->n_prefetch        = غوره_n_prefetch;
            kparams->n_weights         = n_weights;
        } وإلا {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            أرجع;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ارزښتونه(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ارزښتونه(ne11);
    kparams->div_r2       = init_fastdiv_ارزښتونه(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ارزښتونه(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ارزښتونه(ne12);
}

ساكن فراغ ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * src0, // W0
    ثابت بنية ggml_tensor * act,  // x
    ثابت بنية ggml_tensor * dst,  // dst0
    int32_t n_weights,
    بنية htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ساكن منطقي ggml_hexagon_op_mul_mat_use_fwht(ثابت بنية ggml_tensor * dst) {
    إذا (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        أرجع خطأ;
    }

    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    ثابت int64_t n = src1->ne[0];
    إذا (n < 32 || (n & (n - 1)) != 0) {
        أرجع خطأ;
    }

    أرجع src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_لیکې(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ساكن منطقي ggml_hexagon_supported_mul_mat(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * dst) {
    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    إذا (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        إذا (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            أرجع خطأ;
        }
        بنية htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, تش_اشاره, dst, &ukparams);
        أرجع ukparams.vtcm_لیکه_فی_thread > 0 && (size_t) ukparams.vtcm_اندازه <= sess->vtcm_اندازه;
    }

    إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    بدل (src0->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q2_K: {
            إذا (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                أرجع خطأ;
            }

            ثابت int64_t qk = ggml_blck_اندازه(src0->type);
            إذا (src0->ne[0] % qk) {
                أرجع خطأ;
            }

            إذا (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                أرجع خطأ;
            }
            اكسر;
        }

        حالة GGML_TYPE_F16:
            إذا (src0->nb[1] < src0->nb[0]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                أرجع خطأ;
            }
            اكسر;

        حالة GGML_TYPE_F32:
            إذا (src1->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }
            إذا (src0->nb[1] < src0->nb[0]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                أرجع خطأ;
            }
            اكسر;

        حالة GGML_TYPE_BF16:
            إذا (src0->nb[1] < src0->nb[0]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                أرجع خطأ;
            }
            إذا (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                أرجع خطأ;
            }
            اكسر;

        افتراضي:
            أرجع خطأ;
    }

    بنية htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    إذا (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_mul_mat_id(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    إذا (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        أرجع خطأ;
    }

    بدل (src0->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q2_K:
            إذا (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                أرجع خطأ;
            }

            إذا (src0->ne[0] % ggml_blck_اندازه(src0->type) != 0) {
                أرجع خطأ;
            }
            اكسر;

        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_BF16:
            إذا (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                أرجع خطأ;
            }
            إذا (src0->ne[0] % 32 != 0) {
                أرجع خطأ;
            }
            اكسر;

        افتراضي:
            أرجع خطأ;
    }

    بنية htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    إذا (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_اندازه, sess->vtcm_اندازه);
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_دوګونی(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type == GGML_TYPE_F32) {
        إذا (src1->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        إذا (dst->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
    }
    وإلا إذا (src0->type == GGML_TYPE_F16) {
        إذا (src1->type != GGML_TYPE_F16) {
            أرجع خطأ;
        }
        إذا (dst->type != GGML_TYPE_F16) {
            أرجع خطأ;
        }
    }
    وإلا {
        أرجع خطأ;
    }

    إذا (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        أرجع خطأ;
    }
    إذا (!ggml_are_same_shape(src0, dst)) {
        أرجع خطأ;
    }
    إذا (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        أرجع خطأ;
    }

    بنية htp_دوګونی_kernel_params kparams;
    أرجع ggml_hexagon_precompute_دوګونی_params(sess, op_remap_منزل_htp(op), src0, src1, dst, &kparams);
}

ساكن منطقي ggml_hexagon_supported_جمع_کړه_id(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!src2) {
        أرجع خطأ;
    }
    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }
    إذا (!ggml_are_same_shape(src0, dst)) {
        أرجع خطأ;
    }
    إذا (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        أرجع خطأ;
    }
    إذا (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        أرجع خطأ;
    }
    إذا (src0->nb[0] != حجم(عائم) || src1->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t)) {
        أرجع خطأ;
    }

    // REVISIT: add support for non-contiguous tensors
    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }
    إذا (src2->nb[1] < src2->ne[0] * حجم(int32_t)) {
        أرجع خطأ;
    }

    بنية htp_دوګونی_kernel_params kparams;
    أرجع ggml_hexagon_precompute_دوګونی_params(sess, HTP_OP_جمع_کړه_ID, src0, src1, dst, &kparams, src2);
}

ساكن منطقي ggml_hexagon_supported_unary(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }
    إذا (dst->type != src0->type) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous_لیکې(src0)) {
        أرجع خطأ;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    إذا (src0->type == GGML_TYPE_F16) {
        بدل (op->op) {
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_L2_NORM:
            حالة GGML_OP_SCALE:
            حالة GGML_OP_CLAMP:
            حالة GGML_OP_SQR:
            حالة GGML_OP_SQRT:
            حالة GGML_OP_LOG:
                اكسر;
            حالة GGML_OP_UNARY:
                إذا (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_ګام) {
                    أرجع خطأ;
                }
                اكسر;
            افتراضي:
                أرجع خطأ;
        }
    }

    إذا (!ggml_are_same_shape(src0, dst)) {
        أرجع خطأ;
    }

    // dst must be contiguous; src0 may be non-contiguous
    إذا (!ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_مجموعه(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_مجموعه_لیکې(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    // TODO: add support for non-contigiuos tensors
    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_argmax(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (dst->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_pool_2d(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    إذا (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        أرجع خطأ;
    }
    إذا (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        أرجع خطأ;
    }
    إذا (!ggml_hexagon_pool_shape_ځاییږي(src0, op)) {
        أرجع خطأ;
    }

    ثابت int32_t kernel_x = params[1];
    ثابت int32_t kernel_y = params[2];
    ثابت int32_t stride_x = params[3];
    ثابت int32_t stride_y = params[4];
    ثابت int32_t pad_x    = params[5];
    ثابت int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    إذا (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        أرجع خطأ;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_تارونه = (std::min)((uint32_t) sess->n_تارونه, planes > 0 ? planes : 1);

    بنية htp_pool_vtcm_layout layout;
    أرجع htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_تارونه, sess->vtcm_اندازه);
}

ساكن منطقي ggml_hexagon_supported_pool_1d(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    إذا (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        أرجع خطأ;
    }
    إذا (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        أرجع خطأ;
    }
    إذا (!ggml_hexagon_pool_shape_ځاییږي(src0, op)) {
        أرجع خطأ;
    }

    ثابت int32_t kernel = params[1];
    ثابت int32_t stride = params[2];
    ثابت int32_t pad    = params[3];
    إذا (kernel <= 0 || stride <= 0 || pad < 0) {
        أرجع خطأ;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_تارونه = (std::min)((uint32_t) sess->n_تارونه, planes > 0 ? planes : 1);

    بنية htp_pool_vtcm_layout layout;
    أرجع htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_تارونه, sess->vtcm_اندازه);
}

ساكن منطقي ggml_hexagon_supported_activations(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }
    إذا (dst->type != src0->type) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous_1(src0)) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    إذا (src1) {
        إذا (src1->type != src0->type) {
            أرجع خطأ;
        }
        إذا (!ggml_are_same_shape(src0, src1)) {
            أرجع خطأ;
        }
        إذا (!ggml_is_contiguous_1(src1)) {
            أرجع خطأ;
        }
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_softmax(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src2) {
        أرجع خطأ;  // FIXME: add support for sinks
    }

    إذا (src0->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (src0->ne[2] > 512) {
        أرجع خطأ;
    }

    إذا (src1) {
        إذا (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            أرجع خطأ;
        }
        إذا (src0->ne[0] != src1->ne[0]) {
            أرجع خطأ;
        }
        إذا (src1->ne[1] < src0->ne[1]) {
            أرجع خطأ;
        }
        إذا (src0->ne[2] % src1->ne[2] != 0) {
            أرجع خطأ;
        }
        إذا (src0->ne[3] % src1->ne[3] != 0) {
            أرجع خطأ;
        }
    }

    إذا (src1) {
        إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            أرجع خطأ;
        }
    } وإلا {
        إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            أرجع خطأ;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ثابت int64_t ne0 = src0->ne[0];
    إذا (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        أرجع خطأ;
    }

    // HVX vector size constraints for softmax
    #تعریف_کړه SOFTMAX_MAX_لیکه_اندازه 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    إذا (ne0 > SOFTMAX_MAX_لیکه_اندازه) {
        أرجع خطأ;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه  = (std::min)((uint32_t) sess->n_تارونه, src0_nrows);
    بنية htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != تش_اشاره, src1 && src1->type == GGML_TYPE_F16, n_تارونه);
    إذا (layout.ټول_بایټونه > sess->vtcm_اندازه) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_سېټ_لیکې(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0]; //ارزښتونه
    ثابت بنية ggml_tensor * src1 = op->src[1]; // indices
    ثابت بنية ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    إذا (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        أرجع خطأ;
    }

    إذا (src0->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        أرجع خطأ;
    }

    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_get_لیکې(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0]; //ارزښتونه
    ثابت بنية ggml_tensor * src1 = op->src[1]; // indices
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous_لیکې(src0) || !ggml_is_contiguous_لیکې(src1) || !ggml_is_contiguous_لیکې(dst)) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        أرجع خطأ;
    }

    إذا (src0->type == dst->type) {
        إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            أرجع خطأ;
        }
    } وإلا إذا (src0->type == GGML_TYPE_I32) {
        أرجع خطأ;
    } وإلا إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    إذا (ggml_is_تشه(op)) {
        أرجع ریښتیا;
    }

    بنية htp_get_لیکې_kernel_params kparams;
    ggml_hexagon_precompute_get_لیکې_params(sess, src0, src1, dst, &kparams);
    إذا (kparams.n_تارونه == 0 || (size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_argsort(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0]; //ارزښتونه
    ثابت بنية ggml_tensor * dst  = op;         // indices

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    ثابت uint32_t ټول_لیکې = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه_max = sess->n_تارونه > 0 ? sess->n_تارونه : 4;
    ثابت size_t vtcm_budget = sess->vtcm_اندازه > 0 ? sess->vtcm_اندازه : (8 * 1024 * 1024);

    بنية htp_ترتیبول_vtcm_layout layout;
    إذا (!htp_ترتیبول_solve_layout(&layout, src0->ne[0], ټول_لیکې, dst->ne[0], n_تارونه_max, vtcm_budget, خطأ)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_سر_k(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0]; //ارزښتونه
    ثابت بنية ggml_tensor * dst  = op;         // indices

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    ثابت uint32_t ټول_لیکې = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_تارونه_max = sess->n_تارونه > 0 ? sess->n_تارونه : 4;
    ثابت size_t vtcm_budget = sess->vtcm_اندازه > 0 ? sess->vtcm_اندازه : (8 * 1024 * 1024);

    بنية htp_ترتیبول_vtcm_layout layout;
    إذا (!htp_ترتیبول_solve_layout(&layout, src0->ne[0], ټول_لیکې, dst->ne[0], n_تارونه_max, vtcm_budget, ریښتیا)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_rope(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * src2 = op->src[2];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!ggml_are_same_shape(src0, dst)) {
        أرجع خطأ;
    }

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    إذا (src0->ne[0] <= 0) {
        أرجع خطأ;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    إذا (src0_nrows == 0) {
        أرجع خطأ;
    }

    ثابت int32_t * op_params = &op->op_params[0];
    ثابت صحيح n_dims = op_params[1];
    ثابت صحيح mode   = op_params[2];
    ثابت صحيح n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    إذا (n_dims < 0 || n_dims % 2 != 0) {
        أرجع خطأ;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    إذا (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        أرجع خطأ;
    }

    عائم freq_بنسټ;
    حافظه_نقل(&freq_بنسټ, op_params + 5, حجم(عائم));
    إذا (freq_بنسټ < 0.0f) {
        أرجع خطأ;
    }

    إذا (mode != GGML_ROPE_TYPE_نورمال &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        أرجع خطأ;
    }

    ثابت منطقي is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    إذا (mode == GGML_ROPE_TYPE_VISION) {
        إذا (n_dims != (صحيح) (src0->ne[0] / 2) || n_offs != 0) {
            أرجع خطأ;
        }
    }

    إذا (is_mrope) {
        ثابت int32_t * sections = op_params + 11;
        إذا (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            أرجع خطأ;
        }
    }

    ثابت int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    إذا (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        أرجع خطأ;
    }

    إذا (src2) {
        إذا (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            أرجع خطأ;
        }
        إذا (src2->ne[0] < (n_dims / 2)) {
            أرجع خطأ;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    إذا (src0->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم)) {
        أرجع خطأ;
    }
    إذا (src0->nb[1] < src0->ne[0] * حجم(عائم) || dst->nb[1] < dst->ne[0] * حجم(عائم)) {
        أرجع خطأ;
    }

    ثابت uint32_t n_تارونه = (std::min)((uint32_t) sess->n_تارونه, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    بنية htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_تارونه, n_freq_factors);
    إذا (layout.ټول_بایټونه > sess->vtcm_اندازه) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_ssm_conv(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    // Only support FP32 for now
    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    // Check IO tensor shapes and dims
    إذا (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        أرجع خطأ; // src0 should be effectively 3D
    }

    ثابت صحيح d_conv  = src1->ne[0];
    ثابت صحيح d_inner = src0->ne[1];
    ثابت صحيح n_t = dst->ne[1];
    ثابت صحيح n_s = dst->ne[2];

    إذا (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        أرجع خطأ;
    }
    إذا (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        أرجع خطأ;
    }
    إذا (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        أرجع خطأ;
    }
    إذا (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        أرجع خطأ;
    }
    إذا (src0->nb[0] != حجم(عائم) || src1->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم)) {
        أرجع خطأ;
    }
    إذا (src0->nb[1] != src0->ne[0] * حجم(عائم) || src1->nb[1] != src1->ne[0] * حجم(عائم)) {
        أرجع خطأ;
    }

    بنية htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    إذا ((size_t) kparams.vtcm_اندازه > sess->vtcm_اندازه) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي im2col_3d_calc_output_اندازه(uint32_t ننوت, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    إذا (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        أرجع خطأ;
    }

    ثابت int64_t numerator = (int64_t) ننوت + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    إذا (numerator < 0) {
        أرجع خطأ;
    }

    ثابت uint64_t size = (uint64_t) (numerator / stride) + 1;
    إذا (size == 0 || size > UINT32_MAX) {
        أرجع خطأ;
    }

    *output = (uint32_t) size;
    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_precompute_im2col_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_IM2_ستون_KERNEL_UNSUPPORTED;

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!src0 || !src1 || !dst) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        أرجع خطأ;
    }

    إذا (src1->nb[0] != حجم(عائم) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    ثابت int32_t * op_params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = op_params[0];
    ثابت int32_t s1 = op_params[1];
    ثابت int32_t p0 = op_params[2];
    ثابت int32_t p1 = op_params[3];
    ثابت int32_t d0 = op_params[4];
    ثابت int32_t d1 = op_params[5];
    ثابت منطقي is_2D = (op_params[6] == 1);

    إذا (s0 <= 0 || d0 <= 0 || p0 < 0) {
        أرجع خطأ;
    }
    إذا (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        أرجع خطأ;
    }

    ثابت uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ثابت uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ثابت uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ثابت uint32_t IW = (uint32_t) src1->ne[0];
    ثابت uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ثابت uint32_t OW = (uint32_t) dst->ne[1];

    إذا (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        أرجع خطأ;
    }

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ثابت uint64_t ټول_لیکې_64   = (uint64_t) N * OH;
    إذا (patch_stride_64 > UINT32_MAX || ټول_لیکې_64 > UINT32_MAX) {
        أرجع خطأ;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t ټول_لیکې   = (uint32_t) ټول_لیکې_64;

    إذا (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        أرجع خطأ;
    }
    إذا (is_2D && dst->ne[2] != OH) {
        أرجع خطأ;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(عائم);

    منطقي is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    إذا (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ثابت uint32_t max_تارونه = (sess && sess->n_تارونه > 0) ? (uint32_t) sess->n_تارونه : 8;
    ثابت uint32_t n_تارونه   = (ټول_لیکې < max_تارونه) ? ټول_لیکې : max_تارونه;
    ثابت uint32_t vtcm_اندازه   = (sess && sess->vtcm_اندازه > 0) ? sess->vtcm_اندازه : 0;

    إذا (is_patchembed) {
        ثابت uint64_t src_لیکه_بایټونه_raw = (uint64_t) IC * KH * IW * حجم(عائم);
        ثابت uint64_t dst_لیکه_بایټونه_raw = (uint64_t) OW * patch_stride * dst_elem;
        إذا (src_لیکه_بایټونه_raw <= UINT32_MAX && dst_لیکه_بایټونه_raw <= UINT32_MAX) {
            ثابت uint32_t src_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) src_لیکه_بایټونه_raw, 256);
            ثابت uint32_t dst_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) dst_لیکه_بایټونه_raw, 256);

            بنية htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_لیکه_بایټونه, dst_لیکه_بایټونه, n_تارونه);

            إذا (vtcm_اندازه == 0 || layout.ټول_بایټونه <= vtcm_اندازه) {
                kparams->kernel_type          = HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_تارونه            = (uint8_t) n_تارونه;
                kparams->vtcm_اندازه            = layout.ټول_بایټونه;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_بایټونه_فی_thread = layout.src_بایټونه_فی_thread;
                kparams->dst_بایټونه_فی_thread = layout.dst_بایټونه_فی_thread;
                kparams->src_لیکه_بایټونه        = src_لیکه_بایټونه;
                kparams->dst_لیکه_بایټونه        = dst_لیکه_بایټونه;
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
                kparams->ټول_لیکې           = ټول_لیکې;
                أرجع ریښتیا;
            }
        }
    }

    لكل (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        إذا (Wb > UINT32_MAX) {
            إذا (owb == 1) اكسر;
            استمر;
        }

        ثابت uint64_t src_لیکه_بایټونه_raw = (uint64_t) IC * KH * Wb * حجم(عائم);
        ثابت uint64_t dst_لیکه_بایټونه_raw = (uint64_t) owb * patch_stride * dst_elem;
        إذا (src_لیکه_بایټونه_raw > UINT32_MAX || dst_لیکه_بایټونه_raw > UINT32_MAX) {
            إذا (owb == 1) اكسر;
            استمر;
        }

        ثابت uint32_t src_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) src_لیکه_بایټونه_raw, 256);
        ثابت uint32_t dst_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) dst_لیکه_بایټونه_raw, 256);

        بنية htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_لیکه_بایټونه, dst_لیکه_بایټونه, n_تارونه);

        إذا (vtcm_اندازه == 0 || layout.ټول_بایټونه <= vtcm_اندازه) {
            kparams->kernel_type          = HTP_IM2_ستون_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_تارونه            = (uint8_t) n_تارونه;
            kparams->vtcm_اندازه            = layout.ټول_بایټونه;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_بایټونه_فی_thread = layout.src_بایټونه_فی_thread;
            kparams->dst_بایټونه_فی_thread = layout.dst_بایټونه_فی_thread;
            kparams->src_لیکه_بایټونه        = src_لیکه_بایټونه;
            kparams->dst_لیکه_بایټونه        = dst_لیکه_بایټونه;
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
            kparams->ټول_لیکې           = ټول_لیکې;
            أرجع ریښتیا;
        }
        إذا (owb == 1) اكسر;
    }

    أرجع خطأ;
}

ساكن منطقي ggml_hexagon_precompute_im2col_3d_params(
    ثابت بنية ggml_hexagon_session * sess,
    ثابت بنية ggml_tensor * op,
    بنية htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_IM2_ستون_KERNEL_UNSUPPORTED;

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (!src0 || !src1 || !dst) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        أرجع خطأ;
    }

    إذا (src1->nb[0] != حجم(عائم) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    ثابت int32_t * params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ثابت int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ثابت int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ثابت int32_t ic = params[9];

    إذا (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        أرجع خطأ;
    }

    إذا (src1->ne[3] % ic != 0) {
        أرجع خطأ;
    }

    ثابت uint32_t IC = (uint32_t) ic;
    ثابت uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    إذا (N == 0) {
        أرجع خطأ;
    }

    uint32_t OW, OH, OD;
    إذا (!im2col_3d_calc_output_اندازه(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_اندازه(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_اندازه(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        أرجع خطأ;
    }

    ثابت uint32_t KD = (uint32_t) src0->ne[2];
    ثابت uint32_t KH = (uint32_t) src0->ne[1];
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t ID = (uint32_t) src1->ne[2];
    ثابت uint32_t IH = (uint32_t) src1->ne[1];
    ثابت uint32_t IW = (uint32_t) src1->ne[0];

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ثابت uint64_t ټول_لیکې_64   = (uint64_t) N * OD * OH;
    ثابت uint64_t dst_ne3_64      = (uint64_t) OD * N;

    إذا (patch_stride_64 > UINT32_MAX || ټول_لیکې_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        أرجع خطأ;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t ټول_لیکې   = (uint32_t) ټول_لیکې_64;
    ثابت uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    إذا (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        أرجع خطأ;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(عائم);

    ثابت منطقي is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ثابت uint32_t max_تارونه = (sess && sess->n_تارونه > 0) ? (uint32_t) sess->n_تارونه : 8;
    ثابت uint32_t n_تارونه   = (ټول_لیکې < max_تارونه) ? ټول_لیکې : max_تارونه;
    ثابت uint32_t vtcm_اندازه   = (sess && sess->vtcm_اندازه > 0) ? sess->vtcm_اندازه : 0;

    إذا (is_patchembed) {
        ثابت uint64_t src_لیکه_بایټونه_raw = (uint64_t) IC * KD * KH * IW * حجم(عائم);
        ثابت uint64_t dst_لیکه_بایټونه_raw = (uint64_t) OW * patch_stride * dst_elem;
        إذا (src_لیکه_بایټونه_raw <= UINT32_MAX && dst_لیکه_بایټونه_raw <= UINT32_MAX) {
            ثابت uint32_t src_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) src_لیکه_بایټونه_raw, 256);
            ثابت uint32_t dst_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) dst_لیکه_بایټونه_raw, 256);

            بنية htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_لیکه_بایټونه, dst_لیکه_بایټونه, n_تارونه);

            إذا (vtcm_اندازه == 0 || layout.ټول_بایټونه <= vtcm_اندازه) {
                kparams->kernel_type          = HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_تارونه            = (uint8_t) n_تارونه;
                kparams->vtcm_اندازه            = layout.ټول_بایټونه;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_بایټونه_فی_thread = layout.src_بایټونه_فی_thread;
                kparams->dst_بایټونه_فی_thread = layout.dst_بایټونه_فی_thread;
                kparams->src_لیکه_بایټونه        = src_لیکه_بایټونه;
                kparams->dst_لیکه_بایټونه        = dst_لیکه_بایټونه;
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
                kparams->ټول_لیکې           = ټول_لیکې;
                أرجع ریښتیا;
            }
        }
    }

    لكل (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        إذا (Wb > UINT32_MAX) {
            إذا (owb == 1) اكسر;
            استمر;
        }

        ثابت uint64_t src_لیکه_بایټونه_raw = (uint64_t) IC * KD * KH * Wb * حجم(عائم);
        ثابت uint64_t dst_لیکه_بایټونه_raw = (uint64_t) owb * patch_stride * dst_elem;
        إذا (src_لیکه_بایټونه_raw > UINT32_MAX || dst_لیکه_بایټونه_raw > UINT32_MAX) {
            إذا (owb == 1) اكسر;
            استمر;
        }

        ثابت uint32_t src_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) src_لیکه_بایټونه_raw, 256);
        ثابت uint32_t dst_لیکه_بایټونه = hex_پړاو_پورته((uint32_t) dst_لیکه_بایټونه_raw, 256);

        بنية htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_لیکه_بایټونه, dst_لیکه_بایټونه, n_تارونه);

        إذا (vtcm_اندازه == 0 || layout.ټول_بایټونه <= vtcm_اندازه) {
            kparams->kernel_type          = HTP_IM2_ستون_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_تارونه            = (uint8_t) n_تارونه;
            kparams->vtcm_اندازه            = layout.ټول_بایټونه;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_بایټونه_فی_thread = layout.src_بایټونه_فی_thread;
            kparams->dst_بایټونه_فی_thread = layout.dst_بایټونه_فی_thread;
            kparams->src_لیکه_بایټونه        = src_لیکه_بایټونه;
            kparams->dst_لیکه_بایټونه        = dst_لیکه_بایټونه;
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
            kparams->ټول_لیکې           = ټول_لیکې;
            أرجع ریښتیا;
        }
        إذا (owb == 1) اكسر;
    }

    أرجع خطأ;
}

ساكن منطقي ggml_hexagon_supported_im2col(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    بنية htp_im2col_kernel_params kparams;
    أرجع ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ساكن منطقي ggml_hexagon_supported_im2col_3d(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    بنية htp_im2col_kernel_params kparams;
    أرجع ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ساكن منطقي ggml_hexagon_supported_pad(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    ثابت int32_t lp0 = ((ثابت int32_t *) op->op_params)[0];
    ثابت int32_t rp0 = ((ثابت int32_t *) op->op_params)[1];
    ثابت int32_t circular = ((ثابت int32_t *) op->op_params)[8];

    إذا (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_cumsum(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_diag(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    // diag only supports F32 currently
    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    // Input must have ne[1] == 1 (vector input)
    إذا (src0->ne[1] != 1) {
        أرجع خطأ;
    }

    // Output must be square in first two dimensions
    إذا (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_solve_tri(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0]; // A
    ثابت بنية ggml_tensor * src1 = op->src[1]; // B
    ثابت بنية ggml_tensor * dst  = op;         // X

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (src0->ne[0] != src0->ne[1]) {
        أرجع خطأ;
    }

    إذا (src0->ne[1] != src1->ne[1]) {
        أرجع خطأ;
    }

    إذا (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        أرجع خطأ;
    }

    إذا (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        أرجع خطأ;
    }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_tri(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32) { أرجع خطأ; }
    إذا (dst->type  != GGML_TYPE_F32) { أرجع خطأ; }
    إذا (!ggml_are_same_shape(src0, dst)) { أرجع خطأ; }
    إذا (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { أرجع خطأ; }

    أرجع ریښتیا;

    GGML_ناکارول_شوی(sess);
}

ساكن ثابت حرف * ggml_backend_hexagon_name(ggml_backend_t backend) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    أرجع sess->c_name();
}

ساكن فراغ ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    ړنګول backend;
}

ساكن htp_op_کوډ op_remap_منزل_htp(ثابت ggml_tensor * t) {
    بدل (t->op) {
        حالة GGML_OP_FLASH_ATTN_EXT:  أرجع HTP_OP_FLASH_ATTN_EXT;
        حالة GGML_OP_MUL_MAT:
            إذا (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                أرجع HTP_OP_HADAMARD;
            }
            أرجع HTP_OP_MUL_MAT;
        حالة GGML_OP_MUL_MAT_ID:      أرجع HTP_OP_MUL_MAT_ID;
        حالة GGML_OP_MUL:             أرجع HTP_OP_MUL;
        حالة GGML_OP_جمع_کړه:             أرجع HTP_OP_جمع_کړه;
        حالة GGML_OP_جمع_کړه_ID:          أرجع HTP_OP_جمع_کړه_ID;
        حالة GGML_OP_SUB:             أرجع HTP_OP_SUB;
        حالة GGML_OP_DIV:             أرجع HTP_OP_DIV;
        حالة GGML_OP_DUP:
        حالة GGML_OP_CPY:             أرجع HTP_OP_CPY;
        حالة GGML_OP_CONT:            أرجع HTP_OP_CPY;
        حالة GGML_OP_GET_لیکې:        أرجع HTP_OP_GET_لیکې;
        حالة GGML_OP_سېټ_لیکې:        أرجع HTP_OP_سېټ_لیکې;
        حالة GGML_OP_مجموعه:             أرجع HTP_OP_مجموعه;
        حالة GGML_OP_مجموعه_لیکې:        أرجع HTP_OP_مجموعه_لیکې;
        حالة GGML_OP_POOL_2D:         أرجع HTP_OP_POOL_2D;
        حالة GGML_OP_POOL_1D:         أرجع HTP_OP_POOL_1D;
        حالة GGML_OP_ARGSORT:         أرجع HTP_OP_ARGSORT;
        حالة GGML_OP_سر_K:           أرجع HTP_OP_سر_K;
        حالة GGML_OP_ARGMAX:          أرجع HTP_OP_ARGMAX;
        حالة GGML_OP_NORM:            أرجع HTP_OP_NORM;
        حالة GGML_OP_L2_NORM:         أرجع HTP_OP_L2_NORM;
        حالة GGML_OP_RMS_NORM:        أرجع HTP_OP_RMS_NORM;
        حالة GGML_OP_CONCAT:          أرجع HTP_OP_CONCAT;
        حالة GGML_OP_SCALE:           أرجع HTP_OP_SCALE;
        حالة GGML_OP_CLAMP:           أرجع HTP_OP_CLAMP;
        حالة GGML_OP_LEAKY_RELU:      أرجع HTP_OP_LEAKY_RELU;
        حالة GGML_OP_SQR:             أرجع HTP_OP_SQR;
        حالة GGML_OP_SQRT:            أرجع HTP_OP_SQRT;
        حالة GGML_OP_LOG:             أرجع HTP_OP_UNARY_LOG;
        حالة GGML_OP_SOFT_MAX:        أرجع HTP_OP_SOFTMAX;
        حالة GGML_OP_SSM_CONV:        أرجع HTP_OP_SSM_CONV;
        حالة GGML_OP_GATED_DELTA_NET: أرجع HTP_OP_GATED_DELTA_NET;
        حالة GGML_OP_ROPE:            أرجع HTP_OP_ROPE;
        حالة GGML_OP_REPEAT:          أرجع HTP_OP_REPEAT;
        حالة GGML_OP_CUMSUM:          أرجع HTP_OP_CUMSUM;
        حالة GGML_OP_FILL:            أرجع HTP_OP_FILL;
        حالة GGML_OP_DIAG:            أرجع HTP_OP_DIAG;
        حالة GGML_OP_SOLVE_TRI:       أرجع HTP_OP_SOLVE_TRI;
        حالة GGML_OP_TRI:             أرجع HTP_OP_TRI;
        حالة GGML_OP_PAD:             أرجع HTP_OP_PAD;
        حالة GGML_OP_IM2_ستون:          أرجع HTP_OP_IM2_ستون;
        حالة GGML_OP_IM2_ستون_3_D:       أرجع HTP_OP_IM2_ستون_3_D;
        حالة GGML_OP_ROLL:            أرجع HTP_OP_ROLL;

        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(t)) {
                حالة GGML_UNARY_OP_SILU:       أرجع HTP_OP_UNARY_SILU;
                حالة GGML_UNARY_OP_GELU:       أرجع HTP_OP_UNARY_GELU;
                حالة GGML_UNARY_OP_GELU_چټک: أرجع HTP_OP_UNARY_GELU_چټک;
                حالة GGML_UNARY_OP_GELU_ERF:   أرجع HTP_OP_UNARY_GELU_ERF;
                حالة GGML_UNARY_OP_SIGMOID:    أرجع HTP_OP_UNARY_SIGMOID;
                حالة GGML_UNARY_OP_NEG:        أرجع HTP_OP_UNARY_NEG;
                حالة GGML_UNARY_OP_EXP:        أرجع HTP_OP_UNARY_EXP;
                حالة GGML_UNARY_OP_SOFTPLUS:   أرجع HTP_OP_UNARY_SOFTPLUS;
                حالة GGML_UNARY_OP_TANH:       أرجع HTP_OP_UNARY_TANH;
                حالة GGML_UNARY_OP_ABS:        أرجع HTP_OP_UNARY_ABS;
                حالة GGML_UNARY_OP_RELU:       أرجع HTP_OP_UNARY_RELU;
                حالة GGML_UNARY_OP_ګام:       أرجع HTP_OP_UNARY_ګام;
            افتراضي:
                اكسر;
            }
            اكسر;

        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(t)) {
                حالة GGML_GLU_OP_SWIGLU:       أرجع HTP_OP_GLU_SWIGLU;
                حالة GGML_GLU_OP_SWIGLU_OAI:   أرجع HTP_OP_GLU_SWIGLU_OAI;
                حالة GGML_GLU_OP_SWIGLU_CLAMP: أرجع HTP_OP_GLU_SWIGLU_CLAMP;
                حالة GGML_GLU_OP_GEGLU:        أرجع HTP_OP_GLU_GEGLU;
                حالة GGML_GLU_OP_GEGLU_چټک:  أرجع HTP_OP_GLU_GEGLU_چټک;
                حالة GGML_GLU_OP_GEGLU_ERF:    أرجع HTP_OP_GLU_GEGLU_ERF;
                افتراضي: اكسر;
            }
            اكسر;

        افتراضي:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    أرجع HTP_OP_INVALID;
}

ساكن مضمن منطقي op_is_compute(ggml_tensor *غوټه)
{
    أرجع !ggml_op_is_تشه(غوټه->op) && !ggml_is_تشه(غوټه) && (غوټه->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ساكن منطقي mm_is_hmx_eligible(ثابت ggml_tensor * t) {
    إذا (opt_nhmx == 0) { أرجع خطأ; }

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];

    إذا (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ثابت ggml_tensor src1_c = ggml_hexagon_tensor_collapse_لیکې(src1);
        أرجع ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, خطأ, خطأ);
    }

    ثابت منطقي is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقي is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    أرجع ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ساكن منطقي is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت بنية htp_mm_kernel_params * kparams) {
    إذا (kparams->n_hmx) {
        أرجع kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    إذا (!ggml_hexagon_tensor_is_tiled(src0)) {
        أرجع خطأ;
    }

    أرجع kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه;
}

ساكن منطقي is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت بنية htp_mm_kernel_params * kparams) {
    إذا (kparams->n_hmx) {
        أرجع kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    إذا (!ggml_hexagon_tensor_is_tiled(src0)) {
        أرجع خطأ;
    }

    أرجع kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک;
}

ساكن منطقي is_mergeable_mul_mat(ثابت ggml_tensor * t) {
    إذا (t->op != GGML_OP_MUL_MAT) أرجع خطأ;
    إذا (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) أرجع خطأ;

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];
    إذا (src0->ne[2] != 1 || src0->ne[3] != 1) أرجع خطأ;

    إذا (mm_is_hmx_eligible(t)) {
        أرجع htp_is_hmx_وزن_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    إذا (src1->type != GGML_TYPE_F32) أرجع خطأ;

    أرجع ggml_hexagon_tensor_is_tiled(src0);
}

ساكن منطقي is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    إذا (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        أرجع خطأ;
    }
    إذا (n1->src[1] != n2->src[1]) {
        أرجع خطأ;
    }
    إذا (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        أرجع خطأ;
    }
    إذا (n1->src[0]->type != n2->src[0]->type) {
        أرجع خطأ;
    }
    إذا (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        أرجع خطأ;
    }
    إذا (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        أرجع خطأ;
    }
    أرجع ریښتیا;
}

ساكن منطقي is_mergeable_mul_mat_id(ثابت ggml_tensor * t) {
    إذا (t->op != GGML_OP_MUL_MAT_ID) أرجع خطأ;

    ثابت ggml_tensor * src0 = t->src[0];
    إذا (mm_is_hmx_eligible(t)) {
        أرجع htp_is_hmx_وزن_type(src0->type);
    }

    أرجع ggml_hexagon_tensor_is_tiled(src0);
}

ساكن منطقي is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    إذا (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        أرجع خطأ;
    }
    إذا (n1->src[1] != n2->src[1]) {
        أرجع خطأ;
    }
    إذا (n1->src[2] != n2->src[2]) {
        أرجع خطأ;
    }
    إذا (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        أرجع خطأ;
    }
    إذا (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        أرجع خطأ;
    }
    إذا (n1->src[0]->type != n2->src[0]->type) {
        أرجع خطأ;
    }
    إذا (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        أرجع خطأ;
    }
    إذا (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        أرجع خطأ;
    }
    أرجع ریښتیا;
}

ساكن ggml_حالت ggml_backend_hexagon_ګراف_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);

    إذا (sess->وروستی_error > HTP_حالت_OK) {
        أرجع GGML_حالت_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ثابت std::vector<htp_opnode> * nodes_ptr = تش_اشاره;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    منطقي cache_موندنه = (graph->uid != 0 && sess->cached_uid == graph->uid);
    إذا (cache_موندنه) {
        nodes_ptr = &sess->cached_nodes;
    } وإلا {
        // Tag fusable tensors in graph
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            خپلکاری * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            إذا (!extra) استمر;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            إذا (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } وإلا إذا ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                إذا ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_جمع_کړه && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_جمع_کړه })) ||
                    ggml_غوټه_has_n_کاروي(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            إذا (!op_is_compute(n)) {
                استمر;
            }

            htp_opnode غوټه(HTP_OP_INVALID, n);
            غوټه.opcode = op_remap_منزل_htp(n);
            إذا (غوټه.opcode == HTP_OP_MUL_MAT || غوټه.opcode == HTP_OP_MUL_MAT_ID || غوټه.opcode == HTP_OP_GET_لیکې) {
                ggml_tensor * src0 = غوټه.غوټه->src[0];
                إذا (ggml_hexagon_needs_repack(src0)) {
                    إذا (opt_verbose) {
                        حرف dims_buf[64];
                        حرف strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, حجم(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, حجم(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    حافظه_نقل(raw.معلومات(), src0->معلومات, raw.size());
                    htp_repack_tensor_tiled(src0, raw.معلومات(), raw.size());
                    خپلکاری * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            إذا (غوټه.opcode == HTP_OP_MUL_MAT || غوټه.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    غوټه.غوټه->src[0], غوټه.غوټه->src[1], غوټه.غوټه,
                    (بنية htp_mm_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_MUL || غوټه.opcode == HTP_OP_جمع_کړه || غوټه.opcode == HTP_OP_جمع_کړه_ID || غوټه.opcode == HTP_OP_SUB || غوټه.opcode == HTP_OP_DIV) {
                ثابت ggml_tensor * src1 = غوټه.غوټه->src[1];
                ثابت ggml_tensor * src2 = (غوټه.opcode == HTP_OP_جمع_کړه_ID) ? غوټه.غوټه->src[2] : تش_اشاره;
                GGML_ASSERT(ggml_hexagon_precompute_دوګونی_params(sess,
                    غوټه.opcode, غوټه.غوټه->src[0], src1, غوټه.غوټه,
                    (بنية htp_دوګونی_kernel_params *) غوټه.kernel_params, src2));
            } وإلا إذا (غوټه.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    غوټه.غوټه,
                    (بنية htp_fa_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (htp_op_is_unary(غوټه.opcode)) {
                خپلکاری inputs = غوټه.get_inputs();
                ثابت بنية ggml_tensor * src0 = inputs[0];
                ثابت بنية ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : تش_اشاره;
                ggml_hexagon_precompute_unary_params(sess,
                    غوټه.opcode, src0, src1, غوټه.dst(),
                    (بنية htp_unary_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_GET_لیکې) {
                ggml_hexagon_precompute_get_لیکې_params(sess,
                    غوټه.غوټه->src[0], غوټه.غوټه->src[1], غوټه.dst(),
                    (بنية htp_get_لیکې_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_سېټ_لیکې) {
                ggml_hexagon_precompute_سېټ_لیکې_params(sess,
                    غوټه.غوټه->src[0], غوټه.غوټه->src[1], غوټه.dst(),
                    (بنية htp_سېټ_لیکې_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    غوټه.غوټه,
                    (بنية htp_rope_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    غوټه.غوټه->src[0], غوټه.غوټه->src[1], غوټه.dst(),
                    (بنية htp_ssm_conv_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    غوټه.غوټه,
                    (بنية htp_softmax_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    غوټه.غوټه,
                    (بنية htp_gdn_kernel_params *)غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_ARGSORT || غوټه.opcode == HTP_OP_سر_K) {
                ggml_hexagon_precompute_ترتیبول_params(sess,
                    غوټه.غوټه,
                    غوټه.opcode == HTP_OP_سر_K,
                    (بنية htp_ترتیبول_kernel_params *) غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_POOL_2D || غوټه.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, غوټه.غوټه->src[0], غوټه.dst(),
                    (بنية htp_pool_2d_kernel_params *)غوټه.kernel_params,
                    غوټه.opcode == HTP_OP_POOL_1D);
            } وإلا إذا (غوټه.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    غوټه.غوټه,
                    (بنية htp_concat_kernel_params *) غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_CPY || غوټه.opcode == HTP_OP_CPY_FENCE) {
                ثابت منطقي ok = ggml_hexagon_precompute_cpy_params(sess,
                    غوټه.غوټه,
                    (بنية htp_copy_kernel_params *) غوټه.kernel_params
                );
                ثابت خپلکاری * kparams = (ثابت بنية htp_copy_kernel_params *) غوټه.kernel_params;
                إذا (ok && غوټه.opcode == HTP_OP_CPY && kparams->ټول_elems == 0) {
                    استمر;
                }
            } وإلا إذا (غوټه.opcode == HTP_OP_IM2_ستون) {
                ggml_hexagon_precompute_im2col_params(sess,
                    غوټه.غوټه,
                    (بنية htp_im2col_kernel_params *) غوټه.kernel_params
                );
            } وإلا إذا (غوټه.opcode == HTP_OP_IM2_ستون_3_D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    غوټه.غوټه,
                    (بنية htp_im2col_kernel_params *) غوټه.kernel_params
                );
            }
            computed_nodes.push_back(std::move(غوټه));
        }

        إذا (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } وإلا {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    لكل (ثابت خپلکاری & غوټه : *nodes_ptr) {
        sess->کتار_ته_ورزیاتول_op(غوټه);
    }

    إذا (sess->وروستی_error > HTP_حالت_OK) {
        أرجع GGML_حالت_FAILED;
    }

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    إذا (sess->وروستی_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), حالت_منزل_str(sess->وروستی_error));
    }
}

تعداد ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

بنية ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

بنية ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    فراغ reset() {
        ranges.clear();
    }

    فراغ جمع_کړه_2(ثابت ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    منطقي check(ثابت ggml_hexagon_mem_range & mr) ثابت {
        لكل (ثابت خپلکاری & cmp : ranges) {
            إذا (mr.pb != cmp.pb) {
                استمر;
            }
            إذا (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                استمر;
            }
            إذا (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                أرجع خطأ;
            }
        }
        أرجع ریښتیا;
    }
};

ساكن ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ثابت ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ثابت ggml_tensor * بنسټ = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    إذا (tensor->بفر) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->بفر,
            /*.p0 =*/ (uint64_t) tensor->معلومات,
            /*.p1 =*/ (uint64_t) tensor->معلومات + ggml_backend_buft_get_alloc_اندازه(tensor->بفر->buft, tensor),
            /*.pt =*/ pt,
        };
    } وإلا {
        mr = {
            /*.pb =*/ (uint64_t) بنسټ,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    أرجع mr;
}

ساكن فراغ ggml_hexagon_mem_ranges_جمع_کړه_غوټه(ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & غوټه) {
    إذا (غوټه.is_تشه()) أرجع;

    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (غوټه.غوټه->src[i]) {
            mrs.جمع_کړه_2(ggml_hexagon_mem_range_from_tensor(غوټه.غوټه->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    لكل (ثابت خپلکاری * fused : غوټه.fused) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (fused->src[i]) {
                mrs.جمع_کړه_2(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.جمع_کړه_2(ggml_hexagon_mem_range_from_tensor(غوټه.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ساكن منطقي ggml_hexagon_mem_ranges_check_غوټه(ثابت ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & غوټه) {
    إذا (غوټه.is_تشه()) أرجع ریښتیا;

    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (غوټه.غوټه->src[i]) {
            إذا (!mrs.check(ggml_hexagon_mem_range_from_tensor(غوټه.غوټه->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                أرجع خطأ;
            }
        }
    }
    لكل (ثابت خپلکاری * fused : غوټه.fused) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (fused->src[i]) {
                إذا (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    أرجع خطأ;
                }
            }
        }
    }
    أرجع mrs.check(ggml_hexagon_mem_range_from_tensor(غوټه.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ساكن std::vector<صحيح> ggml_hexagon_ګراف_optimize_reorder(ثابت std::vector<htp_opnode> & nodes) {
    ثابت صحيح n = nodes.size();

    std::vector<صحيح> res;
    res.reserve(n);

    std::vector<منطقي> کارول_شوی(n, خطأ);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    لكل (صحيح i0 = 0; i0 < n; i0++) {
        إذا (کارول_شوی[i0]) {
            استمر;
        }

        ثابت خپلکاری & node0 = nodes[i0];

        إذا (!node0.stackable()) {
            res.push_back(i0);
            کارول_شوی[i0] = ریښتیا;
            استمر;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr صحيح N_FORWARD = 16;

        std::vector<صحيح> ډېری;
        ډېری.push_back(i0);

        mrs.reset();

        لكل (صحيح i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            إذا (کارول_شوی[i1]) {
                استمر;
            }

            ثابت خپلکاری & node1 = nodes[i1];

            إذا (node1.stackable() && node1.same_ننوت(node0) && ggml_hexagon_mem_ranges_check_غوټه(mrs, node1)) {
                ډېری.push_back(i1);
            } وإلا {
                ggml_hexagon_mem_ranges_جمع_کړه_غوټه(mrs, node1);
            }
        }

        لكل (صحيح idx : ډېری) {
            res.push_back(idx);
            کارول_شوی[idx] = ریښتیا;
        }
    }

    أرجع res;
}

ساكن فراغ ggml_backend_hexagon_ګراف_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_ګراف_optimize_params * params) {
    GGML_ناکارول_شوی(params);

    ثابت صحيح n = gf->n_nodes;

    constexpr صحيح MAX_FUSE = 16;

    تعداد ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    لكل (صحيح i = 0; i < n; i++) {
        htp_opnode غوټه(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        إذا (غوټه.op() == GGML_OP_جمع_کړه ||
            غوټه.op() == GGML_OP_NORM ||
            غوټه.op() == GGML_OP_RMS_NORM) {
            ops[0] = غوټه.op();

            صحيح f = i + 1;
            طالما (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                إذا (gf->nodes[f]->op != GGML_OP_جمع_کړه &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    اكسر;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            لكل (; f > 1; f--) {
                إذا (ggml_can_fuse(gf, i, ops, f)) {
                    اكسر;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            لكل (صحيح k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                غوټه.جمع_کړه_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(غوټه));
    }

    ثابت خپلکاری ترتیب = ggml_hexagon_ګراف_optimize_reorder(nodes);

    // unfuse
    {
        صحيح j = 0;
        لكل (ثابت خپلکاری i : ترتیب) {
            ثابت خپلکاری & غوټه = nodes[i];

            gf->nodes[j++] = غوټه.غوټه;

            لكل (خپلکاری * fused : غوټه.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_ناکارول_شوی(backend);
}

ساكن uint64_t ggml_hexagon_session_کلی(ثابت ggml_hexagon_session * sess) {
    أرجع ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ساكن منطقي ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    خپلکاری sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خپلکاری sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    خپلکاری sbuf_dst = sess_dst->mmap_tensor(dst);

    إذا (!sess_src->clone_بفر(sbuf_dst)) { أرجع خطأ; }

    ثابت uint64_t src_کلی = ggml_hexagon_session_کلی(sess_src);
    خپلکاری & fence_خونه = sess_dst->cpy_fence_slots[src_کلی];
    إذا (!fence_خونه) {
        fence_خونه = (متطاير uint32_t *) sess_dst->alloc_fence(1);
    }

    إذا (!sess_src->clone_بفر(sess_dst->fence_buf)) { أرجع خطأ; }

    إذا (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ساكن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.بفر = &sess_dst->fence_buf->backend_بفر;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.معلومات   = (فراغ *) fence_خونه;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = حجم(int32_t);
    fence_tensor.nb[1]  = حجم(int32_t);
    fence_tensor.nb[2]  = حجم(int32_t);
    fence_tensor.nb[3]  = حجم(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->کتار_ته_ورزیاتول_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->کتار_ته_ورزیاتول_fence(&fence_tensor, fence_seq, /* wait = */ ریښتیا);

    sess_dst->جمع_کړه_peer(sess_src);

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    خپلکاری sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خپلکاری sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    خپلکاری sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    إذا (!sess_dst->clone_بفر(sbuf_src)) { أرجع خطأ; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->کتار_ته_ورزیاتول_cpy(src, dst);
    sess_dst->جمع_کړه_peer(sess_src);

    أرجع ریښتیا;
}

ساكن منطقي ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        أرجع خطأ;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    إذا (!dst->extra) {
        ggml_backend_بفر_init_tensor(dst->بفر, dst);
    }

    خپلکاری * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ثابت خپلکاری * src_extra = static_cast<ثابت ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    خپلکاری sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خپلکاری sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    إذا (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->کتار_ته_ورزیاتول_cpy(src, dst);
        أرجع ریښتیا;
    }

    إذا (sess_src->phys_idx != sess_dst->phys_idx)
        أرجع ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    أرجع ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ساكن ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خپلکاری sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = جديد ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_خونه = (متطاير uint32_t *) sess->alloc_fence(1);

    ساكن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.بفر = &sess->fence_buf->backend_بفر;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.معلومات   = (فراغ *) hex_event->fence_خونه;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = حجم(int32_t);
    hex_event->fence_tensor.nb[1]  = حجم(int32_t);
    hex_event->fence_tensor.nb[2]  = حجم(int32_t);
    hex_event->fence_tensor.nb[3]  = حجم(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (فراغ *)hex_event, (فراغ *)hex_event->fence_خونه);

    أرجع جديد ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ساكن فراغ ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    إذا (hex_event->seq == 0) {
        أرجع;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (فراغ *)hex_event, hex_event->seq, (فراغ *)hex_event->fence_خونه);

    خپلکاری * fence = reinterpret_cast<ثابت متطاير std::atomic<uint32_t> *>(hex_event->fence_خونه);

    إذا ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    طالما (ریښتیا) {
        إذا ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t حالت = fence[1].load(std::memory_order_acquire);
            إذا (حالت > HTP_حالت_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), حالت_منزل_str(حالت));
            }
            اكسر;
        }
        std::this_thread::yield();
    }
}

ساكن فراغ ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    خپلکاری * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (فراغ *)hex_event);
    hex_event->fence_sess->free_fence((فراغ *) hex_event->fence_خونه, 1);
    ړنګول hex_event;
    ړنګول event;
}

ساكن فراغ ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    خپلکاری * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ساكن فراغ ggml_backend_hexagon_event_ریکارډ(ggml_backend_t backend, ggml_backend_event_t event) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    خپلکاری hex_event = static_cast<ggml_hexagon_event *>(event->context);

    إذا (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->کتار_ته_ورزیاتول_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ خطأ);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (فراغ *)hex_event, hex_event->seq, (فراغ *)hex_event->fence_خونه);
}

ساكن فراغ ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    خپلکاری hex_event = static_cast<ggml_hexagon_event *>(event->context);

    إذا (hex_event->seq == 0) {
        أرجع;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (فراغ *)hex_event, hex_event->seq, (فراغ *)hex_event->fence_خونه);

    // same physical NPU runs sequentially in FIFO order
    إذا (sess->phys_idx == hex_event->sess->phys_idx) {
        إذا (sess != hex_event->sess) {
            sess->جمع_کړه_peer(hex_event->sess);
        }
        أرجع;
    }

    sess->clone_بفر(hex_event->fence_sess->fence_buf);
    sess->جمع_کړه_peer(hex_event->sess);
    sess->کتار_ته_ورزیاتول_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ریښتیا);
}

ساكن فراغ ggml_backend_hexagon_سېټ_tensor_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, tensor->بفر ? (صحيح) tensor->بفر->usage : -1);
    ggml_backend_tensor_سېټ(tensor, معلومات, دننه_واټن, size);
}

ساكن فراغ ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, tensor->بفر ? (صحيح) tensor->بفر->usage : -1);
    sess->flush_sync();
    إذا (sess->وروستی_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), حالت_منزل_str(sess->وروستی_error));
    }
    ggml_backend_tensor_get(tensor, معلومات, دننه_واټن, size);
}

ساكن فراغ ggml_backend_hexagon_سېټ_tensor_2d_async(ggml_backend_t backend,
                                                     بنية ggml_tensor * tensor,
                                                     ثابت فراغ * معلومات,
                                                     size_t دننه_واټن,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_معلومات) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات, tensor->بفر ? (صحيح) tensor->بفر->usage : -1);
    ggml_backend_tensor_سېټ_2d(tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

ساكن فراغ ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ثابت بنية ggml_tensor * tensor,
                                                     فراغ * معلومات,
                                                     size_t دننه_واټن,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_معلومات) {
    خپلکاری sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات, tensor->بفر ? (صحيح) tensor->بفر->usage : -1);
    sess->flush_sync();
    إذا (sess->وروستی_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), حالت_منزل_str(sess->وروستی_error));
    }
    ggml_backend_tensor_get_2d(tensor, معلومات, دننه_واټن, size, n_copies, stride_tensor, stride_معلومات);
}

ساكن بنية ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_سېټ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_سېټ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_hexagon_ګراف_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_ریکارډ,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_ګراف_optimize,
};

ساكن ggml_guid_t ggml_backend_hexagon_guid() {
    ساكن ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    أرجع &guid;
}

منطقي ggml_backend_is_hexagon(ggml_backend_t backend) {
    أرجع backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ساكن ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خپلکاری sess    = dev_ctx->session();

    أرجع جديد ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_ناکارول_شوی(params);
}

ساكن ثابت حرف * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    أرجع dev_ctx->c_name();

    GGML_ناکارول_شوی(dev);
}

ساكن ثابت حرف * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    أرجع "Hexagon";
    GGML_ناکارول_شوی(dev);
}

ساكن فراغ ggml_backend_hexagon_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * ټول) {
    *حرر  = opt_mbuf_ټول;
    *ټول = opt_mbuf_ټول;

    GGML_ناکارول_شوی(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ناکارول_شوی(dev);
}

ساكن فراغ ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_ټول);
    props->caps = {
        /* .async                 = */ ریښتیا,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ ریښتیا,
        /* .mmap_support          = */ خطأ,
    };
}

ساكن ggml_backend_بفر_type_t ggml_backend_hexagon_device_get_بفر_type(ggml_backend_dev_t dev) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    أرجع &dev_ctx->بفر_type;
}

ساكن ggml_backend_بفر_type_t ggml_backend_hexagon_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    إذا (!opt_hostbuf) {
        أرجع عدم;
    }
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    أرجع &dev_ctx->host_بفر_type;
}

ساكن منطقي ggml_hexagon_supported_cpy(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    بنية htp_copy_kernel_params kparams;
    أرجع ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ساكن منطقي ggml_hexagon_supported_cont(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    بنية htp_copy_kernel_params kparams;
    أرجع ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ساكن منطقي ggml_hexagon_supported_repeat(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    GGML_ناکارول_شوی(sess);
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    // Support f32 and f16
    إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) أرجع خطأ;

    // src and dst must be the same type
    إذا (src0->type != dst->type) أرجع خطأ;

    // dst dims must be multiples of src dims
    إذا (dst->ne[0] % src0->ne[0] != 0) أرجع خطأ;
    إذا (dst->ne[1] % src0->ne[1] != 0) أرجع خطأ;
    إذا (dst->ne[2] % src0->ne[2] != 0) أرجع خطأ;
    إذا (dst->ne[3] % src0->ne[3] != 0) أرجع خطأ;

    // require contiguous tensors (no transposition)
    إذا (ggml_is_transposed(src0) || ggml_is_transposed(dst)) أرجع خطأ;

    أرجع ریښتیا;
}

ساكن منطقي ggml_hexagon_supported_concat(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    بنية htp_concat_kernel_params kparams;
    أرجع ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ساكن منطقي ggml_hexagon_supported_fill(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * dst = op;

    إذا (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
    GGML_ناکارول_شوی(sess);
}

ساكن منطقي ggml_hexagon_supported_roll(ثابت بنية ggml_hexagon_session * sess, ثابت بنية ggml_tensor * op) {
    GGML_ناکارول_شوی(sess);

    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * dst  = op;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_are_same_shape(src0, dst)) {
        أرجع خطأ;
    }

    إذا (src0->nb[0] != ggml_type_اندازه(src0->type) || dst->nb[0] != ggml_type_اندازه(dst->type)) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ساكن منطقي ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خپلکاری sess    = dev_ctx->session();

    // reject ops that match the filter
    إذا (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        أرجع خطأ;
    }

    منطقي supp = خطأ;
    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلول:
        حالة GGML_OP_TRANSPOSE:
            supp = ریښتیا;
            اكسر;

        حالة GGML_OP_MUL:
        حالة GGML_OP_جمع_کړه:
        حالة GGML_OP_SUB:
        حالة GGML_OP_DIV:
            supp = ggml_hexagon_supported_دوګونی(sess, op);
            اكسر;

        حالة GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            اكسر;

        حالة GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            اكسر;

        حالة GGML_OP_جمع_کړه_ID:
            supp = ggml_hexagon_supported_جمع_کړه_id(sess, op);
            اكسر;

        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            اكسر;

        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            اكسر;

        حالة GGML_OP_مجموعه:
            supp = ggml_hexagon_supported_مجموعه(sess, op);
            اكسر;

        حالة GGML_OP_مجموعه_لیکې:
            supp = ggml_hexagon_supported_مجموعه_لیکې(sess, op);
            اكسر;

        حالة GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            اكسر;

        حالة GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            اكسر;

        حالة GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            اكسر;

        حالة GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            اكسر;

        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(op)) {
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_SOFTPLUS:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_GELU_چټک:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_ګام:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    اكسر;
                افتراضي:
                    supp = خطأ;
                    اكسر;
            }
            اكسر;

        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_SWIGLU_OAI:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                حالة GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_چټک:
                حالة GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    اكسر;
                افتراضي:
                    supp = خطأ;
                    اكسر;
            }
            اكسر;

        حالة GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            اكسر;

        حالة GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            اكسر;

        حالة GGML_OP_سېټ_لیکې:
            supp = ggml_hexagon_supported_سېټ_لیکې(sess, op);
            اكسر;

        حالة GGML_OP_GET_لیکې:
            supp = ggml_hexagon_supported_get_لیکې(sess, op);
            اكسر;

        حالة GGML_OP_DUP:
        حالة GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            اكسر;

        حالة GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            اكسر;

        حالة GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            اكسر;

        حالة GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            اكسر;

        حالة GGML_OP_سر_K:
            supp = ggml_hexagon_supported_سر_k(sess, op);
            اكسر;

        حالة GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            اكسر;

        حالة GGML_OP_IM2_ستون:
            supp = ggml_hexagon_supported_im2col(sess, op);
            اكسر;

        حالة GGML_OP_IM2_ستون_3_D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            اكسر;

        حالة GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            اكسر;

        حالة GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            اكسر;

        حالة GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            اكسر;

        حالة GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            اكسر;

        حالة GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            اكسر;

        حالة GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            اكسر;

        حالة GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            اكسر;

        حالة GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            اكسر;

        حالة GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            اكسر;

        افتراضي:
            اكسر;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    أرجع supp;
}

ساكن منطقي ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    منطقي supp = (buft == &dev_ctx->host_بفر_type) || (buft == &dev_ctx->بفر_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    أرجع supp;
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_حافظه,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ عدم,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ عدم,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    لكل (size_t i = 0; i < opt_ndev; i++) {
        ثابت خپلکاری & cfg = opt_device_configs[i];
        إذا (cfg.mdev_ډله.تشه()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } وإلا {
            std::string peers_str;
            لكل (ثابت خپلکاری & p : cfg.mdev_ډله) {
                إذا (!peers_str.تشه()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = جديد ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    لكل (size_t i = 0; i < opt_ndev; i++) {
        خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        ړنګول dev_ctx;
    }
}

ساكن ثابت حرف * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    أرجع "HTP";
    GGML_ناکارول_شوی(reg);
}

ساكن size_t ggml_backend_hexagon_reg_get_device_شمېر(ggml_backend_reg_t reg) {
    أرجع opt_ndev;
    GGML_ناکارول_شوی(reg);
}

ساكن ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t شاخص) {
    خپلکاری hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    إذا (شاخص >= opt_ndev || !hreg->devices[شاخص].context) {
        أرجع تش_اشاره;
    }

    أرجع &hreg->devices[شاخص];
}

// ** communication context for tensor-split allreduce

ساكن فراغ * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    إذا (n_backends < 2 || n_backends > 4) {
        أرجع تش_اشاره;
    }

    لكل (size_t i = 0; i < n_backends; ++i) {
        إذا (!ggml_backend_is_hexagon(backends[i])) {
            أرجع تش_اشاره;
        }
    }

    لكل (size_t i = 0; i < n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        لكل (size_t j = i + 1; j < n_backends; j++) {
            خپلکاری sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            إذا (sess_i->phys_idx == sess_j->phys_idx) {
                أرجع تش_اشاره;
            }
        }
    }

    خپلکاری * ctx = جديد ggml_backend_hexagon_comm_context();
    ctx->backends.ټاکل(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ساكن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    لكل (size_t i = 0; i < n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (متطاير uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].بفر = &sess_i->fence_buf->backend_بفر;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].معلومات   = (فراغ *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = حجم(int32_t);
        ctx->fence_tensors[i].nb[1]  = حجم(int32_t);
        ctx->fence_tensors[i].nb[2]  = حجم(int32_t);
        ctx->fence_tensors[i].nb[3]  = حجم(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    أرجع ctx;
}

ساكن فراغ ggml_backend_hexagon_comm_free(فراغ * comm_ctx_v) {
    إذا (!comm_ctx_v) أرجع;
    خپلکاری * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    لكل (size_t i = 0; i < ctx->n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((فراغ *) ctx->fence_slots[i], 1);
    }
    ړنګول ctx;
}

ساكن منطقي ggml_backend_hexagon_comm_allreduce_tensor(فراغ * comm_ctx_v, بنية ggml_tensor ** tensors) {
    إذا (opt_ar_select == 0 || !comm_ctx_v) أرجع خطأ;
    خپلکاری * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ثابت size_t n_backends = comm_ctx->n_backends;

    إذا (n_backends < 2 || n_backends > 4) أرجع خطأ;

    لكل (size_t i = 0; i < n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        لكل (size_t j = i + 1; j < n_backends; j++) {
            خپلکاری sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            إذا (sess_i->phys_idx == sess_j->phys_idx) {
                أرجع خطأ;
            }
        }
    }

    لكل (size_t i = 0; i < n_backends; i++) {
        إذا (!tensors[i] || !tensors[i]->بفر || !ggml_backend_بفر_is_hexagon(tensors[i]->بفر)) {
            أرجع خطأ;
        }
        إذا (tensors[i]->type != tensors[0]->type) {
            أرجع خطأ;
        }
        إذا (!ggml_is_contiguous(tensors[i])) {
            أرجع خطأ;
        }
        إذا (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            أرجع خطأ;
        }
    }

    إذا (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    لكل (size_t r = 0; r < n_backends; r++) {
        خپلکاری sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        بنية htp_allreduce_kernel_params kparams;
        إذا (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, خطأ, خطأ, /*is_shard_ok=*/ خطأ, &kparams)) {
            أرجع خطأ;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    لكل (size_t i = 1; i < n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        إذا ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    إذا (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_ثبت = max_seq;
    إذا (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    لكل (size_t i = 0; i < n_backends; i++) {
        خپلکاری sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ثابت ggml_tensor *> معلومات_tensors(n_backends);
    std::vector<ثابت ggml_tensor *> sync_tensors(n_backends);
    لكل (size_t i = 0; i < n_backends; i++) {
        معلومات_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    لكل (size_t r = 0; r < n_backends; r++) {
        خپلکاری sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->کتار_ته_ورزیاتول_allreduce(tensors[r], معلومات_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_ثبت, fence_seq_exit);
        لكل (size_t j = 0; j < n_backends; j++) {
            إذا (r != j) {
                sess->جمع_کړه_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    أرجع ریښتیا;
}

ساكن ggml_backend_بفر_type_t ggml_backend_hexagon_وېشل_بفر_type(صحيح main_device, ثابت عائم * tensor_وېشل) {
    GGML_ناکارول_شوی(tensor_وېشل);
    خپلکاری reg = ggml_backend_hexagon_reg();
    خپلکاری dev = ggml_backend_reg_dev_get(reg, main_device);
    إذا (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    إذا (!dev) أرجع تش_اشاره;
    خپلکاری dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    أرجع &dev_ctx->بفر_type;
}

ساكن فراغ * ggml_backend_hexagon_get_proc_پته(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ناکارول_شوی(reg);
    إذا (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        أرجع (فراغ *) ggml_backend_hexagon_وېشل_بفر_type;
    }
    إذا (strcmp(name, "ggml_backend_comm_init") == 0) {
        أرجع (فراغ *) ggml_backend_hexagon_comm_init;
    }
    إذا (strcmp(name, "ggml_backend_comm_free") == 0) {
        أرجع (فراغ *) ggml_backend_hexagon_comm_free;
    }
    إذا (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        أرجع (فراغ *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    أرجع عدم;
}

کينډۍ<ډول_نوم T> std::vector<T> str_منزل_vec(ثابت حرف* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    طالما (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, تش_اشاره, 0));
    }

    أرجع v;
}

کينډۍ<ډول_نوم T, صحيح بنسټ_2=10> std::string vec_منزل_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(بنسټ_2) << std::showbase;
    لكل (خپلکاری i : v) { ss << i << ','; }
    خپلکاری str = ss.str(); str.pop_back(); // drop last comma
    أرجع str;
}

ساكن فراغ ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, منطقي discovery_supported, ثابت std::unordered_map<صحيح, fastrpc_domain> & cdsp_map) {
    إذا (discovery_supported) {
        خپلکاری it = cdsp_map.find(cfg.physical_idx);
        إذا (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } وإلا {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } وإلا {
        بدل (cfg.physical_idx) {
            حالة 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                اكسر;
            حالة 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                اكسر;
            افتراضي:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                اكسر;
        }
    }
    لكل (خپلکاری & sub_cfg : cfg.mdev_ډله) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ساكن فراغ ggml_hexagon_discover_devices() {
    std::unordered_map<صحيح, fastrpc_domain> cdsp_map;
    منطقي discovery_supported = خطأ;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = تش_اشاره;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_سېټ_TYPE(0, FASTRPC_NSP);

    صحيح err = remote_system_request(&domain_info);
    إذا (err == AEE_SUCCESS && domain_info.sys.عدد_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.عدد_domains);
        domain_info.sys.domains     = domains.معلومات();
        domain_info.sys.max_domains = (صحيح) domains.size();

        err = remote_system_request(&domain_info);
        إذا (err == AEE_SUCCESS) {
            discovery_supported = ریښتیا;
            ثابت صحيح n_domains = std::min(domain_info.sys.عدد_domains, (صحيح) domains.size());
            لكل (صحيح i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (صحيح) domains[i].type, domains[i].id, domains[i].name, domains[i].حالت, domains[i].instance_id);
                إذا (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (صحيح) domains[i].type);
                    استمر;
                }
                إذا (!domains[i].حالت) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    استمر;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } وإلا {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (بینښه) err);
        }
    } وإلا إذا (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (بینښه) err);
    }

    // Populate domain IDs and names for all configured devices
    لكل (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ساكن فراغ ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((بینښه صحيح) HTP_TYPE_Q4_0 == (بینښه صحيح) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q4_1 == (بینښه صحيح) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q8_0 == (بینښه صحيح) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_MXFP4 == (بینښه صحيح) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_IQ4_NL == (بینښه صحيح) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q4_K == (بینښه صحيح) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q5_K == (بینښه صحيح) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q6_K == (بینښه صحيح) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q3_K == (بینښه صحيح) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q2_K == (بینښه صحيح) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q1_0 == (بینښه صحيح) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_Q2_0 == (بینښه صحيح) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((بینښه صحيح) HTP_TYPE_BF16 == (بینښه صحيح) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ثابت حرف * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ثابت حرف * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ثابت حرف * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ثابت حرف * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ثابت حرف * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ثابت حرف * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ثابت حرف * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ثابت حرف * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ثابت حرف * str_etm      = getenv("GGML_HEXAGON_ETM");
    ثابت حرف * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ثابت حرف * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ثابت حرف * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ثابت حرف * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ثابت حرف * str_fa_سر_وېشل = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ثابت حرف * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ثابت حرف * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ثابت حرف * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ثابت حرف * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ثابت حرف * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ثابت حرف * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ثابت حرف * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ثابت حرف * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ثابت حرف * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ثابت حرف * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ثابت حرف * str_power_ډېرپېښ = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    إذا (!str_arch) {
        صحيح err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        إذا (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } وإلا {
            إذا (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } وإلا إذا (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } وإلا {
        إذا (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, عدم, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    خپلکاری RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? جديد std::regex(str_opfilter, RE_ICASE) : عدم;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, عدم, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, عدم, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, عدم, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  عدم, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    إذا (str_power_ډېرپېښ) {
        صحيح m = htp_power_ډېرپېښ_from_str(str_power_ډېرپېښ);
        إذا (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_ډېرپېښ);
        } وإلا {
            opt_power_ډېرپېښ = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, عدم, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_سر_وېشل = str_fa_سر_وېشل ? atoi(str_fa_سر_وېشل)       : opt_fa_سر_وېشل;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    إذا (str_mbuf) {
        ثابت حرف * p = str_mbuf;
        لكل (صحيح idx = 0; idx < 3 && p && *p; idx++) {
            طالما (*p == ' ') p++;
            ثابت حرف * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : مزي_اوږدوالی(p);
            طالما (len > 0 && p[len - 1] == ' ') len--;
            إذا (len > 0) {
                std::string token(p, len);
                إذا (idx == 0) opt_mbuf_dyn    = parse_اندازه(token.c_str());
                إذا (idx == 1) opt_mbuf_static = parse_اندازه(token.c_str());
                إذا (idx == 2) opt_mbuf_ټول  = parse_اندازه(token.c_str());
            }
            إذا (!comma) اكسر;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_اندازه(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ثابت حرف * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    إذا (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    إذا (str_devices && str_devices[0] != '\0') {
        منطقي is_single_شمېره = ریښتیا;
        لكل (صحيح i = 0; str_devices[i] != '\0'; i++) {
            إذا (!که_عدد((بینښه حرف)str_devices[i])) {
                is_single_شمېره = خطأ;
                اكسر;
            }
        }
        إذا (is_single_شمېره) {
            صحيح n = atoi(str_devices);
            إذا (n < 1) n = 1;
            إذا (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            لكل (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (صحيح)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_ډله.clear();
            }
        } وإلا {
            std::string s_devices(str_devices);
            std::vector<std::string> توکي;
            std::string curr_توکی;
            صحيح bracket_ژوروالی = 0;
            لكل (حرف ch : s_devices) {
                إذا (ch == '[') {
                    bracket_ژوروالی++;
                    curr_توکی += ch;
                } وإلا إذا (ch == ']') {
                    إذا (bracket_ژوروالی > 0) bracket_ژوروالی--;
                    curr_توکی += ch;
                } وإلا إذا (ch == ',' && bracket_ژوروالی == 0) {
                    size_t s = curr_توکی.find_first_not_of(" \t\r\n");
                    size_t e = curr_توکی.find_last_not_of(" \t\r\n");
                    إذا (s != std::string::npos) {
                        توکي.push_back(curr_توکی.substr(s, e - s + 1));
                    }
                    curr_توکی.clear();
                } وإلا {
                    curr_توکی += ch;
                }
            }
            size_t s = curr_توکی.find_first_not_of(" \t\r\n");
            size_t e = curr_توکی.find_last_not_of(" \t\r\n");
            إذا (s != std::string::npos) {
                توکي.push_back(curr_توکی.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            لكل (ثابت خپلکاری & توکی : توکي) {
                size_t b_open  = توکی.find('[');
                size_t b_close = توکی.rfind(']');

                إذا (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = توکی.substr(0, b_open);
                    std::string content  = توکی.substr(b_open + 1, b_close - b_open - 1);

                    صحيح virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    إذا (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        حاول {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } امسك (...) {
                            virt = 0;
                        }
                    } وإلا {
                        size_t dev_colon = dev_name.find(':');
                        إذا (dev_colon != std::string::npos) {
                            حاول {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } امسك (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<صحيح> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    طالما (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        إذا (ps == std::string::npos) استمر;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        إذا (dash_pos != std::string::npos) {
                            حاول {
                                صحيح p_پیل = std::stoi(p_part.substr(0, dash_pos));
                                صحيح p_پای_ټکی   = std::stoi(p_part.substr(dash_pos + 1));
                                لكل (صحيح p = p_پیل; p <= p_پای_ټکی; p++) {
                                    إذا (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } امسك (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } وإلا {
                            حاول {
                                صحيح p = std::stoi(p_part);
                                إذا (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } امسك (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    إذا (phys_list.تشه()) {
                        phys_list.push_back(0);
                    }

                    إذا (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        خپلکاری & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_ډله.clear();

                        لكل (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_ډله.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } وإلا {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, توکی.c_str());
                    }
                } وإلا إذا (توکی.rfind("HTP", 0) == 0) {
                    std::string rest = توکی.substr(3);
                    size_t colon_pos = rest.find(':');
                    صحيح phys = 0;
                    صحيح virt = 0;
                    حاول {
                        إذا (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } وإلا {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } امسك (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", توکی.c_str());
                        استمر;
                    }

                    إذا (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_ډله.clear();
                        opt_ndev++;
                    } وإلا {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, توکی.c_str());
                    }
                } وإلا {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", توکی.c_str());
                }
            }
        }
    } وإلا {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_ډله.clear();
    }

#إذا defined(__ANDROID__)
    إذا (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#د_که_پای

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    إذا (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            خپلکاری v  = str_منزل_vec<uint32_t>(str_profile);
            بدل (v.size()) {
                حالة 1:  opt_profile = v[0]; أرجع opt_pmu_evt; // mode with default pmu events
                حالة 8:  opt_profile = 2;    أرجع v;           // mode with custom  pmu events
                افتراضي: opt_profile = 0;    أرجع {};          // garbage input
            }}();
        إذا (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_منزل_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = جديد ggml_hexagon_registry(reg);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_شمېر,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_پته,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(فراغ) {
    ساكن منطقي initialized = خطأ;

    ساكن ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_نسخه,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ عدم };

    {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> قلف(mutex);
        إذا (!initialized) {
            خپلکاری nErr = htpdrv_init();
            إذا (nErr != AEE_SUCCESS) {
                أرجع عدم;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = ریښتیا;
    }

    أرجع &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
