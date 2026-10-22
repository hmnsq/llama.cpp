#تعریف_کرو USE_MATH_متعین_کرتا

#شامل <assert.h>
#شامل <inttypes.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <time.h>

#شامل <atomic>
#شامل <memory>
#شامل <chrono>
#شامل <mutex>
#شامل <thread>
#شامل <cstddef>
#شامل <stdexcept>
#شامل <string>
#شامل <sstream>
#شامل <iomanip>
#شامل <unordered_set>
#شامل <unordered_map>
#شامل <regex>
#شامل <queue>
#شامل <deque>
#شامل <algorithm>
#شامل <cmath>
#شامل <initializer_list>

#اگر_غیر_تعریف_شدہ M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#اگر_ختم

#اگر_تعریف_شدہ _WIN32
#    define WIN32_LEAN_AND_اوسط
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#ورنہ
#    include <semaphore.h>
#    include <unistd.h>
#اگر_ختم

#ہدایت clang diagnostic ignored "-Wnested-anon-types"
#ہدایت clang diagnostic ignored "-Wlanguage-extension-token"
#ہدایت clang diagnostic ignored "-Wgnu-anonymous-struct"
#ہدایت clang diagnostic ignored "-Wmicrosoft-enum-value"

#شامل <AEEStdErr.h>
#شامل <dspqueue.h>
#شامل <rpcmem.h>

#تعریف_کرو GGML_COMMON_IMPL_CPP
#شامل "ggml-backend-impl.h"
#شامل "ggml-alloc.h"
#شامل "ggml-common.h"
#شامل "ggml-hexagon.h"
#شامل "ggml-impl.h"
#شامل "ggml-quants.h"
#شامل "htp-opnode.h"
#شامل "htp-ops.h"
#شامل "htp/matmul-ops.h"
#شامل "htp/binary-ops.h"
#شامل "htp/flash-attn-ops.h"
#شامل "htp/unary-ops.h"
#شامل "htp/get-rows-ops.h"
#شامل "htp/set-rows-ops.h"
#شامل "htp/softmax-ops.h"
#شامل "htp/pool-ops.h"
#شامل "htp/rope-ops.h"
#شامل "htp/ssm-conv.h"
#شامل "htp/gated-delta-net-ops.h"
#شامل "htp/argsort-ops.h"
#شامل "htp/concat-ops.h"
#شامل "htp/cpy-ops.h"
#شامل "htp/im2col-ops.h"
#شامل "htp_iface.h"
#شامل "htp-drv.h"
#شامل "htp-repack.h"

استعمال intvec  = std::vector<صحیح>;
استعمال uintvec = std::vector<غیرعلامتی صحیح>;
استعمال u32vec  = std::vector<uint32_t>;

#تعریف_کرو GGML_HEXAGON_MAX_SESSIONS          16

#تعریف_کرو GGML_HEXAGON_FENCE_خانہ_سائز       128

ساخت ggml_hexagon_device_config {
    صحیح         physical_idx = 0;
    صحیح         virtual_idx  = 0;
    صحیح         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_گروہ;
};

ساکن ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ساکن صحیح    opt_arch    = 0; // autodetect
ساکن size_t opt_ndev    = 1;
ساکن size_t opt_nhvx    = 0; // use all
ساکن صحیح    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ساکن size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ساکن صحیح    opt_etm     = 0;
ساکن صحیح    opt_verbose = 0;
ساکن صحیح    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ساکن منطقی   opt_hostbuf = جھوٹ;
ساکن منطقی   opt_dma64   = جھوٹ;
ساکن صحیح    opt_power_منوال = HTP_POWER_منوال_مدت; // DCVS/HMX power profile requested at session start

ساکن ثابت std::pair<ثابت حرف *, صحیح> htp_power_منوال_نام[] = {
    { "low_power_saver",            HTP_POWER_منوال_کم_POWER_SAVER },
    { "power_saver",                HTP_POWER_منوال_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_منوال_بلند_POWER_SAVER },
    { "low_balanced",               HTP_POWER_منوال_کم_متوازن },
    { "balanced",                   HTP_POWER_منوال_متوازن },
    { "high_performance",           HTP_POWER_منوال_بلند_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_منوال_SUSTAINED_بلند_PERFORMANCE },
    { "burst",                      HTP_POWER_منوال_مدت },
    { "default",                    HTP_POWER_منوال_مدت }, // flip this to change the default mode
};

ساکن صحیح htp_power_منوال_from_str(ثابت حرف * s) {
    برائے (خودکار & e : htp_power_منوال_نام) {
#اگر defined(_MSC_VER)
        اگر (_stricmp(s, e.پہلا) == 0) {
#ورنہ
        اگر (strcasecmp(s, e.پہلا) == 0) {
#اگر_ختم
            واپس e.second;
        }
    }
    واپس -1;
}

ساکن ثابت حرف * htp_power_منوال_مقصد_str(صحیح mode) {
    برائے (خودکار & e : htp_power_منوال_نام) {
        اگر (e.second == mode) {
            واپس e.پہلا;
        }
    }
    واپس "unknown";
}

ساکن size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ساکن size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ساکن size_t opt_mbuf_کل  = 0;                        // total buffer space limit (0 = unconstrained)

ساکن صحیح    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ساکن صحیح    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ساکن صحیح    opt_fa_سر_بانٹو = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ساکن صحیح    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ساکن صحیح    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ساکن صحیح    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ساکن u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ساکن صحیح opt_opbatch  = 1280; // max number of ops in a batch
ساکن صحیح opt_opqueue  = 8;   // max number of pending batches
ساکن صحیح opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ساکن صحیح opt_oppoll   = 0;    // polling for batch completions
ساکن صحیح opt_opfusion  = 1;    // enable/disable op fusion
ساکن صحیح opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

شمار ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_جوڑو = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_جوڑو   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ساکن اندرونی منطقی ggml_hexagon_is_fusion_enabled(صحیح flag) {
    اگر (opt_opfusion <= 0) واپس جھوٹ;
    اگر (opt_opfusion == 1) واپس سچ; // 1 enables all
    واپس (opt_opfusion & flag) != 0;
}

ساکن std::regex* opt_opfilter = تہی; // regex of ops to not claim

#تعریف_کرو HEX_VERBOSE(...) \
    اگر (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ساکن ثابت حرف * حالت_مقصد_str(uint32_t حالت_2) {
    انتخاب (حالت_2) {
        حالت HTP_حالت_OK:
            واپس "OK";
        حالت HTP_حالت_NO_SUPPORT:
            واپس "NO-SUPPORT";
        حالت HTP_حالت_INVAL_PARAMS:
            واپس "INVAL-PARAMS";
        حالت HTP_حالت_VTCM_TOO_SMALL:
            واپس "VTCM-TOO-SMALL";
        حالت HTP_حالت_اندرونی_ERR:
            واپس "INTERNAL-ERROR";
        طےشدہ:
            واپس "UNKNOWN";
    }
}

// ** debug helpers

ساکن خالی ggml_hexagon_dump_op_exec(ثابت std::string &sess_name, ثابت htp_opnode & گرہ, ثابت uint32_t req_flags) {
    اگر (!opt_verbose) واپس;

    htp_opformat fmt(گرہ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                گرہ.op_name().c_str(), fmt.نام, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ساکن خالی ggml_hexagon_dump_op_supp(ثابت std::string &sess_name, ثابت ساخت ggml_tensor * op, منطقی supp) {
    اگر (!opt_verbose) واپس;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.نام, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ساکن ثابت حرف * htp_event_name(uint16_t id) {
    انتخاب (id) {
        حالت HTP_TRACE_EVT_DMA:            واپس "DMA";
        حالت HTP_TRACE_EVT_HVX_COMP:       واپس "HVX_COMP";
        حالت HTP_TRACE_EVT_HVX_A_QUANT:    واپس "HVX_A_QUANT";
        حالت HTP_TRACE_EVT_HVX_A_PREP:     واپس "HVX_A_PREP";
        حالت HTP_TRACE_EVT_HVX_W_DEQUANT:  واپس "HVX_W_DEQUANT";
        حالت HTP_TRACE_EVT_HVX_W_PREP:     واپس "HVX_W_PREP";
        حالت HTP_TRACE_EVT_HVX_O_PROC:     واپس "HVX_O_PROC";
        حالت HTP_TRACE_EVT_HVX_FA_QK:      واپس "HVX_QK_FA";
        حالت HTP_TRACE_EVT_HVX_FA_SFM:     واپس "HVX_SFM_FA";
        حالت HTP_TRACE_EVT_HVX_FA_Q_PREP:  واپس "HVX_Q_PREP";
        حالت HTP_TRACE_EVT_HVX_FA_K_PREP:  واپس "HVX_K_PREP";
        حالت HTP_TRACE_EVT_HVX_FA_V_PREP:  واپس "HVX_V_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_PREP:   واپس "HVX_GDN_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_SOLVE:  واپس "HVX_GDN_SOLVE";
        حالت HTP_TRACE_EVT_HVX_GDN_V_PREP: واپس "HVX_GDN_V_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_D_PREP: واپس "HVX_GDN_D_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_برآمد:    واپس "HVX_GDN_OUT";
        حالت HTP_TRACE_EVT_HVX_GDN_STATE:  واپس "HVX_GDN_STATE";
        حالت HTP_TRACE_EVT_HVX_GDN_REM:    واپس "HVX_GDN_REM";
        حالت HTP_TRACE_EVT_HMX_COMP:       واپس "HMX_COMP";
        حالت HTP_TRACE_EVT_L2FLUSH:        واپس "L2FLUSH";
        حالت HTP_TRACE_EVT_INIT:           واپس "INIT";
        حالت HTP_TRACE_EVT_BUFF:           واپس "BUFF";
        حالت HTP_TRACE_EVT_FENCE:          واپس "FENCE";
        طےشدہ:                           واپس "UNKNOWN";
    }
}

ساکن خالی ggml_hexagon_dump_op_prof(ثابت std::string &sess_name, ثابت htp_opnode & گرہ, ثابت htp_prof_desc & pd) {
    اگر (!opt_profile) واپس;

    uint32_t op_usec = pd.usecs;
    uint32_t op_ادوار = pd.ادوار_stop - pd.ادوار_آغاز;
    ثابت uint32_t * pmu = pd.pmu;

    حرف pmu_str[256] = "";
    اگر (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, حجم(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(گرہ);
    اعشاری mhz = op_usec > 0 ? (اعشاری) op_ادوار / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            گرہ.op_name().c_str(), fmt.نام, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_ادوار, pd.ادوار_آغاز, mhz, pmu_str);
}

ساکن خالی ggml_hexagon_dump_batch_prof(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp) {
    uint64_t batch_ادوار = rsp.ادوار_stop - rsp.ادوار_آغاز;
    اعشاری batch_mhz = rsp.usecs > 0 ? (اعشاری) batch_ادوار / rsp.usecs : 0.0f;

    حرف evt_str[256] = "----";
    اگر (opt_profile == 3) {
        snprintf(evt_str, حجم(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (غیرعلامتی طویل طویل) batch_ادوار, (غیرعلامتی طویل طویل) rsp.ادوار_آغاز, batch_mhz);
}

ساکن خالی ggml_hexagon_dump_trace_events(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp,
                                           ثابت htp_trace_desc * trace_events, uint32_t n_traces) {
    اگر (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        برائے (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        برائے (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            برائے (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ثابت خودکار & e = trace_events[t * n_traces + idx];
                منطقی is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.ادوار);
            }
        }
    }
}

شمار ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_وزن    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ساخت ggml_hexagon_session;

ساکن خالی ggml_hexagon_precompute_matmul_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_mm_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_fused_matmul_جوڑو_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * src2,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_mm_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_unary_params(
    ثابت ساخت ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_unary_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_precompute_ثنائی_params(
    ثابت ساخت ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_ثنائی_kernel_params * kparams,
    ثابت ساخت ggml_tensor * src2 = خالی_اشاریہ
);

ساکن خالی ggml_hexagon_precompute_get_صفیں_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_get_صفیں_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_سیٹ_صفیں_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_سیٹ_صفیں_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_softmax_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_softmax_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_rope_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_rope_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_ssm_conv_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_ssm_conv_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_gated_delta_net_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_gdn_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_ترتیب_دو_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    منطقی is_چوٹی_k,
    ساخت htp_ترتیب_دو_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_pool_2d_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_pool_2d_kernel_params * kparams,
    منطقی is_pool_1d
);

ساکن منطقی ggml_hexagon_precompute_concat_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_concat_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_precompute_cpy_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_copy_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_precompute_im2col_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_im2col_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_precompute_im2col_3d_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_im2col_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_fused_mmnx_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * act,
    int32_t n_weights,
    ساخت htp_mm_kernel_params * kparams
);

ساکن خالی ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * act,
    ثابت ساخت ggml_tensor * dst,
    int32_t n_weights,
    ساخت htp_mm_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_precompute_allreduce_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقی has_جوڑو,
    منطقی is_صف_bcast,
    منطقی is_shard_ok,
    ساخت htp_allreduce_kernel_params * kparams
);

ساکن منطقی ggml_hexagon_صفیں_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride);
ساکن منطقی ggml_hexagon_matmul_can_collapse(ثابت ساخت ggml_tensor * src0, ثابت ساخت ggml_tensor * src1, ثابت ساخت ggml_tensor * dst);
ساکن ggml_tensor ggml_hexagon_tensor_collapse_صفیں(ثابت ساخت ggml_tensor * t);

ساکن منطقی mm_is_hmx_eligible(ثابت ggml_tensor * t);
ساکن htp_op_رمز op_remap_مقصد_htp(ثابت ggml_tensor * t);
ساکن منطقی is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت ساخت htp_mm_kernel_params * kparams);
ساکن منطقی is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت ساخت htp_mm_kernel_params * kparams);
ساکن منطقی is_mergeable_mul_mat(ثابت ggml_tensor * t);
ساکن منطقی is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);
ساکن منطقی is_mergeable_mul_mat_id(ثابت ggml_tensor * t);
ساکن منطقی is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);

// ** backend sessions

ساخت ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_سائز { 0 };
    uint32_t             flags { 0 };
};

ساکن اندرونی منطقی ggml_hexagon_tensor_is_fuseable(ثابت ساخت ggml_tensor * t) {
    اگر (!t->extra) واپس جھوٹ;
    خودکار extra = (ثابت ساخت ggml_hexagon_tensor_extra *) t->extra;
    واپس (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ساکن اندرونی منطقی ggml_hexagon_can_rt_repack(ثابت ساخت ggml_tensor * t) {
    اگر (opt_rt_repack == 0 || !htp_is_tiled_وزن_tensor(t)) {
        واپس جھوٹ;
    }
    اگر (t->بفر && ggml_backend_buft_is_host(t->بفر->buft)) {
        واپس جھوٹ;
    }
    واپس opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ساکن اندرونی منطقی ggml_hexagon_tensor_is_tiled(ثابت ساخت ggml_tensor * t) {
    ثابت خودکار * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    اگر (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        واپس سچ;
    }
    واپس ggml_hexagon_can_rt_repack(t);
}

ساکن اندرونی منطقی ggml_hexagon_needs_repack(ثابت ساخت ggml_tensor * t) {
    ثابت خودکار * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    اگر (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        واپس جھوٹ;
    }
    واپس ggml_hexagon_can_rt_repack(t);
}

ساکن اندرونی منطقی ggml_hexagon_tensors_overlap(ثابت ساخت ggml_tensor * a, ثابت ساخت ggml_tensor * b) {
    ثابت uintptr_t a0 = (uintptr_t) a->ڈیٹا;
    ثابت uintptr_t b0 = (uintptr_t) b->ڈیٹا;
    ثابت uintptr_t a1 = a0 + ggml_nbytes(a);
    ثابت uintptr_t b1 = b0 + ggml_nbytes(b);

    واپس a0 < b1 && b0 < a1;
}

ساکن اندرونی منطقی ggml_hexagon_can_صف_تقسیم(ثابت ساخت ggml_tensor * t) {
    اگر (t->ne[1] > 1 && (t->nb[1] & 127) != 0) واپس جھوٹ;
    اگر (t->ne[2] > 1 && (t->nb[2] & 127) != 0) واپس جھوٹ;
    اگر (t->ne[3] > 1 && (t->nb[3] & 127) != 0) واپس جھوٹ;
    واپس سچ;
}

ساخت htp_opnode;

ساخت ggml_hexagon_opbatch;
ساخت ggml_hexagon_opqueue;
ساخت ggml_hexagon_shared_بفر;
ساخت ggml_hexagon_fence_بفر;
ساخت ggml_hexagon_session;
ساخت ggml_backend_hexagon_device_context;

ساخت ggml_hexagon_mdev_گروہ {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ساخت ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    متلون uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ساخت ggml_hexagon_event {
    ggml_hexagon_session * sess         = خالی_اشاریہ;
    ggml_hexagon_session * fence_sess   = خالی_اشاریہ;
    متلون uint32_t *    fence_خانہ   = خالی_اشاریہ;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ساخت ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         قطار_id;
    صحیح              phys_idx;
    صحیح              virt_idx;
    منطقی             valid_session;
    منطقی             valid_handle;
    منطقی             valid_قطار;
    منطقی             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_قطار;

    std::unordered_map<صحیح, std::unique_ptr<ggml_hexagon_shared_بفر>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_دھاگے   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_سائز   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> آخری_error{HTP_حالت_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_گروہ                mdev;
    ggml_backend_dev_t                     dev       = خالی_اشاریہ;
    ggml_backend_hexagon_device_context *  dev_ctx   = خالی_اشاریہ;
    ggml_hexagon_fence_بفر *            fence_buf = خالی_اشاریہ;

    ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev = خالی_اشاریہ, uint32_t mdev_idx = 0, uint32_t mdev_گنتی = 0) noexcept(جھوٹ);
    ~ggml_hexagon_session() noexcept(سچ);

    ثابت حرف* c_name() ثابت { واپس name.c_str(); }

    خالی allocate(ثابت ggml_hexagon_device_config & config) noexcept(جھوٹ);
    خالی release() noexcept(سچ);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    خالی      free_fence(خالی * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_خانہ = خالی_اشاریہ;
    std::unordered_map<uint64_t, متلون uint32_t *> cpy_fence_slots;

    خالی قطار_میں_ڈالو_mdev_گروہ();
    خالی قطار_میں_ڈالو_op(ثابت htp_opnode & گرہ);
    خالی قطار_میں_ڈالو_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor = خالی_اشاریہ, uint32_t fence_seq = 0);
    خالی قطار_میں_ڈالو_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq = 0, منطقی wait = سچ);
    خالی قطار_میں_ڈالو_allreduce(ثابت ggml_tensor * dst, ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
                           ثابت std::vector<ثابت ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_اندراج = 0, uint32_t fence_seq_exit = 0);

    خالی آغاز_batch();
    خالی flush_sync(منطقی all = سچ);
    خالی flush_async();
    خالی flush_batch(size_t min_ops = 1);
    خالی flush_peers();
    خالی flush_pending(منطقی all = سچ);

    ggml_hexagon_shared_بفر * mmap_tensor(ثابت ggml_tensor * t);
    منطقی clone_بفر(ثابت ggml_hexagon_shared_بفر*);
    خالی release_بفر(ثابت ggml_hexagon_shared_بفر*);
    خالی unclone_بفر(ثابت ggml_hexagon_shared_بفر*);

    خالی جوڑو_peer(ggml_hexagon_session * peer) {
        اگر (یہ->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } ورنہ {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ساخت ggml_backend_hexagon_device_context {
    صحیح                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = خالی_اشاریہ;

    ggml_backend_بفر_type بفر_type       = {};
    ggml_backend_بفر_type host_بفر_type  = {};
    ggml_backend_بفر_type fence_بفر_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(صحیح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ثابت حرف * c_name() ثابت { واپس config.name.c_str(); }

    ggml_hexagon_session * session() {
        اگر (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        واپس sess.get();
    }
};

ساخت ggml_backend_hexagon_بفر_type_context {
    ggml_backend_hexagon_بفر_type_context(ثابت std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        یہ->dev_ctx = dev_ctx;
        یہ->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ساخت ggml_hexagon_rpcmem_بلاک {
    uint8_t * بنیاد = خالی_اشاریہ;
    صحیح       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_بلاک(size_t size) {
        بنیاد = (uint8_t *) rpcmem_alloc2(RPCMEM_ڈھیر_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        اگر (!بنیاد) {
            پھینکو std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_مقصد_fd(بنیاد);
        اگر (fd < 0) {
            rpcmem_free(بنیاد);
            پھینکو std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        یہ->size = size;
    }

    ~ggml_hexagon_rpcmem_بلاک() {
        اگر (بنیاد) {
            rpcmem_free(بنیاد);
        }
    }
};

ساخت ggml_hexagon_shared_بفر {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_بلاک> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    منطقی     mapped;
    منطقی     pinned;
    منطقی     extended;

    ثابت حرف * c_name() ثابت { واپس sess->c_name(); }
    uint8_t *    بنیاد()   ثابت { واپس mem ? mem->بنیاد : خالی_اشاریہ; }
    size_t       size()   ثابت { واپس mem ? mem->size : 0;  }
    صحیح          fd()     ثابت { واپس mem ? mem->fd   : -1; }

    خالی mmap(منطقی extended = جھوٹ) {
        اگر (!یہ->mem)   واپس;
        اگر (یہ->mapped) واپس;

        GGML_ASSERT(!یہ->pinned || !extended);

        یہ->extended = extended;
        fastrpc_map_flags flags = یہ->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        صحیح err = fastrpc_mmap(sess->domain_id, fd(), (خالی *) بنیاد(), 0, size(), flags);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (غیرعلامتی) err);
            پھینکو std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (خالی *) بنیاد(), size(), fd(), pinned, extended);

        یہ->mapped = سچ;
    }

    خالی unmap() {
        اگر (!یہ->mapped) واپس;

        اگر (!یہ->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        اگر (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (خالی *) بنیاد(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (خالی *) بنیاد(), size(), fd());

        یہ->mapped = جھوٹ;
    }

    خالی alloc(size_t size) {
        اگر (یہ->mem) واپس;

        یہ->mem = std::make_shared<ggml_hexagon_rpcmem_بلاک>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (خالی *) بنیاد(), یہ->size(), fd(), (صحیح) pinned);
        اگر (یہ->pinned) {
            mmap();
        }
    }

    خالی آزاد() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (خالی *) بنیاد(), size(), fd());
        یہ->mem  = خالی_اشاریہ;
    }

    ggml_hexagon_shared_بفر(ggml_hexagon_session * sess, size_t size, منطقی pinned = جھوٹ) {
        یہ->sess     = sess;
        یہ->mapped   = جھوٹ;
        یہ->pinned   = pinned;
        یہ->extended = جھوٹ;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_انحراف = (size + 4095) & ~4095;
        size_t کل_سائز   = guard_انحراف + 4096;
        اگر (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            کل_سائز = (کل_سائز + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(کل_سائز);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_بفر(ggml_hexagon_session * sess, ثابت ggml_hexagon_shared_بفر & other) {
        یہ->sess   = sess;
        یہ->mem    = other.mem;
        یہ->mapped = جھوٹ;
        یہ->pinned = other.pinned;
        یہ->extended = other.extended;
    }

    ~ggml_hexagon_shared_بفر() {
        آزاد();
        برائے (خودکار * extra : tensor_extra) {
            حذف extra;
        }
    }
};

ساخت ggml_hexagon_fence_بفر : عوامی ggml_hexagon_shared_بفر {
    uint32_t              خانہ_گنتی = 0;
    uint32_t              خانہ_سر  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_بفر   backend_بفر{};

    ggml_hexagon_fence_بفر(ggml_hexagon_session * sess, ggml_backend_بفر_type_t buft, size_t size)
        : ggml_hexagon_shared_بفر(sess, size, جھوٹ /* pinned */),
          خانہ_گنتی(size / GGML_HEXAGON_FENCE_خانہ_سائز),
          خانہ_سر(0) {
        backend_بفر.buft    = buft;
        backend_بفر.context = static_cast<ggml_hexagon_shared_بفر *>(یہ);
        backend_بفر.size    = size;
        mmap(جھوٹ);
    }

    uint8_t * alloc_خانہ(uint32_t n_slots = 1) {
        uint8_t * ptr = خالی_اشاریہ;
        اگر (n_slots == 1 && !free_slots.خالی_خانے()) {
            uint32_t خانہ = free_slots.back();
            free_slots.pop_back();
            ptr = بنیاد() + (size_t) خانہ * GGML_HEXAGON_FENCE_خانہ_سائز;
        } ورنہ اگر (خانہ_سر + n_slots <= خانہ_گنتی) {
            uint32_t خانہ = خانہ_سر;
            خانہ_سر += n_slots;
            ptr = بنیاد() + (size_t) خانہ * GGML_HEXAGON_FENCE_خانہ_سائز;
        }
        اگر (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_خانہ_سائز);
        }
        واپس ptr;
    }

    خالی free_خانہ(خالی * ptr, uint32_t n_slots = 1) {
        اگر (!ptr) واپس;
        uint32_t خانہ = ((uint8_t *) ptr - بنیاد()) / GGML_HEXAGON_FENCE_خانہ_سائز;
        برائے (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(خانہ + i);
        }
    }
};

اندرونی uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_خانہ(n_slots);
    GGML_ASSERT(ptr);
    واپس ptr;
}

اندرونی خالی ggml_hexagon_session::free_fence(خالی * ptr, uint32_t n_slots) {
    اگر (fence_buf) {
        fence_buf->free_خانہ(ptr, n_slots);
    }
}

ساکن خالی ggml_backend_hexagon_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    sbuf->sess->unclone_بفر(sbuf);
    حذف sbuf;
}

ساکن خالی * ggml_backend_hexagon_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    واپس sbuf->بنیاد();
}

ساکن شمار ggml_حالت ggml_backend_hexagon_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(بفر->context);
    خودکار sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (خالی *) sbuf->بنیاد(), tensor->ڈیٹا, ggml_nbytes(tensor));

    خودکار extra = نیا ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_hexagon_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                   ggml_tensor *         tensor,
                                                   ثابت خالی *          ڈیٹا,
                                                   size_t                انحراف,
                                                   size_t                size) {
    خودکار extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خودکار sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خودکار sess  = sbuf->sess;

    اگر (ggml_backend_بفر_get_usage(بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_وزن;
        اگر (htp_is_tiled_وزن_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, ڈیٹا, انحراف, size, (صحیح) بفر->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size);
        واپس;
    }

    اگر (انحراف == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.خالی_خانے()) {
        htp_repack_tensor_tiled(tensor, ڈیٹا, size);
        واپس;
    }

    اگر (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    حافظہ_نقل(extra->shadow_buf.ڈیٹا() + انحراف, ڈیٹا, size);
    extra->shadow_سائز += size;

    اگر (extra->shadow_سائز >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ڈیٹا(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_سائز = 0;
    }
}

ساکن خالی ggml_backend_hexagon_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                   ثابت ggml_tensor *   tensor,
                                                   خالی *                ڈیٹا,
                                                   size_t                انحراف,
                                                   size_t                size) {
    خودکار extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خودکار sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خودکار sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, ڈیٹا, انحراف, size, (صحیح) بفر->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        حافظہ_نقل(ڈیٹا, (ثابت حرف *) tensor->ڈیٹا + انحراف, size);
        واپس;
    }

    htp_unpack_tensor_tiled(ڈیٹا, tensor, انحراف, size);
}

ساکن منطقی ggml_backend_hexagon_بفر_cpy_tensor(ggml_backend_بفر_t      بفر,
                                                   ثابت ساخت ggml_tensor * src,
                                                   ساخت ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    واپس جھوٹ;

    GGML_غیر_مستعمل(بفر);
    GGML_غیر_مستعمل(src);
    GGML_غیر_مستعمل(dst);
}

ساکن خالی ggml_backend_hexagon_بفر_سیٹ_tensor_2d(ggml_backend_بفر_t بفر,
                                                          ggml_tensor *         tensor,
                                                          ثابت خالی *          ڈیٹا,
                                                          size_t                انحراف,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_ڈیٹا) {
    خودکار extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خودکار sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خودکار sess  = sbuf->sess;

    اگر (ggml_backend_بفر_get_usage(بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_وزن;
        اگر (htp_is_tiled_وزن_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا, (صحیح) بفر->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        برائے (size_t i = 0; i < n_copies; i++) {
            حافظہ_نقل((uint8_t *) tensor->ڈیٹا + انحراف + i * stride_tensor, (ثابت uint8_t *) ڈیٹا + i * stride_ڈیٹا, size);
        }
        واپس;
    }

    اگر (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    برائے (size_t i = 0; i < n_copies; i++) {
        حافظہ_نقل(extra->shadow_buf.ڈیٹا() + انحراف + i * stride_tensor, (ثابت uint8_t *) ڈیٹا + i * stride_ڈیٹا, size);
    }
    extra->shadow_سائز += n_copies * size;

    اگر (extra->shadow_سائز >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ڈیٹا(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_سائز = 0;
    }
}

ساکن خالی ggml_backend_hexagon_بفر_get_tensor_2d(ggml_backend_بفر_t بفر,
                                                      ثابت ggml_tensor *   tensor,
                                                      خالی *                ڈیٹا,
                                                      size_t                انحراف,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_ڈیٹا) {
    خودکار extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    خودکار sbuf  = (ggml_hexagon_shared_بفر *) بفر->context;
    خودکار sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا, (صحیح) بفر->usage);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        برائے (size_t i = 0; i < n_copies; i++) {
            حافظہ_نقل((uint8_t *)ڈیٹا + i * stride_ڈیٹا, (ثابت uint8_t *)tensor->ڈیٹا + انحراف + i * stride_tensor, size);
        }
        واپس;
    }

    size_t عارضی_سائز      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t قاش_سائز     = tensor->ne[1] * ggml_صف_سائز(tensor->type, tensor->ne[0]);
    size_t قاش_انحراف   = انحراف % قاش_سائز;
    size_t صف_سائز_بائٹیں = ggml_صف_سائز(tensor->type, tensor->ne[0]);

    GGML_ASSERT((قاش_انحراف % صف_سائز_بائٹیں) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((عارضی_سائز % صف_سائز_بائٹیں)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((انحراف + عارضی_سائز) <= ggml_nbytes(tensor));

    std::vector<uint8_t> عارضی_buf(عارضی_سائز);
    htp_unpack_tensor_tiled(عارضی_buf.ڈیٹا(), tensor, انحراف, عارضی_سائز);

    برائے (size_t i = 0; i < n_copies; i++) {
        حافظہ_نقل((uint8_t *) ڈیٹا + i * stride_ڈیٹا, عارضی_buf.ڈیٹا() + i * stride_tensor, size);
    }
}

ساکن خالی ggml_backend_hexagon_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    خودکار sbuf = (ggml_hexagon_shared_بفر *) بفر->context;
    خودکار sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (خالی *) sbuf->بنیاد(), sbuf->size());
    memset(sbuf->بنیاد(), value, sbuf->size());
}

ساکن ggml_backend_بفر_i ggml_backend_hexagon_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_hexagon_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_hexagon_بفر_init_tensor,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_hexagon_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_بفر_سیٹ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_بفر_clear,
    /* .reset           = */ تہی,
};

// ** backend buffer type

ساکن خالی ggml_backend_hexagon_host_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                        ggml_tensor *         tensor,
                                                        ثابت خالی *          ڈیٹا,
                                                        size_t                انحراف,
                                                        size_t                size) {
    حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size);
    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_hexagon_host_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                        ثابت ggml_tensor *   tensor,
                                                        خالی *                ڈیٹا,
                                                        size_t                انحراف,
                                                        size_t                size) {
    حافظہ_نقل(ڈیٹا, (ثابت حرف *) tensor->ڈیٹا + انحراف, size);
    GGML_غیر_مستعمل(بفر);
}

ساکن ggml_backend_بفر_i ggml_backend_hexagon_host_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_hexagon_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_hexagon_بفر_init_tensor,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_hexagon_host_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_hexagon_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_بفر_clear,
    /* .reset           = */ تہی,
};

// ** backend buffer type

ساکن ثابت حرف * ggml_backend_hexagon_بفر_type_name(ggml_backend_بفر_type_t بفر_type) {
    واپس static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_hexagon_بفر_type_alloc_بفر(
            ggml_backend_بفر_type_t بفر_type, size_t size) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->dev_ctx;
    خودکار sess    = dev_ctx->session();
    اگر (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        واپس خالی_اشاریہ;
    }
    کوشش {
        ggml_hexagon_shared_بفر * sbuf = نیا ggml_hexagon_shared_بفر(sess, size, جھوٹ);
        واپس ggml_backend_بفر_init(بفر_type, ggml_backend_hexagon_بفر_interface, sbuf, size);
    } پکڑو (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        واپس خالی_اشاریہ;
    }
}

ساکن ggml_backend_بفر_t ggml_backend_hexagon_host_بفر_type_alloc_بفر(
            ggml_backend_بفر_type_t بفر_type, size_t size) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type->context)->dev_ctx;
    خودکار sess    = dev_ctx->session();
    اگر (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        واپس خالی_اشاریہ;
    }
    کوشش {
        ggml_hexagon_shared_بفر * sbuf = نیا ggml_hexagon_shared_بفر(sess, size, جھوٹ);
        واپس ggml_backend_بفر_init(بفر_type, ggml_backend_hexagon_host_بفر_interface, sbuf, size);
    } پکڑو (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        واپس خالی_اشاریہ;
    }
}

ساکن size_t ggml_backend_hexagon_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس 128;  // HVX alignment
    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_hexagon_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ساخت ggml_tensor * t) {
    اگر (!ggml_backend_buft_is_host(buft) && htp_is_tiled_وزن_tensor(t)) {
        int64_t ne0 = hex_باری_اوپر(t->ne[0], 32);
        int64_t ne1 = hex_باری_اوپر(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_سائز = htp_tiled_صف_سائز(t->type, ne0) * ne1 * ne2 * ne3;
        واپس std::max(tiled_سائز, ggml_nbytes(t));
    }
    واپس ggml_nbytes(t);
}

ساکن size_t parse_سائز(ثابت حرف * str, size_t default_unit = 1024 * 1024) {
    اگر (!str || str[0] == '\0') {
        واپس 0;
    }
    حرف * end = تہی;
    دوہرا val = strtod(str, &end);
    اگر (val < 0) {
        واپس 0;
    }
    اگر (end && *end) {
        جب (*end == ' ') end++;
        اگر (*end == 'k' || *end == 'K') {
            واپس (size_t) (val * 1024);
        } ورنہ اگر (*end == 'm' || *end == 'M') {
            واپس (size_t) (val * 1024 * 1024);
        } ورنہ اگر (*end == 'g' || *end == 'G') {
            واپس (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    واپس (size_t) (val * default_unit);
}

ساکن size_t ggml_backend_hexagon_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    واپس opt_mbuf_dyn;
    GGML_غیر_مستعمل(buft);
}

ساکن منطقی ggml_backend_hexagon_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    واپس جھوٹ;
    GGML_غیر_مستعمل(buft);
}

ساکن منطقی ggml_backend_hexagon_host_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    واپس سچ;
    GGML_غیر_مستعمل(buft);
}

ساخت ggml_backend_hexagon_alloc_بفر_n_plan_شے {
    size_t size;
    صحیح    پہلا;
    صحیح    آخری;
};

استعمال ggml_backend_hexagon_alloc_بفر_n_plan_t = std::vector<ggml_backend_hexagon_alloc_بفر_n_plan_شے>;

ساکن ثابت حرف * ggml_hexagon_kv_layer_suffix(ثابت ساخت ggml_tensor * t) {
    اگر (لڑی_موازنہ(t->name, "cache_", 6) != 0) {
        واپس تہی;
    }
    ثابت حرف * p = لڑی_تلاش(t->name, "_l");
    اگر (!p || !ہندسہ_ہے((غیرعلامتی حرف)p[2])) {
        واپس تہی;
    }
    واپس p;
}

ساخت ggml_backend_hexagon_alloc_unit {
    size_t size;
    صحیح    پہلا;
    صحیح    آخری;
};

ساکن ggml_backend_hexagon_alloc_بفر_n_plan_t ggml_backend_hexagon_alloc_بفر_n_plan(
        ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ggml_backend_hexagon_alloc_بفر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    صحیح i = 0;
    جب (i < n_tensors) {
        ساخت ggml_tensor * t = tensors[i];
        size_t unit_سائز = 0;
        صحیح unit_پہلا = i;
        صحیح unit_آخری = i + 1;

        اگر (t->ڈیٹا == تہی && t->view_src == تہی) {
            unit_سائز += GGML_PAD(ggml_backend_buft_get_alloc_سائز(buft, t), alignment);
        }

        ثابت حرف * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        جب (unit_آخری < n_tensors) {
            ساخت ggml_tensor * اگلا = tensors[unit_آخری];

            اگر (اگلا->view_src != تہی) {
                unit_آخری++;
                جاری;
            }

            اگر (layer_suffix != تہی) {
                ثابت حرف * اگلا_suffix = ggml_hexagon_kv_layer_suffix(اگلا);
                اگر (اگلا_suffix != تہی && strcmp(layer_suffix, اگلا_suffix) == 0) {
                    اگر (اگلا->ڈیٹا == تہی) {
                        unit_سائز += GGML_PAD(ggml_backend_buft_get_alloc_سائز(buft, اگلا), alignment);
                    }
                    unit_آخری++;
                    جاری;
                }
            }

            توڑو;
        }

        units.push_back({ unit_سائز, unit_پہلا, unit_آخری });
        i = unit_آخری;
    }

    size_t cur_buf_سائز  = 0;
    صحیح    cur_buf_پہلا = 0;

    برائے (ثابت خودکار & unit : units) {
        اگر (unit.size == 0) {
            جاری;
        }

        اگر (cur_buf_سائز > 0 && (cur_buf_سائز + unit.size) > max_size) {
            plan.push_back({ cur_buf_سائز, cur_buf_پہلا, unit.پہلا });
            cur_buf_سائز  = 0;
            cur_buf_پہلا = unit.پہلا;
        }

        cur_buf_سائز += unit.size;
    }

    اگر (cur_buf_سائز > 0) {
        plan.push_back({ cur_buf_سائز, cur_buf_پہلا, n_tensors });
    }

    واپس plan;
}

ساکن ggml_backend_بفر_t ggml_backend_hexagon_بفر_type_alloc_بفر_n(
        ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_بفر_n_plan_t plan = ggml_backend_hexagon_alloc_بفر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_بفر_t> buffers;
    buffers.reserve(plan.size());

    برائے (ثابت خودکار & شے : plan) {
        ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, شے.size);
        اگر (بفر == تہی) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), شے.size);
            برائے (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            واپس تہی;
        }

        ساخت ggml_tallocr tallocr = ggml_tallocr_new(بفر);

        ساخت ggml_tensor * t_failed = تہی;
        برائے (صحیح j = شے.پہلا; j < شے.آخری; j++) {
            ساخت ggml_tensor * t = tensors[j];
            اگر (t->ڈیٹا == تہی) {
                اگر (t->view_src == تہی) {
                    اگر (ggml_tallocr_alloc(&tallocr, t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                } ورنہ اگر (t->بفر == تہی) {
                    اگر (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                }
            } ورنہ {
                اگر (t->view_src != تہی && t->بفر == تہی) {
                    اگر (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        توڑو;
                    }
                }
            }
        }
        اگر (t_failed != تہی) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            برائے (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            ggml_backend_بفر_free(بفر);
            واپس تہی;
        }

        buffers.push_back(بفر);
    }

    اگر (buffers.خالی_خانے()) {
        واپس تہی;
    }

    اگر (buffers.size() == 1) {
        واپس buffers[0];
    }

    واپس ggml_backend_multi_بفر_alloc_بفر(buffers.ڈیٹا(), buffers.size());
}

ساکن size_t ggml_backend_hexagon_بفر_type_get_alloc_سائز_n(
        ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_بفر_n_plan_t plan = ggml_backend_hexagon_alloc_بفر_n_plan(buft, tensors, n_tensors);

    size_t کل = 0;
    برائے (ثابت خودکار & شے : plan) {
        کل += شے.size;
    }
    واپس کل;
}

ساکن ggml_backend_بفر_type_i ggml_backend_hexagon_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_hexagon_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_hexagon_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_بفر_type_get_alloc_سائز_n,
    /* .is_host             = */ ggml_backend_hexagon_بفر_type_is_host,
};

ساکن ggml_backend_بفر_type_i ggml_backend_hexagon_host_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_hexagon_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_hexagon_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_بفر_type_get_alloc_سائز_n,
    /* .is_host             = */ ggml_backend_hexagon_host_بفر_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(صحیح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    بفر_type.device  = dev;
    بفر_type.iface   = ggml_backend_hexagon_بفر_type_interface;
    بفر_type.context = نیا ggml_backend_hexagon_بفر_type_context(config.name, یہ);

    host_بفر_type.device  = dev;
    host_بفر_type.iface   = ggml_backend_hexagon_host_بفر_type_interface;
    host_بفر_type.context = نیا ggml_backend_hexagon_بفر_type_context(config.name + "-HOST", یہ);

    fence_بفر_type.device  = dev;
    fence_بفر_type.iface   = ggml_backend_hexagon_بفر_type_interface;
    fence_بفر_type.context = نیا ggml_backend_hexagon_بفر_type_context(config.name + "-FENCE", یہ);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    حذف static_cast<ggml_backend_hexagon_بفر_type_context *>(بفر_type.context);
    حذف static_cast<ggml_backend_hexagon_بفر_type_context *>(host_بفر_type.context);
    حذف static_cast<ggml_backend_hexagon_بفر_type_context *>(fence_بفر_type.context);
}

ساکن منطقی ggml_backend_بفر_is_hexagon(ثابت ساخت ggml_backend_بفر * b) {
    واپس b->buft->iface.get_alignment == ggml_backend_hexagon_بفر_type_get_alignment;
}

ساخت ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<صحیح, صحیح>                b_map; // buffer fd   to index
    std::unordered_map<ثابت ggml_tensor*, صحیح> t_map; // tensor ptr  to index
    std::unordered_multimap<خالی*, صحیح>         d_map; // tensor data to index

    غیرعلامتی صحیح n_bufs;     // num buffers in the batch
    غیرعلامتی صحیح n_tens;     // num tensors ...
    غیرعلامتی صحیح n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    غیرعلامتی صحیح n_bufs_max;
    غیرعلامتی صحیح n_tens_max;
    غیرعلامتی صحیح n_ops_max;
    size_t       b_vmem_max;

    خالی reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_سائز, size_t max_vmem) {
        یہ->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_سائز;
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

    منطقی خالی_خانے() ثابت { واپس n_ops == 0; }

    // add buffer and return its index
    صحیح جوڑو_بفر(ggml_hexagon_shared_بفر * sbuf) {
        // Lookup by fd
        خودکار it = b_map.find(sbuf->fd());
        اگر (it != b_map.end()) { واپس it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        صحیح bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.بنیاد  = (uint64_t) sbuf->بنیاد();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        اگر (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (خالی*) sbuf->بنیاد(), (size_t) b.size, b_vmem);

        واپس bi;
    }

    منطقی same_shape(ثابت htp_tensor * h, ثابت ggml_tensor * t) ثابت {
        خودکار extra = (ggml_hexagon_tensor_extra *) t->extra;

        ثابت منطقی is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ثابت int64_t ne0_padded = hex_باری_اوپر(t->ne[0], 32);
        ثابت int64_t ne1_padded = hex_باری_اوپر(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_صف_سائز(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? حجم(ggml_fp16_t) : t->nb[0];
        شمار ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        واپس (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    صحیح جوڑو_tensor(ثابت ggml_tensor * t) {
        خودکار extra = (ggml_hexagon_tensor_extra *) t->extra;
        خودکار sbuf  = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);

        // First lookup by tensor data
        خودکار range = d_map.equal_range(t->ڈیٹا);
        برائے (خودکار it = range.پہلا; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            اگر (same_shape(h, t)) { واپس it->second; }
        }

        // Lookup by tensor ptr
        خودکار it = t_map.find(t);
        اگر (it != t_map.end()) { واپس it->second; }

        // Add new tensor to the batch
        صحیح ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->ڈیٹا, ti});

        uint64_t t_انحراف = (uint8_t *) t->ڈیٹا - sbuf->بنیاد();
        size_t   t_سائز   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = جوڑو_بفر(sbuf);
        h.ti    = ti;
        h.ڈیٹا  = t_انحراف;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ثابت منطقی is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        اگر (is_tiled) {
            اگر (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = حجم(ggml_fp16_t);
            } ورنہ {
                h.nb[0] = t->nb[0];
            }
            ثابت int64_t ne0_padded = hex_باری_اوپر(t->ne[0], 32);
            ثابت int64_t ne1_padded = hex_باری_اوپر(t->ne[1], 32);
            h.nb[1] = htp_tiled_صف_سائز(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_سائز  = h.size;
        } ورنہ {
            h.size  = t_سائز;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_وزن) != 0) {
            h.flags |= HTP_TENSOR_وزن;
        }
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (خالی*) t->ڈیٹا, (size_t) t_انحراف, t_سائز, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        واپس ti;
    }

    منطقی fit_op(ثابت htp_opnode & گرہ) ثابت {
        اگر (n_ops >= n_ops_max) واپس جھوٹ;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        صحیح نظر_آیا_bufs[HTP_OP_MAX_BUFS];
        صحیح n_نظر_آیا_bufs = 0;

        خودکار fit_tensor = [&](ثابت ggml_tensor *t) {
            اگر (!t) واپس;
            اگر (!t_map.count(t)) {
                extra_tens++;

                خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
                صحیح fd = sbuf->fd();
                اگر (!b_map.count(fd)) {
                    برائے (صحیح i = 0; i < n_نظر_آیا_bufs; i++) {
                        اگر (نظر_آیا_bufs[i] == fd) واپس;
                    }
                    اگر (n_نظر_آیا_bufs < HTP_OP_MAX_BUFS) {
                        نظر_آیا_bufs[n_نظر_آیا_bufs++] = fd;
                    }
                    اگر (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        برائے (ثابت خودکار * src : گرہ.get_inputs()) {
            fit_tensor(src);
        }
        برائے (ثابت خودکار * output : گرہ.get_outputs()) {
            fit_tensor(output);
        }

        اگر ((extra_bufs + n_bufs) > n_bufs_max) واپس جھوٹ;
        اگر ((extra_tens + n_tens) > n_tens_max) واپس جھوٹ;
        اگر ((extra_vmem + b_vmem) > b_vmem_max) واپس جھوٹ;

        واپس سچ;
    }

    // assumes that fit_op() was called first and returned true
    خالی جوڑو_op(ثابت htp_opnode & گرہ) {
        // Add new op

        غیرعلامتی صحیح n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = گرہ;

        htp_op_desc &o = h_ops[n];
        حافظہ_نقل(o.params,        گرہ.گرہ->op_params, حجم(گرہ.گرہ->op_params));
        حافظہ_نقل(o.kernel_params, گرہ.kernel_params,   حجم(o.kernel_params));
        o.opcode = گرہ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        خودکار inputs = گرہ.get_inputs();
        برائے (غیرعلامتی صحیح i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? جوڑو_tensor(inputs[i]) : 0xffff;
        }

        خودکار outputs = گرہ.get_outputs();
        برائے (غیرعلامتی صحیح i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? جوڑو_tensor(outputs[i]) : 0xffff;
        }
    }

    خالی ترتیب_دو_buffers() {
        اگر (n_bufs <= 1) واپس;

        std::vector<صحیح> ترتیب(n_bufs);
        برائے (غیرعلامتی صحیح i = 0; i < n_bufs; i++) { ترتیب[i] = (صحیح) i; }

        std::stable_sort(ترتیب.begin(), ترتیب.end(), [&](صحیح a, صحیح b) {
            واپس h_bufs[a].size > h_bufs[b].size;
        });

        منطقی already_sorted = سچ;
        برائے (غیرعلامتی صحیح i = 0; i < n_bufs; i++) {
            اگر (ترتیب[i] != (صحیح) i) {
                already_sorted = جھوٹ;
                توڑو;
            }
        }
        اگر (already_sorted) واپس;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        برائے (غیرعلامتی صحیح new_bi = 0; new_bi < n_bufs; new_bi++) {
            صحیح old_bi = ترتیب[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        برائے (غیرعلامتی صحیح i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        برائے (غیرعلامتی صحیح i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    خالی update_mdev_گروہ(uint32_t mdev_idx) {
        اگر (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_گروہ) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    منطقی try_fuse_common(std::initializer_list<ثابت ggml_tensor *> tensors) ثابت {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        صحیح نظر_آیا_bufs[HTP_OP_MAX_BUFS];
        صحیح n_نظر_آیا_bufs = 0;

        برائے (ثابت خودکار * t : tensors) {
            اگر (!t || t_map.count(t)) {
                جاری;
            }
            extra_tens++;
            خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
            صحیح fd = sbuf->fd();
            اگر (!b_map.count(fd)) {
                منطقی ملا = جھوٹ;
                برائے (صحیح i = 0; i < n_نظر_آیا_bufs; i++) {
                    اگر (نظر_آیا_bufs[i] == fd) {
                        ملا = سچ;
                        توڑو;
                    }
                }
                اگر (!ملا) {
                    اگر (n_نظر_آیا_bufs < HTP_OP_MAX_BUFS) {
                        نظر_آیا_bufs[n_نظر_آیا_bufs++] = fd;
                    }
                    اگر (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        اگر ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            واپس جھوٹ;
        }

        واپس سچ;
    }

    منطقی try_fuse_common(ثابت ggml_tensor * t1, ثابت ggml_tensor * t2) ثابت {
        واپس try_fuse_common({t1, t2});
    }

    منطقی try_fuse_allreduce_جوڑو(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0 || opt_ar_select != 2) واپس جھوٹ;
        اگر (گرہ.opcode != HTP_OP_جوڑو) واپس جھوٹ;

        htp_opnode & آخری_گرہ = ops[n_ops - 1];
        اگر (آخری_گرہ.opcode != HTP_OP_ALLREDUCE) واپس جھوٹ;

        خودکار * ar_kparams = (ساخت htp_allreduce_kernel_params *) آخری_گرہ.kernel_params;
        ثابت uint32_t rank    = (uint32_t) ar_kparams->rank;
        ثابت uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ثابت ggml_tensor * ar_local = آخری_گرہ.inputs[rank];
        ثابت ggml_tensor * جوڑو_src0 = گرہ.src0();
        ثابت ggml_tensor * جوڑو_src1 = گرہ.src1();
        ثابت ggml_tensor * جوڑو_dst  = گرہ.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(ar_local)) واپس جھوٹ;

        ثابت ggml_tensor * res_tensor;
        اگر (جوڑو_src0 == ar_local || جوڑو_src0->ڈیٹا == ar_local->ڈیٹا) {
            res_tensor = جوڑو_src1;
        } ورنہ اگر (جوڑو_src1 == ar_local || جوڑو_src1->ڈیٹا == ar_local->ڈیٹا) {
            res_tensor = جوڑو_src0;
        } ورنہ {
            واپس جھوٹ;
        }

        اگر (ar_local->type != res_tensor->type) واپس جھوٹ;

        ثابت منطقی is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ثابت منطقی is_صف_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        اگر (!is_same_shape && !is_صف_bcast) واپس جھوٹ;

        اگر (is_same_shape) {
            اگر (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                واپس جھوٹ;
            }
            اگر (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                واپس جھوٹ;
            }
        }
        اگر (ggml_is_contiguous(ar_local) != ggml_is_contiguous(جوڑو_dst)) {
            واپس جھوٹ;
        }

        // scatter is only valid for in-place add within max outputs
        ثابت منطقی is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && جوڑو_dst->ڈیٹا == ar_local->ڈیٹا;

        ساخت htp_allreduce_kernel_params new_kparams;
        اگر (!ggml_hexagon_precompute_allreduce_params(
            sess, جوڑو_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, سچ, is_صف_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            واپس جھوٹ;
        }

        ثابت منطقی scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_بھرا;

        برائے (uint32_t r = 0; r < n_ranks; r++) {
            ثابت ggml_tensor * ar_src = آخری_گرہ.inputs[r];
            اگر (!ggml_hexagon_tensors_overlap(جوڑو_dst, ar_src)) {
                جاری;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            اگر (scatter_ok && r == rank) {
                جاری;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            واپس جھوٹ;
        }

        اگر (!try_fuse_common(res_tensor, جوڑو_dst)) {
            واپس جھوٹ;
        }

        آخری_گرہ.opcode = HTP_OP_ALLREDUCE_جوڑو;
        آخری_گرہ.name   = "ALLREDUCE+ADD";
        آخری_گرہ.inputs.push_back(res_tensor);
        آخری_گرہ.outputs.clear();
        آخری_گرہ.outputs.push_back(جوڑو_dst);
        آخری_گرہ.fused.push_back(گرہ.گرہ);
        حافظہ_نقل(آخری_گرہ.kernel_params, &new_kparams, حجم(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_جوڑو;
        حافظہ_نقل(o.kernel_params, &new_kparams, حجم(new_kparams));

        o.src[2 * n_ranks] = جوڑو_tensor(res_tensor);
        اگر (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            برائے (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            برائے (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } ورنہ {
            o.dst[0] = جوڑو_tensor(جوڑو_dst);
            برائے (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (صحیح) new_kparams.mode, (صحیح) new_kparams.n_dsts);
        واپس سچ;
    }

    منطقی try_fuse_rms_norm_mul(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0) واپس جھوٹ;
        اگر (گرہ.opcode != HTP_OP_MUL) واپس جھوٹ;

        htp_opnode & آخری_گرہ = ops[n_ops - 1];
        اگر (آخری_گرہ.opcode != HTP_OP_RMS_NORM) واپس جھوٹ;

        ثابت ggml_tensor * mul_src0 = گرہ.src0();
        ثابت ggml_tensor * mul_src1 = گرہ.src1();
        ثابت ggml_tensor * rms_برآمد  = آخری_گرہ.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(rms_برآمد)) واپس جھوٹ;

        ثابت ggml_tensor * وزن;
        اگر (mul_src0 == rms_برآمد || mul_src0->ڈیٹا == rms_برآمد->ڈیٹا) {
            وزن = mul_src1;
        } ورنہ اگر (mul_src1 == rms_برآمد || mul_src1->ڈیٹا == rms_برآمد->ڈیٹا) {
            وزن = mul_src0;
        } ورنہ {
            واپس جھوٹ;
        }

        ثابت ggml_tensor * src0 = آخری_گرہ.src0();

        اگر (src0->ne[0] != وزن->ne[0] || src0->ne[0] != گرہ.dst()->ne[0]) {
            واپس جھوٹ;
        }

        ثابت منطقی is_صف_bcast = (وزن->ne[1] == 1 && وزن->ne[2] == 1 && وزن->ne[3] == 1);
        ثابت منطقی is_same_shape = (src0->ne[0] == وزن->ne[0] && src0->ne[1] == وزن->ne[1] &&
                                    src0->ne[2] == وزن->ne[2] && src0->ne[3] == وزن->ne[3]);
        اگر (!is_صف_bcast && !is_same_shape) واپس جھوٹ;

        اگر (!ggml_are_same_shape(src0, گرہ.dst())) {
            واپس جھوٹ;
        }
        اگر (ggml_is_contiguous(src0) != ggml_is_contiguous(گرہ.dst())) {
            واپس جھوٹ;
        }

        ساخت htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, وزن, گرہ.dst(), &new_kparams
        );

        اگر ((size_t) new_kparams.vtcm_سائز > sess->vtcm_سائز) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_سائز, sess->vtcm_سائز);
            واپس جھوٹ;
        }

        اگر (!try_fuse_common(وزن, گرہ.dst())) {
            واپس جھوٹ;
        }

        آخری_گرہ.opcode = HTP_OP_RMS_NORM_MUL;
        آخری_گرہ.name   = "RMS_NORM+MUL";
        آخری_گرہ.inputs.clear();
        آخری_گرہ.inputs.push_back(src0);
        آخری_گرہ.inputs.push_back(وزن);
        آخری_گرہ.outputs.clear();
        آخری_گرہ.outputs.push_back(گرہ.dst());
        آخری_گرہ.fused.push_back(گرہ.گرہ);
        حافظہ_نقل(آخری_گرہ.kernel_params, &new_kparams, حجم(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        حافظہ_نقل(o.kernel_params, &new_kparams, حجم(new_kparams));

        o.src[0] = جوڑو_tensor(src0);
        o.src[1] = جوڑو_tensor(وزن);
        برائے (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = جوڑو_tensor(گرہ.dst());
        برائے (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        واپس سچ;
    }

    منطقی try_fuse_mul_mat_جوڑو(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0) واپس جھوٹ;
        اگر (گرہ.opcode != HTP_OP_جوڑو) واپس جھوٹ;

        htp_opnode & آخری_گرہ = ops[n_ops - 1];
        اگر (آخری_گرہ.opcode != HTP_OP_MUL_MAT) واپس جھوٹ;

        ثابت ggml_tensor * جوڑو_src0 = گرہ.src0();
        ثابت ggml_tensor * جوڑو_src1 = گرہ.src1();
        ثابت ggml_tensor * mm_برآمد   = آخری_گرہ.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(mm_برآمد)) واپس جھوٹ;

        ثابت ggml_tensor * src2;
        اگر (جوڑو_src0 == mm_برآمد || جوڑو_src0->ڈیٹا == mm_برآمد->ڈیٹا) {
            src2 = جوڑو_src1;
        } ورنہ اگر (جوڑو_src1 == mm_برآمد || جوڑو_src1->ڈیٹا == mm_برآمد->ڈیٹا) {
            src2 = جوڑو_src0;
        } ورنہ {
            واپس جھوٹ;
        }

        ثابت ggml_tensor * src0 = آخری_گرہ.src0();
        ثابت ggml_tensor * src1 = آخری_گرہ.src1();

        اگر (src2->type != GGML_TYPE_F32) واپس جھوٹ;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        اگر (!ggml_are_same_shape(mm_برآمد, گرہ.dst())) واپس جھوٹ;

        ثابت ساخت htp_mm_kernel_params * orig_kparams = (ثابت ساخت htp_mm_kernel_params *) آخری_گرہ.kernel_params;
        ساخت htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_جوڑو_params(sess, src0, src1, src2, گرہ.dst(), &kparams);
        اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            واپس جھوٹ;
        }

        اگر (orig_kparams->collapse != kparams.collapse) {
            واپس جھوٹ;
        }

        ثابت صحیح src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ثابت منطقی can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        اگر (!can_fuse) واپس جھوٹ;

        اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
            واپس جھوٹ;
        }

        اگر (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            اگر (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_دھاگے < orig_kparams->n_act_دھاگے) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_دھاگے, kparams.n_act_دھاگے);
                واپس جھوٹ;
            }
        }

        اگر (!try_fuse_common(src2, گرہ.dst())) {
            واپس جھوٹ;
        }

        آخری_گرہ.opcode = HTP_OP_MUL_MAT_جوڑو;
        آخری_گرہ.name   = "MUL_MAT+ADD";
        آخری_گرہ.inputs.clear();
        آخری_گرہ.inputs.push_back(src0);
        آخری_گرہ.inputs.push_back(src1);
        آخری_گرہ.inputs.push_back(src2);
        آخری_گرہ.outputs.clear();
        آخری_گرہ.outputs.push_back(گرہ.dst());
        آخری_گرہ.fused.push_back(گرہ.گرہ);
        حافظہ_نقل(آخری_گرہ.kernel_params, &kparams, حجم(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_جوڑو;
        حافظہ_نقل(o.kernel_params, &kparams, حجم(kparams));

        o.src[0] = جوڑو_tensor(src0);
        o.src[1] = جوڑو_tensor(src1);
        o.src[2] = جوڑو_tensor(src2);
        برائے (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = جوڑو_tensor(گرہ.dst());
        برائے (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        واپس سچ;
    }

    منطقی try_fuse_mul_mat_nx(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0 || گرہ.opcode != HTP_OP_MUL_MAT) واپس جھوٹ;
        اگر (!is_mergeable_mul_mat(گرہ.گرہ)) واپس جھوٹ;

        ثابت ggml_tensor * w_اندر = گرہ.src0();
        ثابت ggml_tensor * x_اندر = گرہ.src1();
        ثابت ggml_tensor * d_اندر = گرہ.dst();

        htp_opnode & آخری_گرہ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        اگر (آخری_گرہ.opcode == HTP_OP_MUL_MAT_NX) {
            ثابت uint32_t curr_n = (uint32_t) آخری_گرہ.outputs.size();
            اگر (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                واپس جھوٹ;
            }

            ثابت ggml_tensor * w0 = آخری_گرہ.inputs[0];
            ثابت ggml_tensor * x  = آخری_گرہ.inputs[curr_n];

            اگر (x_اندر != x || w_اندر->type != w0->type || w_اندر->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_اندر) != ggml_hexagon_tensor_is_tiled(w0)) {
                واپس جھوٹ;
            }
            اگر (!آخری_گرہ.fused.خالی_خانے() && (mm_is_hmx_eligible(آخری_گرہ.fused[0]) != mm_is_hmx_eligible(گرہ.گرہ))) {
                واپس جھوٹ;
            }

            ثابت ساخت htp_mm_kernel_params * orig_kparams = (ثابت ساخت htp_mm_kernel_params *) آخری_گرہ.kernel_params;
            ثابت منطقی collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_اندر, x, d_اندر);
            اگر (orig_kparams->collapse && !collapse) {
                واپس جھوٹ;
            }

            ساخت htp_mm_kernel_params kparams;
            اگر (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_صفیں(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } ورنہ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            اگر (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                واپس جھوٹ;
            }
            اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
                واپس جھوٹ;
            }

            اگر (!try_fuse_common(w_اندر, d_اندر)) {
                واپس جھوٹ;
            }

            آخری_گرہ.inputs[curr_n] = w_اندر;
            آخری_گرہ.inputs.push_back(x);
            آخری_گرہ.outputs.push_back(d_اندر);
            آخری_گرہ.fused.push_back(گرہ.گرہ);
            حافظہ_نقل(آخری_گرہ.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            حافظہ_نقل(o.kernel_params, &kparams, حجم(kparams));

            برائے (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = جوڑو_tensor(آخری_گرہ.inputs[s]);
            }
            برائے (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            برائے (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = جوڑو_tensor(آخری_گرہ.outputs[d]);
            }
            برائے (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            واپس سچ;
        }

        // Case 2: last_node is single MUL_MAT
        اگر (آخری_گرہ.opcode == HTP_OP_MUL_MAT) {
            اگر (!is_mergeable_mul_mat_pair(آخری_گرہ.گرہ, گرہ.گرہ)) {
                واپس جھوٹ;
            }

            ثابت ggml_tensor * w0 = آخری_گرہ.src0();
            ثابت ggml_tensor * x  = آخری_گرہ.src1();
            ثابت ggml_tensor * w1 = گرہ.src0();
            ثابت ggml_tensor * dst_0 = آخری_گرہ.dst();
            ثابت ggml_tensor * dst_1 = گرہ.dst();

            ثابت ساخت htp_mm_kernel_params * orig_kparams = (ثابت ساخت htp_mm_kernel_params *) آخری_گرہ.kernel_params;
            ثابت منطقی collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            اگر (orig_kparams->collapse && !collapse) {
                واپس جھوٹ;
            }

            ساخت htp_mm_kernel_params kparams;
            اگر (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_صفیں(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } ورنہ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            اگر (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                واپس جھوٹ;
            }
            اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
                واپس جھوٹ;
            }

            اگر (!try_fuse_common(w1, گرہ.dst())) {
                واپس جھوٹ;
            }

            آخری_گرہ.opcode = HTP_OP_MUL_MAT_NX;
            آخری_گرہ.name   = "MUL_MAT_NX";
            آخری_گرہ.inputs.clear();
            آخری_گرہ.inputs.push_back(w0);
            آخری_گرہ.inputs.push_back(w1);
            آخری_گرہ.inputs.push_back(x);
            آخری_گرہ.outputs.clear();
            آخری_گرہ.outputs.push_back(dst_0);
            آخری_گرہ.outputs.push_back(dst_1);
            آخری_گرہ.fused.push_back(گرہ.گرہ);
            حافظہ_نقل(آخری_گرہ.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            حافظہ_نقل(o.kernel_params, &kparams, حجم(kparams));

            o.src[0] = جوڑو_tensor(w0);
            o.src[1] = جوڑو_tensor(w1);
            o.src[2] = جوڑو_tensor(x);
            برائے (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = جوڑو_tensor(dst_0);
            o.dst[1] = جوڑو_tensor(dst_1);
            برائے (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            واپس سچ;
        }

        واپس جھوٹ;
    }

    منطقی try_fuse_mul_mat_id_nx(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0 || گرہ.opcode != HTP_OP_MUL_MAT_ID) واپس جھوٹ;
        اگر (!is_mergeable_mul_mat_id(گرہ.گرہ)) واپس جھوٹ;

        ثابت ggml_tensor * w_اندر   = گرہ.src0();
        ثابت ggml_tensor * x_اندر   = گرہ.src1();
        ثابت ggml_tensor * ids_اندر = گرہ.گرہ->src[2];
        ثابت ggml_tensor * d_اندر   = گرہ.dst();

        htp_opnode & آخری_گرہ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        اگر (آخری_گرہ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ثابت uint32_t curr_n = (uint32_t) آخری_گرہ.outputs.size();
            اگر (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                واپس جھوٹ;
            }

            ثابت ggml_tensor * w0  = آخری_گرہ.inputs[0];
            ثابت ggml_tensor * x   = آخری_گرہ.inputs[curr_n];
            ثابت ggml_tensor * ids = آخری_گرہ.inputs[curr_n + 1];

            اگر (x_اندر != x || ids_اندر != ids || w_اندر->type != w0->type || w_اندر->ne[0] != w0->ne[0] || w_اندر->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_اندر) != ggml_hexagon_tensor_is_tiled(w0)) {
                واپس جھوٹ;
            }
            اگر (!آخری_گرہ.fused.خالی_خانے() && (mm_is_hmx_eligible(آخری_گرہ.fused[0]) != mm_is_hmx_eligible(گرہ.گرہ))) {
                واپس جھوٹ;
            }

            ساخت htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_اندر, curr_n + 1, &kparams);
            اگر (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                واپس جھوٹ;
            }
            اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
                واپس جھوٹ;
            }

            اگر (!try_fuse_common(w_اندر, d_اندر)) {
                واپس جھوٹ;
            }

            آخری_گرہ.inputs[curr_n] = w_اندر;
            آخری_گرہ.inputs[curr_n + 1] = x;
            آخری_گرہ.inputs.push_back(ids);
            آخری_گرہ.outputs.push_back(d_اندر);
            آخری_گرہ.fused.push_back(گرہ.گرہ);
            حافظہ_نقل(آخری_گرہ.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            حافظہ_نقل(o.kernel_params, &kparams, حجم(kparams));

            برائے (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = جوڑو_tensor(آخری_گرہ.inputs[s]);
            }
            برائے (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            برائے (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = جوڑو_tensor(آخری_گرہ.outputs[d]);
            }
            برائے (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            واپس سچ;
        }

        // Case 2: last_node is single MUL_MAT_ID
        اگر (آخری_گرہ.opcode == HTP_OP_MUL_MAT_ID) {
            اگر (!is_mergeable_mul_mat_id_pair(آخری_گرہ.گرہ, گرہ.گرہ)) {
                واپس جھوٹ;
            }

            ثابت ggml_tensor * w0  = آخری_گرہ.src0();
            ثابت ggml_tensor * x   = آخری_گرہ.src1();
            ثابت ggml_tensor * ids = آخری_گرہ.گرہ->src[2];
            ثابت ggml_tensor * w1  = گرہ.src0();

            ساخت htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, گرہ.dst(), 2, &kparams);
            اگر (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                واپس جھوٹ;
            }
            اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
                واپس جھوٹ;
            }

            اگر (!try_fuse_common(w1, گرہ.dst())) {
                واپس جھوٹ;
            }

            ثابت ggml_tensor * dst_0 = آخری_گرہ.dst();
            ثابت ggml_tensor * dst_1 = گرہ.dst();

            آخری_گرہ.opcode = HTP_OP_MUL_MAT_ID_NX;
            آخری_گرہ.name   = "MUL_MAT_ID_NX";
            آخری_گرہ.inputs.clear();
            آخری_گرہ.inputs.push_back(w0);
            آخری_گرہ.inputs.push_back(w1);
            آخری_گرہ.inputs.push_back(x);
            آخری_گرہ.inputs.push_back(ids);
            آخری_گرہ.outputs.clear();
            آخری_گرہ.outputs.push_back(dst_0);
            آخری_گرہ.outputs.push_back(dst_1);
            آخری_گرہ.fused.push_back(گرہ.گرہ);
            حافظہ_نقل(آخری_گرہ.kernel_params, &kparams, حجم(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            حافظہ_نقل(o.kernel_params, &kparams, حجم(kparams));

            o.src[0] = جوڑو_tensor(w0);
            o.src[1] = جوڑو_tensor(w1);
            o.src[2] = جوڑو_tensor(x);
            o.src[3] = جوڑو_tensor(ids);
            برائے (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = جوڑو_tensor(dst_0);
            o.dst[1] = جوڑو_tensor(dst_1);
            برائے (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            واپس سچ;
        }

        واپس جھوٹ;
    }

    منطقی try_fuse_gdn_cpy(ثابت htp_opnode & گرہ) {
        اگر (n_ops == 0 || گرہ.opcode != HTP_OP_CPY) واپس جھوٹ;

        htp_opnode & آخری_گرہ = ops[n_ops - 1];
        اگر (آخری_گرہ.opcode != HTP_OP_GATED_DELTA_NET) واپس جھوٹ;
        اگر (آخری_گرہ.outputs.size() != 1) واپس جھوٹ;

        ثابت ggml_tensor * gdn_برآمد  = آخری_گرہ.dst();
        ثابت ggml_tensor * cpy_گرہ = گرہ.گرہ;
        ثابت ggml_tensor * cpy_src  = گرہ.src0();
        ثابت ggml_tensor * cpy_dst  = گرہ.dst();

        اگر (!cpy_src || !cpy_dst || !cpy_dst->ڈیٹا) واپس جھوٹ;
        اگر (gdn_برآمد->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) واپس جھوٹ;
        اگر ((gdn_برآمد->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_گرہ->flags & GGML_TENSOR_FLAG_OUTPUT)) واپس جھوٹ;

        ثابت ggml_tensor * v = آخری_گرہ.گرہ->src[2];
        اگر (!v) واپس جھوٹ;

        ثابت int64_t S_v      = v->ne[0];
        ثابت int64_t H        = v->ne[1];
        ثابت int64_t n_علامات = v->ne[2];
        ثابت int64_t n_seqs   = v->ne[3];
        ثابت int64_t K        = ggml_get_op_params_i32(آخری_گرہ.گرہ, 0);
        ثابت size_t  دم_off = (size_t) S_v * H * n_علامات * n_seqs * حجم(اعشاری);

        ثابت int64_t D         = S_v * S_v * H;
        ثابت int64_t n_written = std::min<int64_t>(n_علامات, K);

        اگر (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_برآمد && cpy_src->view_src->ڈیٹا != gdn_برآمد->ڈیٹا) ||
            cpy_src->view_offs != دم_off || !ggml_is_contiguous(cpy_src)) {
            واپس جھوٹ;
        }

        اگر (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != حجم(اعشاری)) {
            واپس جھوٹ;
        }
        اگر (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * حجم(اعشاری)) {
            واپس جھوٹ;
        }
        اگر (n_written > 1) {
            اگر (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * حجم(اعشاری)) {
                واپس جھوٹ;
            }
        }

        اگر (!try_fuse_common({cpy_dst})) {
            واپس جھوٹ;
        }

        آخری_گرہ.name += "+CPY";
        آخری_گرہ.outputs.push_back(cpy_dst);
        آخری_گرہ.fused.push_back(گرہ.گرہ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = جوڑو_tensor(cpy_dst);
        برائے (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        واپس سچ;
    }

    منطقی try_fuse(ثابت htp_opnode & گرہ) {
        اگر (!opt_opfusion) واپس جھوٹ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_جوڑو) && try_fuse_allreduce_جوڑو(گرہ)) واپس سچ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(گرہ))  واپس سچ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_جوڑو)   && try_fuse_mul_mat_جوڑو(گرہ))   واپس سچ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(گرہ))    واپس سچ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(گرہ)) واپس سچ;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(گرہ))       واپس سچ;
        واپس جھوٹ;
    }
};

ساخت ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ساخت ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_بفر *shm_buf;
    size_t                      shm_blk_سائز;
    size_t                      گہرائی;

    استعمال opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       آغاز_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_سائز, size_t گہرائی) : گہرائی(گہرائی) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_سائز;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_سائز = 0;
        اگر (opt_profile == 3) {
            tr_سائز = (HTP_MAX_NTHREADS + 1) * opt_optrace * حجم(htp_trace_desc);
        }

        shm_blk_سائز = حجم(htp_buf_desc)  * n_bufs    +
                       حجم(htp_tensor)    * n_tensors +
                       حجم(htp_op_desc)   * n_ops     +
                       حجم(htp_prof_desc) * n_ops     +
                       tr_سائز;

        shm_buf = نیا ggml_hexagon_shared_بفر(sess, shm_blk_سائز * گہرائی, سچ /* pinned */);

        op_cache.resize(گہرائی);
        آغاز_usec.resize(گہرائی, 0);

        اگر (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_سائز, گہرائی, shm_buf->size(), shm_blk_سائز);
        }
    }

    ~ggml_hexagon_opqueue() {
        حذف shm_buf;
    }

    size_t shm_سائز() ثابت { واپس shm_buf ? shm_buf->size() : 0; }

    // push new batch
    منطقی ڈالو(htp_opbatch_req& req, dspqueue_بفر& dbuf, ثابت ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(حجم(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(حجم(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(حجم(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(حجم(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(حجم(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(حجم(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        اگر (seq - shm_buf->sess->batch_rsp_seq > گہرائی) { واپس جھوٹ; }

        ثابت uint32_t خانہ = (uint32_t) ((seq - 1) % گہرائی);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[خانہ]   = op_batch->ops;
        آغاز_usec[خانہ] = ggml_وقت_us();

        ثابت size_t b_سائز = حجم(htp_buf_desc)  * req.n_bufs;
        ثابت size_t t_سائز = حجم(htp_tensor)    * req.n_tensors;
        ثابت size_t o_سائز = حجم(htp_op_desc)   * req.n_ops;
        ثابت size_t p_سائز = حجم(htp_prof_desc) * req.n_ops;

        size_t tr_سائز = 0;
        اگر (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_سائز = (HTP_MAX_NTHREADS + 1) * req.n_traces * حجم(htp_trace_desc);
        } ورنہ {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->بنیاد() + ((size_t) خانہ * shm_blk_سائز);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_بفر_FLAG_FLUSH_SENDER | DSPQUEUE_بفر_FLAG_INVALIDATE_RECIPIENT;
        dbuf.انحراف   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->بنیاد();
        dbuf.size     = b_سائز + t_سائز + o_سائز + p_سائز + tr_سائز;

        GGML_ASSERT(dbuf.size <= shm_blk_سائز);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_سائز;
        uint8_t * t_ptr = m_ptr; m_ptr += t_سائز;
        uint8_t * o_ptr = m_ptr;

        حافظہ_نقل(b_ptr, (خالی *) op_batch->h_bufs.ڈیٹا(), b_سائز);
        حافظہ_نقل(t_ptr, (خالی *) op_batch->h_tens.ڈیٹا(), t_سائز);
        حافظہ_نقل(o_ptr, (خالی *) op_batch->h_ops.ڈیٹا(),  o_سائز);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (غیرعلامتی طویل طویل) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_سائز, t_سائز, o_سائز, (size_t) dbuf.size);

        اگر (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            برائے (غیرعلامتی صحیح i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (خالی *) b[i].بنیاد, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            برائے (غیرعلامتی صحیح i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (غیرعلامتی طویل طویل) t[i].ڈیٹا, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        واپس سچ;
    }

    خالی نکالو(htp_opbatch_rsp rsp, dspqueue_بفر dbuf) {
        ثابت uint32_t خانہ = (uint32_t) ((rsp.seq - 1) % گہرائی);

        ثابت size_t b_سائز = حجم(htp_buf_desc)  * rsp.n_bufs;
        ثابت size_t t_سائز = حجم(htp_tensor)    * rsp.n_tensors;
        ثابت size_t o_سائز = حجم(htp_op_desc)   * rsp.n_ops;
        ثابت size_t p_سائز = حجم(htp_prof_desc) * rsp.n_ops;

        size_t tr_سائز = 0;
        uint32_t n_traces = 0;
        اگر (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_سائز = (HTP_MAX_NTHREADS + 1) * n_traces * حجم(htp_trace_desc);
        }

        ثابت size_t m_سائز = b_سائز + t_سائز + o_سائز + p_سائز + tr_سائز;
        GGML_ASSERT(m_سائز <= shm_blk_سائز);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (غیرعلامتی طویل طویل) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_سائز, t_سائز, o_سائز);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_سائز + t_سائز + o_سائز);

        اگر (rsp.n_ops > 0) {
            خودکار & ops = op_cache[خانہ];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ثابت htp_prof_desc * pd = (ثابت htp_prof_desc *) p_ptr;
            ثابت htp_trace_desc * trace_events = خالی_اشاریہ;
            اگر (opt_profile == 3) {
                trace_events = (ثابت htp_trace_desc *) (p_ptr + p_سائز);
            }

            اگر (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            برائے (uint32_t i = 0; i < rsp.n_ops; i++) {
                اگر (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            اگر (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

خالی ggml_hexagon_session::flush_peers() {
    خودکار vpeers = std::move(virt_peers);
    virt_peers.clear();
    برائے (خودکار * peer : vpeers) {
        peer->flush_sync();
    }

    خودکار ppeers = std::move(phys_peers);
    phys_peers.clear();
    برائے (خودکار * peer : ppeers) {
        peer->flush_async();
    }

    برائے (خودکار & sub : یہ->mdev.sessions) {
        sub->flush_peers();
    }
}

خالی ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

خالی ggml_hexagon_session::flush_pending(منطقی all) {
    برائے (خودکار & sub : یہ->mdev.sessions) {
        sub->flush_pending(all);
        اگر (sub->آخری_error > HTP_حالت_OK) {
            یہ->آخری_error = sub->آخری_error.load();
        }
    }

    جب (یہ->batch_rsp_seq < یہ->batch_req_seq) {
        ساخت htp_opbatch_rsp rsp;
        uint32_t               rsp_سائز;
        uint32_t               flags;

        ساخت dspqueue_بفر dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ثابت uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        صحیح err = dspqueue_read(یہ->queue, &flags, 1, &n_dbufs, &dbuf, حجم(rsp), &rsp_سائز, (uint8_t *) &rsp, timeo);
        اگر (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            جاری;
        }

        اگر (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (غیرعلامتی) err);
        }

        // Basic sanity checks
        اگر (rsp_سائز != حجم(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", یہ->c_name(), rsp_سائز, n_dbufs);
        }

        اگر (rsp.حالت_2 > HTP_حالت_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", یہ->c_name(), حالت_مقصد_str(rsp.حالت_2));
            یہ->آخری_error = rsp.حالت_2;
            برائے (خودکار & sub : یہ->mdev.sessions) {
                sub->آخری_error = rsp.حالت_2;
            }
        }

        op_قطار->نکالو(rsp, dbuf);

        GGML_ASSERT(rsp.seq == یہ->batch_rsp_seq + 1);
        یہ->batch_rsp_seq = rsp.seq;

        اگر (!all) توڑو;
    }
}

خالی ggml_hexagon_session::flush_sync(منطقی all) {
    flush_async();
    flush_pending(all);
}

خالی ggml_hexagon_session::آغاز_batch() {
    اگر (یہ->mdev.count > 1) {
        قطار_میں_ڈالو_mdev_گروہ();
    }
}

خالی ggml_hexagon_session::flush_batch(size_t min_ops) {
    اگر (op_batch->n_ops < min_ops) { واپس; }

    op_batch->ترتیب_دو_buffers();

    htp_opbatch_req req {};
    dspqueue_بفر dbuf{};

    ثابت uint64_t seq = ++یہ->batch_req_seq;

    op_batch->update_mdev_گروہ(یہ->mdev.idx);

    اگر (!op_قطار->ڈالو(req, dbuf, op_batch, seq)) {
        flush_pending(جھوٹ);
        op_قطار->ڈالو(req, dbuf, op_batch, seq);
    }

    برائے (خودکار & sub : یہ->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_بفر sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_گروہ(sub->mdev.idx);

        اگر (!sub->op_قطار->ڈالو(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(جھوٹ);
            sub->op_قطار->ڈالو(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        صحیح err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, حجم(sub_req), (ثابت uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (غیرعلامتی) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", یہ->c_name(), dbuf.ptr, dbuf.size);

    صحیح err = dspqueue_write(یہ->queue, 0, 1, &dbuf, حجم(req), (ثابت uint8_t*) &req, DSPQUEUE_TIMEOUT);
    اگر (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", یہ->c_name(), (غیرعلامتی) err);
    }

    op_batch->reset();
}

خالی ggml_hexagon_session::قطار_میں_ڈالو_op(ثابت htp_opnode & گرہ) {
    خودکار clone_tensor_بفر = [یہ](ثابت ggml_tensor * t) {
        خودکار sbuf = یہ->mmap_tensor(t);
        اگر (!sbuf) واپس;
        اگر (sbuf->sess != یہ) {
            یہ->clone_بفر(sbuf);
        }
        برائے (خودکار & sub : یہ->mdev.sessions) {
            sub->clone_بفر(sbuf);
        }
    };

    برائے (خودکار t : گرہ.get_inputs()) {
        clone_tensor_بفر(t);
    }
    برائے (خودکار t : گرہ.get_outputs()) {
        clone_tensor_بفر(t);
    }

    اگر (opt_opfusion && op_batch->try_fuse(گرہ)) {
        واپس;
    }

    اگر (!op_batch->fit_op(گرہ)) {
        flush_async();
    }

    اگر (op_batch->خالی_خانے()) {
        آغاز_batch();
    }

    اگر (!op_batch->fit_op(گرہ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->جوڑو_op(گرہ);
}

خالی ggml_hexagon_session::قطار_میں_ڈالو_mdev_گروہ() {
    htp_opnode گروہ_گرہ(HTP_OP_MDEV_گروہ);

    uint8_t * fence_خانہ = یہ->mdev_fence_خانہ;

    ساکن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.بفر       = &یہ->fence_buf->backend_بفر;
    dummy_t.extra        = &fence_extra;
    dummy_t.ڈیٹا         = (خالی *) fence_خانہ;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_خانہ_سائز;
    dummy_t.ne[1]        = (int64_t) یہ->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_خانہ_سائز;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) یہ->mdev.idx;

    ggml_tensor * گرہ = گروہ_گرہ.جوڑو_dummy(dummy_t);
    گرہ->src[0] = گرہ;
    گروہ_گرہ.init(گرہ);
    گروہ_گرہ.outputs.clear();
    گروہ_گرہ.name = "MDEV_GROUP";

    برائے (خودکار & sub : یہ->mdev.sessions) {
        sub->clone_بفر(یہ->fence_buf);
    }

    op_batch->جوڑو_op(گروہ_گرہ);
}

خالی ggml_hexagon_session::قطار_میں_ڈالو_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ثابت منطقی with_fence = sync_tensor != خالی_اشاریہ;
    htp_opnode cpy_گرہ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* گرہ = cpy_گرہ.جوڑو_dummy(*dst);
    گرہ->op     = GGML_OP_CPY;
    گرہ->src[0] = const_cast<ggml_tensor *>(src);
    گرہ->src[1] = with_fence ? cpy_گرہ.جوڑو_dummy(*sync_tensor) : خالی_اشاریہ;
    اگر (with_fence) {
        گرہ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_گرہ.init(گرہ);
    اگر (with_fence) {
        cpy_گرہ.name = "CPY+FENCE";
    }
    ثابت منطقی ok = ggml_hexagon_precompute_cpy_params(یہ, گرہ, (ساخت htp_copy_kernel_params *) cpy_گرہ.kernel_params);
    ثابت خودکار * kparams = (ثابت ساخت htp_copy_kernel_params *) cpy_گرہ.kernel_params;
    اگر (ok && !with_fence && kparams->کل_elems == 0) {
        واپس;
    }
    یہ->قطار_میں_ڈالو_op(cpy_گرہ);
}

خالی ggml_hexagon_session::قطار_میں_ڈالو_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq, منطقی wait) {
    htp_opnode sync_گرہ(HTP_OP_FENCE);

    ggml_tensor* گرہ = sync_گرہ.جوڑو_dummy(*sync_tensor);
    گرہ->op           = GGML_OP_NONE;
    گرہ->src[0]       = گرہ;
    گرہ->op_params[0] = (int32_t) fence_seq;
    گرہ->op_params[1] = wait ? 0 : 1;

    sync_گرہ.init(گرہ);
    sync_گرہ.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    یہ->قطار_میں_ڈالو_op(sync_گرہ);
}

ساکن منطقی ggml_hexagon_precompute_allreduce_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقی has_جوڑو,
    منطقی is_صف_bcast,
    منطقی is_shard_ok,
    ساخت htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_صف_bcast = (has_جوڑو && is_صف_bcast) ? 1 : 0;

    ثابت uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ثابت uint32_t elem_سائز = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(اعشاری);
    ثابت منطقی is_contiguous = ggml_is_contiguous(dst);

    ثابت uint32_t ne0 = (uint32_t) dst->ne[0];
    ثابت uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ثابت منطقی use_1d = is_contiguous && !(has_جوڑو && is_صف_bcast && ne1 > 1);

    اگر (has_جوڑو) {
        // sharded reduce-scatter for contiguous in-place add
        اگر (opt_ar_scatter && is_shard_ok && use_1d && !is_صف_bcast) {
            ثابت uint32_t rank_chunk_elems = hex_باری_اوپر((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_آغاز  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_انتہا    = (std::min)(rank_elem_آغاز + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_آغاز = (int32_t) rank_elem_آغاز;
            kparams->rank_nelem      = (int32_t) (rank_elem_انتہا - rank_elem_آغاز);
        } ورنہ اگر (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_آغاز = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } ورنہ {
            kparams->n_dsts          = 1;
            kparams->rank_elem_آغاز = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } ورنہ {
        kparams->n_dsts = (int32_t) n_ranks;
        اگر (use_1d) {
            ثابت uint32_t rank_chunk_elems = hex_باری_اوپر((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_آغاز  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_انتہا    = (std::min)(rank_elem_آغاز + rank_chunk_elems, nelem);
            ثابت uint32_t rank_nelem       = rank_elem_انتہا - rank_elem_آغاز;
            kparams->rank_elem_آغاز        = (int32_t) rank_elem_آغاز;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } ورنہ {
            ثابت uint32_t rank_chunk_صفیں = (ne1 + n_ranks - 1) / n_ranks;
            ثابت uint32_t rank_r0         = (std::min)(rank * rank_chunk_صفیں, ne1);
            ثابت uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_صفیں, ne1);
            ثابت uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_آغاز       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_بھرا;

    اگر (use_1d) {
        ثابت uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, (std::max)(1u, rank_nelem / 128));
        kparams->n_دھاگے = n_دھاگے;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_گنتی(n_ranks, n_دھاگے, has_جوڑو, is_صف_bcast);

        uint32_t بلاک_elems = 65536;
        اگر (بلاک_elems > rank_nelem / n_دھاگے && rank_nelem / n_دھاگے > 128) {
            بلاک_elems = hex_باری_اوپر(rank_nelem / (n_دھاگے * 2), 128);
        }
        بلاک_elems = (std::max)(128u, بلاک_elems);

        kparams->بلاک_elems          = بلاک_elems;
        kparams->vtcm_سائز_فی_thread = 2 * بلاک_elems * elem_سائز;
        kparams->vtcm_سائز            = n_vtcm_buffers * kparams->vtcm_سائز_فی_thread;

        جب ((size_t) kparams->vtcm_سائز > sess->vtcm_سائز && بلاک_elems > 128) {
            ثابت size_t max_بائٹیں_فی_buf = sess->vtcm_سائز / (n_vtcm_buffers * 2);
            بلاک_elems = (uint32_t) hex_align_نیچے_کو((size_t) (max_بائٹیں_فی_buf / elem_سائز), 128);
            اگر (بلاک_elems < 128) توڑو;
            kparams->بلاک_elems          = بلاک_elems;
            kparams->vtcm_سائز_فی_thread = 2 * بلاک_elems * elem_سائز;
            kparams->vtcm_سائز            = n_vtcm_buffers * kparams->vtcm_سائز_فی_thread;
        }

        اگر (sess->vtcm_سائز < (size_t) kparams->vtcm_سائز || بلاک_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_سائز, sess->vtcm_سائز);
            واپس جھوٹ;
        }

        kparams->elems_فی_thread = hex_باری_اوپر((rank_nelem + n_دھاگے - 1) / n_دھاگے, بلاک_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        واپس سچ;
    } ورنہ {
        ثابت uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, (std::max)(1u, rank_nrows));
        kparams->n_دھاگے = n_دھاگے;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_گنتی(n_ranks, n_دھاگے, has_جوڑو, is_صف_bcast);

        ثابت uint32_t صف_بائٹیں = ne0 * elem_سائز;
        ثابت uint32_t صف_سائز_aligned = (uint32_t) hex_align_اوپر(صف_بائٹیں, 128);
        kparams->صف_سائز_aligned = صف_سائز_aligned;

        ثابت uint32_t nrows_فی_thread = (rank_nrows + n_دھاگے - 1) / n_دھاگے;
        uint32_t بلاک_صفیں = (std::min)(128u, nrows_فی_thread);
        بلاک_صفیں = (std::max)(1u, بلاک_صفیں);
        kparams->بلاک_elems = بلاک_صفیں;

        kparams->vtcm_سائز_فی_thread = 2 * (بلاک_صفیں * صف_سائز_aligned);
        kparams->vtcm_سائز            = n_vtcm_buffers * kparams->vtcm_سائز_فی_thread;

        جب ((size_t) kparams->vtcm_سائز > sess->vtcm_سائز && بلاک_صفیں > 1) {
            ثابت size_t max_صفیں_فی_buf = sess->vtcm_سائز / (n_vtcm_buffers * 2 * صف_سائز_aligned);
            بلاک_صفیں = (std::max)(1u, (uint32_t) max_صفیں_فی_buf);
            kparams->بلاک_elems          = بلاک_صفیں;
            kparams->vtcm_سائز_فی_thread = 2 * (بلاک_صفیں * صف_سائز_aligned);
            kparams->vtcm_سائز            = n_vtcm_buffers * kparams->vtcm_سائز_فی_thread;
            اگر (max_صفیں_فی_buf == 0) توڑو;
        }

        اگر (sess->vtcm_سائز < (size_t) kparams->vtcm_سائز || بلاک_صفیں < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_سائز, sess->vtcm_سائز);
            واپس جھوٹ;
        }

        kparams->elems_فی_thread = nrows_فی_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        واپس سچ;
    }
}

خالی ggml_hexagon_session::قطار_میں_ڈالو_allreduce(
    ثابت ggml_tensor * dst,
    ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
    ثابت std::vector<ثابت ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_اندراج,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_گرہ(HTP_OP_ALLREDUCE);

    ggml_tensor* گرہ = ar_گرہ.جوڑو_dummy(*dst);
    گرہ->op           = GGML_OP_NONE;
    گرہ->op_params[0] = (int32_t) fence_seq_اندراج;
    گرہ->op_params[1] = (int32_t) fence_seq_exit;

    ar_گرہ.init(گرہ);

    ar_گرہ.inputs.clear();
    برائے (size_t i = 0; i < src_tensors.size(); i++) {
        ar_گرہ.inputs.push_back(src_tensors[i]);
    }
    برائے (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_گرہ.inputs.push_back(ar_گرہ.جوڑو_dummy(*sync_tensors[i]));
    }

    ar_گرہ.outputs.clear();
    برائے (size_t i = 0; i < src_tensors.size(); i++) {
        ar_گرہ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        یہ, dst, rank, n_ranks, جھوٹ, جھوٹ, /*is_shard_ok=*/ جھوٹ,
        (ساخت htp_allreduce_kernel_params *) ar_گرہ.kernel_params);

    ar_گرہ.name = "ALLREDUCE";
    یہ->قطار_میں_ڈالو_op(ar_گرہ);
}

ggml_hexagon_shared_بفر * ggml_hexagon_session::mmap_tensor(ثابت ggml_tensor * t) {
    اگر (!t) واپس خالی_اشاریہ;

    خودکار sbuf = static_cast<ggml_hexagon_shared_بفر *>(t->بفر->context);
    اگر (!sbuf->mapped) {
        ثابت منطقی is_وزن = ggml_backend_بفر_get_usage(t->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        ثابت منطقی extended  = opt_dma64 && is_وزن;
        sbuf->mmap(extended);
    }
    واپس sbuf;
}

منطقی ggml_hexagon_session::clone_بفر(ثابت ggml_hexagon_shared_بفر * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    اگر (sbuf->sess == یہ) واپس سچ;

    خودکار mem = sbuf->mem;
    صحیح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    اگر (یہ->cloned_buffers.find(fd) != یہ->cloned_buffers.end()) واپس سچ;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", یہ->name.c_str(),
                sbuf->c_name(), sbuf->بنیاد(), sbuf->size(), fd);

    خودکار clone = std::make_unique<ggml_hexagon_shared_بفر>(یہ, *sbuf);
    کوشش {
        clone->mmap(sbuf->extended);
    } پکڑو (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", یہ->c_name(), exc.what());
        واپس جھوٹ;
    }

    یہ->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(یہ);
    واپس سچ;
}

خالی ggml_hexagon_session::release_بفر(ثابت ggml_hexagon_shared_بفر * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    خودکار mem = sbuf->mem;
    صحیح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    خودکار it = یہ->cloned_buffers.find(fd);
    اگر (it != یہ->cloned_buffers.end()) {
        خودکار clone = std::move(it->second);
        یہ->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(یہ);
}

خالی ggml_hexagon_session::unclone_بفر(ثابت ggml_hexagon_shared_بفر * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    خودکار mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    برائے (خودکار * sess : sessions) {
        sess->release_بفر(sbuf);
    }
}

ساکن size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_بفر *> sbufs;

    ثابت size_t MiB = 1024 * 1024;
    ثابت size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t قدم = 256u * MiB;

    کوشش {
        sbufs.push_back(نیا ggml_hexagon_shared_بفر(sess, GiB, سچ)); vmem += GiB;
        sbufs.push_back(نیا ggml_hexagon_shared_بفر(sess, GiB, سچ)); vmem += GiB;
        sbufs.push_back(نیا ggml_hexagon_shared_بفر(sess, GiB, سچ)); vmem += GiB;

        جب (1) {
            sbufs.push_back(نیا ggml_hexagon_shared_بفر(sess, قدم, سچ));
            vmem += قدم;
        }
    } پکڑو (...) { }

    برائے (خودکار b : sbufs) { حذف b; }

    واپس vmem - قدم; // backoff to account for overhead from internal mappings
}

خالی ggml_hexagon_session::allocate(ثابت ggml_hexagon_device_config & config) noexcept(جھوٹ) {
    صحیح phys_idx = config.physical_idx;
    صحیح virt_idx = config.virtual_idx;

    یہ->valid_session = جھوٹ;
    یہ->valid_handle  = جھوٹ;
    یہ->valid_قطار   = جھوٹ;
    یہ->valid_iface   = جھوٹ;

    یہ->name          = config.name;
    یہ->phys_idx      = phys_idx;
    یہ->virt_idx      = virt_idx;
    یہ->domain_id     = config.domain_id;
    یہ->session_id    = 0;
    یہ->batch_req_seq = 0;
    یہ->batch_rsp_seq = 0;
    یہ->آخری_error    = HTP_حالت_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", یہ->name.c_str(), یہ->domain_id, phys_idx, virt_idx);

    اگر (config.domain_id < 0 || config.domain_name.خالی_خانے()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        پھینکو std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ثابت std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    اگر (virt_idx > 0) {
        ساخت remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<حرف *>(dom_name.c_str());
        n.session_name     = const_cast<حرف *>(یہ->name.c_str());
        n.session_name_len = یہ->name.size();
        n.session_id       = virt_idx;

        صحیح err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (خالی *) &n, حجم(n));
        اگر (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           یہ->c_name(), phys_idx, virt_idx, err);
            پھینکو std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        یہ->session_id    = n.session_id;
        یہ->domain_id     = n.effective_domain_id;
        یہ->valid_session = سچ;
    } ورنہ {
        ساخت remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<حرف *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        صحیح err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (خالی *) &eff, حجم(eff));
        اگر (err == AEE_SUCCESS) {
            یہ->domain_id = eff.effective_domain_id;
        } ورنہ {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           یہ->name.c_str(), err, یہ->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ساخت remote_rpc_control_unsigned_module u;
        u.domain = یہ->domain_id;
        u.enable = 1;
        صحیح err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (خالی *) &u, حجم(u));
        اگر (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", یہ->c_name(), err);
            پھینکو std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    حرف session_uri[256];
    {
        حرف htp_uri[256];
        snprintf(htp_uri, حجم(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ساخت remote_rpc_get_uri u = {};
        u.session_id      = یہ->session_id;
        u.domain_name     = const_cast<حرف *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<حرف *>(htp_uri);
        u.module_uri_len  = لڑی_لمبائی(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = حجم(session_uri);

        صحیح err = remote_session_control(FASTRPC_GET_URI, (خالی *) &u, حجم(u));
        اگر (err != AEE_SUCCESS) {
            snprintf(session_uri, حجم(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), یہ->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          یہ->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    صحیح err = htp_iface_open(session_uri, &یہ->handle);
    اگر (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", یہ->c_name(), session_uri, err);
        پھینکو std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    یہ->valid_handle = سچ;

    // Query HW info and resolve session options
    {
        غیرعلامتی صحیح hw_n_دھاگے = 0;
        غیرعلامتی صحیح hw_n_hvx     = 0;
        غیرعلامتی صحیح hw_n_hmx     = 0;
        غیرعلامتی طویل طویل hw_vtcm_سائز = 0;
        صحیح hw_err = htp_iface_hwinfo(یہ->handle, &hw_n_دھاگے, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_سائز);
        اگر (hw_err == 0) {
            ثابت uint32_t max_n_دھاگے = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_دھاگے);
            یہ->n_دھاگے = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_دھاگے) : max_n_دھاگے;
            یہ->n_hvx     = یہ->n_دھاگے;
            یہ->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            یہ->vtcm_سائز = (uint64_t)hw_vtcm_سائز;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          یہ->c_name(), یہ->n_دھاگے, یہ->n_hvx, یہ->n_hmx,
                          (غیرعلامتی طویل طویل)(یہ->vtcm_سائز / (1024 * 1024)));
        } ورنہ {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", یہ->c_name(), hw_err);
            ثابت uint32_t default_n_دھاگے = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            یہ->n_دھاگے = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_دھاگے;
            یہ->n_hvx     = یہ->n_دھاگے;
            یہ->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            یہ->vtcm_سائز = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ساخت remote_rpc_control_latency l;
        l.enable = 1;

        صحیح err = remote_handle64_control(یہ->handle, DSPRPC_CONTROL_LATENCY, (خالی *) &l, حجم(l));
        اگر (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", یہ->c_name(),
                  یہ->session_id, یہ->domain_id, session_uri, (غیرعلامتی طویل) یہ->handle);

    ثابت size_t req_q_سائز = (حجم(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ثابت size_t rsp_q_سائز = (حجم(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(یہ->domain_id,
                          0,              // Flags
                          req_q_سائز,     // Request  queue size (in bytes)
                          rsp_q_سائز,     // Response queue size (in bytes)
                          خالی_اشاریہ,        // Read packet callback (we handle reads explicitly)
                          خالی_اشاریہ,        // Error callback (we handle errors during reads)
                          (خالی *) یہ,  // Callback context
                          &queue);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", یہ->name.c_str(), (غیرعلامتی) err);
        پھینکو std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    یہ->valid_قطار = سچ;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &یہ->قطار_id);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (غیرعلامتی) err);
        پھینکو std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    اگر (opt_etm) {
        err = htp_iface_etm(یہ->handle, 1);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    // Allocate buffers and state for op batching
    یہ->op_قطار = نیا ggml_hexagon_opqueue(یہ, opt_opbatch, opt_opqueue);

    یہ->fence_buf = نیا ggml_hexagon_fence_بفر(یہ, &dev_ctx->fence_بفر_type, 64 * 1024);
    اگر (یہ->mdev.count > 1) {
        یہ->mdev_fence_خانہ = یہ->alloc_fence(یہ->mdev.count);
    }

    اگر (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(یہ);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", یہ->c_name(), opt_vmem);
    }
    ثابت size_t shm_سائز = یہ->op_قطار->shm_سائز();
    یہ->max_vmem = (opt_vmem > shm_سائز) ? (opt_vmem - shm_سائز) : opt_vmem;

    یہ->op_batch = نیا ggml_hexagon_opbatch(یہ, opt_opbatch, یہ->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_آغاز(یہ->handle, یہ->session_id, یہ->قطار_id, یہ->n_دھاگے, opt_nhmx, یہ->max_vmem, (uint32_t) opt_power_منوال);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", یہ->c_name(), (غیرعلامتی) err);
        پھینکو std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    یہ->valid_iface = سچ;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", یہ->c_name(), htp_power_منوال_مقصد_str(opt_power_منوال));

    اگر (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(یہ->handle, opt_profile, &pmu_conf);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (غیرعلامتی) err);
        }
    }
}

خالی ggml_hexagon_session::release() noexcept(سچ) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", یہ->name.c_str());

    یہ->mdev.sessions.clear();

    صحیح err;

    اگر (یہ->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(یہ->handle);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    حذف یہ->op_batch;
    حذف یہ->op_قطار;
    برائے (خودکار & it : یہ->cpy_fence_slots) {
        free_fence((خالی *) it.second, 1);
    }
    یہ->cpy_fence_slots.clear();

    اگر (یہ->fence_buf) {
        unclone_بفر(یہ->fence_buf);
        حذف یہ->fence_buf;
        یہ->fence_buf = خالی_اشاریہ;
    }
    جب (!یہ->cloned_buffers.خالی_خانے()) {
        release_بفر(یہ->cloned_buffers.begin()->second.get());
    }

    اگر (opt_etm) {
        err = htp_iface_etm(یہ->handle, 0);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    اگر (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(یہ->handle, 0, &pmu_conf);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    اگر (یہ->valid_قطار) {
        err = dspqueue_close(queue);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (غیرعلامتی) err);
        }
    }

    اگر (یہ->valid_handle) {
        htp_iface_close(یہ->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_گنتی) noexcept(جھوٹ) {
    یہ->dev        = dev;
    یہ->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    یہ->mdev.idx   = mdev_idx;
    یہ->mdev.count = mdev_گنتی > 0 ? mdev_گنتی : (uint32_t) (1 + config.mdev_گروہ.size());
    op_batch         = خالی_اشاریہ;
    op_قطار         = خالی_اشاریہ;
    fence_buf        = خالی_اشاریہ;
    fence_seq        = ((uintptr_t)یہ) & 0xFFFF;

    کوشش {
        allocate(config);
        اگر (یہ->mdev.idx == 0 && !config.mdev_گروہ.خالی_خانے()) {
            برائے (size_t i = 0; i < config.mdev_گروہ.size(); i++) {
                یہ->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_گروہ[i], یہ->dev, (uint32_t) (i + 1), یہ->mdev.count));
            }
        }
    } پکڑو (ثابت std::exception & exc) {
        release();
        پھینکو;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(سچ) {
    release();
}

// ** backend interface

ساکن منطقی ggml_hexagon_flash_attn_is_hmx_eligible(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * q,
    ثابت ساخت ggml_tensor * k,
    ثابت ساخت ggml_tensor * v,
    ثابت ساخت ggml_tensor * sinks
) {
    اگر (sess->n_hmx == 0) {
        واپس جھوٹ;
    }

    اگر (opt_fa_select < 2) {
        واپس جھوٹ;
    }

    اگر ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        واپس جھوٹ;
    }

    ثابت uint32_t DK = q->ne[0];
    ثابت uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    اگر (DK % 8 != 0 || DV % 8 != 0) {
        واپس جھوٹ;
    }

    GGML_غیر_مستعمل(sinks);

    // Explicit force mode
    اگر (opt_fa_select > 2) {
        واپس سچ;
    }

    ثابت uint32_t M = q->ne[1];

    // Prefill or batched decode
    اگر (M > 1) {
        واپس سچ;
    }

    ثابت uint32_t n_سر = q->ne[2];
    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G = n_kv_heads > 0 ? n_سر / n_kv_heads : 1;
    ثابت uint32_t S = k->ne[1];
    ثابت uint32_t کل_دھاگے = sess->n_دھاگے * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    اگر (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < کل_دھاگے))) {
        واپس سچ;
    }

    // Tile alignment for 32-row HMX tiles
    ثابت منطقی is_tile_aligned = (G > 0 && (32 % G == 0));
    اگر (!is_tile_aligned) {
        اگر (sess->n_دھاگے >= 6) {
            واپس جھوٹ;
        }
        واپس S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    اگر (DK == 256) {
        s_cross = 4096;
    } ورنہ اگر (DK <= 64) {
        s_cross = (sess->n_دھاگے >= 8) ? 2048 : ((sess->n_دھاگے >= 6) ? 768 : 512);
    } ورنہ {
        s_cross = (sess->n_دھاگے >= 8) ? 1024 : 512;
    }

    واپس S >= s_cross;
}

ساکن منطقی ggml_hexagon_precompute_flash_attn_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_fa_kernel_params * kparams
) {
    اگر (opt_fa_select < 1) {
        واپس جھوٹ;
    }

    memset(kparams, 0, حجم(*kparams));

    ثابت ساخت ggml_tensor * q    = op->src[0];
    ثابت ساخت ggml_tensor * k    = op->src[1];
    ثابت ساخت ggml_tensor * v    = op->src[2];
    ثابت ساخت ggml_tensor * نقاب = op->src[3];

    ثابت uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ثابت uint32_t neq1 = q->ne[1];  // n_tokens
    ثابت uint32_t neq2 = q->ne[2];  // n_heads

    ثابت uint32_t nek1 = k->ne[1];  // kv_len

    ثابت uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ثابت uint32_t DK = neq0;
    ثابت uint32_t DV = nev0;

    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G          = neq2 / n_kv_heads;

    اعشاری scale         = 1.0f;
    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;
    حافظہ_نقل(&scale,         &op->op_params[0], حجم(اعشاری));
    حافظہ_نقل(&max_bias,      &op->op_params[1], حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, &op->op_params[2], حجم(اعشاری));

    اگر (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->سر_بانٹو = (opt_fa_سر_بانٹو != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ثابت uint32_t n_سر = q->ne[2];
    kparams->n_سر_log2 = 1u << (uint32_t) std::floor(std::log2(n_سر));
    kparams->m0 = std::قوت(2.0f, -(max_bias) / kparams->n_سر_log2);
    kparams->m1 = std::قوت(2.0f, -(max_bias / 2.0f) / kparams->n_سر_log2);

    // Check HMX eligibility
    ثابت ساخت ggml_tensor * sinks = op->src[4];
    اگر (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ثابت uint32_t DK_pad = hex_باری_اوپر(DK, 64);
        ثابت uint32_t DV_pad = hex_باری_اوپر(DV, 64);
        size_t Br = 0, Bc = 0;
        صحیح ret = hmx_fa_ڈھونڈو_chunk_سائز(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_سائز, sess->n_دھاگے, (q->type == GGML_TYPE_F32), sinks != خالی_اشاریہ, n_سر);
        اگر (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            اگر (logit_softcap == 0.0f) {
                kparams->scale = scale * (اعشاری) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } ورنہ {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (اعشاری) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_بلاکیں = (nek1 + Bc - 1) / Bc;
            kparams->n_دھاگے = (kparams->n_kv_بلاکیں >= 3 && sess->n_دھاگے >= 2) ? sess->n_دھاگے : 1;

            kparams->u.hmx.g_br = hex_align_اوپر(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_بلاکیں >= 3 && sess->n_دھاگے >= 2) ? 1 : 0;
            kparams->vtcm_سائز = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_دھاگے, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != خالی_اشاریہ, n_سر);

            ثابت size_t صف_vec_بائٹیں = hex_align_اوپر(Bc * حجم(uint16_t), 256);
            kparams->u.hmx.صف_buf_stride = صف_vec_بائٹیں / 128; // HVX vector is 128 bytes

            ثابت size_t m_سطر_بائٹیں = hex_align_اوپر(Bc * حجم(uint16_t), 128);
            kparams->u.hmx.نقاب_buf_صف_stride = m_سطر_بائٹیں / حجم(uint16_t);
            kparams->u.hmx.نقاب_نشر = (نقاب != خالی_اشاریہ && نقاب->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_اقدار(G);
            اگر (نقاب) {
                kparams->src3_div2 = init_fastdiv_اقدار(نقاب->ne[2]);
                kparams->src3_div3 = init_fastdiv_اقدار(نقاب->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_فی_thread = 0;
            واپس سچ;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_بلاکیں = (k->ne[1] + 64 - 1) / 64;
    kparams->n_دھاگے = sess->n_دھاگے;

    ثابت size_t سائز_q_صف_padded = hex_باری_اوپر(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ثابت size_t سائز_k_صف_padded = hex_باری_اوپر(k->ne[0] * 2, 128);
    ثابت size_t سائز_v_صف_padded = hex_باری_اوپر(v->ne[0] * 2, 128);

    kparams->vtcm_سائز = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), نقاب != خالی_اشاریہ, sinks != خالی_اشاریہ, n_سر, sess->n_دھاگے);

    kparams->u.hvx.سائز_q_صف_padded = سائز_q_صف_padded;
    kparams->u.hvx.سائز_k_صف_padded = سائز_k_صف_padded;
    kparams->u.hvx.سائز_v_صف_padded = سائز_v_صف_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_اقدار(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_اقدار(q->ne[1]);
    kparams->نشر_rk2 = init_fastdiv_اقدار(q->ne[2]/k->ne[2]);
    kparams->نشر_rk3 = init_fastdiv_اقدار(q->ne[3]/k->ne[3]);
    kparams->نشر_rv2 = init_fastdiv_اقدار(q->ne[2]/v->ne[2]);
    kparams->نشر_rv3 = init_fastdiv_اقدار(q->ne[3]/v->ne[3]);
    اگر (نقاب) {
        kparams->src3_div2 = init_fastdiv_اقدار(نقاب->ne[2]);
        kparams->src3_div3 = init_fastdiv_اقدار(نقاب->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_فی_thread = (kparams->qrows + sess->n_دھاگے - 1) / sess->n_دھاگے;

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_flash_attn_ext(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * src3 = op->src[3];
    ثابت ساخت ggml_tensor * src4 = op->src[4];
    ثابت ساخت ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    اگر ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        واپس جھوٹ;
    }

    اگر (src3 && src3->type != GGML_TYPE_F16) {  //نقاب
        واپس جھوٹ;
    }

    اگر (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        واپس جھوٹ;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }

    ساخت htp_fa_kernel_params kparams;
    اگر (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        واپس جھوٹ;
    }

    اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_سائز, sess->vtcm_سائز);
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_gated_delta_net(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    اگر (opt_gdn_select < 1) {
        واپس جھوٹ;
    }

    ثابت ساخت ggml_tensor * q     = op->src[0];
    ثابت ساخت ggml_tensor * k     = op->src[1];
    ثابت ساخت ggml_tensor * v     = op->src[2];
    ثابت ساخت ggml_tensor * g     = op->src[3];
    ثابت ساخت ggml_tensor * beta  = op->src[4];
    ثابت ساخت ggml_tensor * state = op->src[5];
    ثابت ساخت ggml_tensor * dst   = op;

    اگر (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous_صفیں(q) || !ggml_is_contiguous_صفیں(k) || !ggml_is_contiguous_صفیں(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    ثابت int64_t S_v      = v->ne[0];
    ثابت int64_t H        = v->ne[1];
    ثابت int64_t n_علامات = v->ne[2];
    ثابت int64_t n_seqs   = v->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(op, 0);

    اگر (S_v <= 0 || S_v > 128 || H <= 0 || n_علامات <= 0 || n_seqs <= 0) {
        واپس جھوٹ;
    }
    اگر (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_علامات || k->ne[2] != n_علامات || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        واپس جھوٹ;
    }
    اگر ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        واپس جھوٹ;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    اگر (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        واپس جھوٹ;
    }
    اگر (dst->ne[0] != S_v * H || dst->ne[1] != n_علامات * n_seqs + S_v * n_seqs * K) {
        واپس جھوٹ;
    }

    ثابت uint32_t کل_صفیں = (uint32_t) (H * n_seqs);
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, کل_صفیں);

    ثابت منطقی can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_علامات >= HTP_GDN_MIN_علامات) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    اگر (can_use_hmx) {
        ساخت htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        اگر (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_سائز, کل_صفیں, sess->vtcm_سائز, n_دھاگے, سچ, &n_heads_batch)) {
            واپس جھوٹ;
        }
    } ورنہ {
        ساخت htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_دھاگے);
        اگر (layout.کل_بائٹیں > sess->vtcm_سائز) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_matmul_is_hmx_eligible(
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    منطقی is_matmul_id,
    منطقی is_batched
) {
    GGML_غیر_مستعمل(dst);

    اگر (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        واپس جھوٹ;
    }

    ثابت صحیح ne00  = src0->ne[0];
    ثابت صحیح ne01  = src0->ne[1];
    ثابت صحیح ne11  = src1->ne[1];
    ثابت صحیح ne12  = src1->ne[2];
    ثابت صحیح wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    اگر (ne01 % HTP_TENSOR_TILE_N_صفیں != 0 && is_matmul_id) {
        واپس جھوٹ;
    }

    // HMX supports F16, F32, and repack quantized types.
    اگر (!htp_is_hmx_وزن_type((ggml_type) wtype)) {
        واپس جھوٹ;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    اگر (ne00 % HTP_TENSOR_TILE_N_ستون_تعداد != 0) {
        واپس جھوٹ;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    اگر (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        واپس جھوٹ;
    }
    اگر (!is_matmul_id && is_batched && is_tiled) {
        واپس جھوٹ;
    }

    // HMX assumes contiguous row-major layout.
    اگر (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        واپس جھوٹ;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ثابت صحیح m = is_matmul_id ? ne12 : ne11;
    اگر (m <= HTP_MM_HMX_MIN_NROWS) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(dst);
}

ساکن منطقی ggml_hexagon_precompute_hmx_mm_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * act,
    ثابت ساخت ggml_tensor * dst,
    صحیح wtype,
    صحیح ne00_padded,
    صحیح ne01_padded,
    صحیح ne02,
    صحیح ne11,
    صحیح ne12,
    صحیح ne11_padded,
    منطقی is_matmul_id,
    منطقی is_batched,
    size_t src2_سائز,
    size_t vtcm_budget,
    ساخت htp_mm_kernel_params * kparams
) {
    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح aligned_tile_سائز = is_tiled ? htp_tensor_get_tile_padded_سائز(wtype) : 0;
    ثابت صحیح n_دھاگے = (صحیح)sess->n_دھاگے;
    ثابت صحیح ne10 = act->ne[0];

    صحیح m_for_solver = ne11;
    صحیح m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    اگر (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ثابت منطقی dst_صف_بانٹو = dst ? ggml_hexagon_can_صف_تقسیم(dst) : سچ;
        ثابت منطقی act_صف_بانٹو = ggml_hexagon_can_صف_تقسیم(act);
        اگر (dst_صف_بانٹو && act_صف_بانٹو) {
            m_for_solver = (ne11 + (صحیح) sess->mdev.count - 1) / (صحیح) sess->mdev.count;
            m_for_solver_padded = hex_باری_اوپر(std::max(m_for_solver, 32), 32);
        }
    }
    ثابت منطقی pipeline = is_matmul_id ? جھوٹ : htp_mm_hmx_pipeline(m_for_solver);

    ثابت منطقی is_batched_val = is_matmul_id ? جھوٹ : is_batched;
    ثابت صحیح گروہ_سائز = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_سائز = 0;
    منطقی use_grouped = جھوٹ;
    صحیح act_دھاگے_selected = 0;

    اگر (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && گروہ_سائز > 1) {
        // Try grouped path first
        اگر (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, گروہ_سائز, n_دھاگے, pipeline, src2_سائز, vtcm_budget, &m_chunk, &n_chunk, &act_دھاگے_selected, &vtcm_سائز)) {
            use_grouped = سچ;
        }
    }

    اگر (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        صحیح m_id_صفیں = 0;
        اگر (dst && is_matmul_id) {
            ثابت صحیح n_experts = ne02 > 0 ? ne02 : 1;
            ثابت size_t کل_expert_صفیں = (size_t) dst->ne[1] * dst->ne[2];
            صحیح m_فی_expert = (صحیح) ((کل_expert_صفیں + n_experts - 1) / n_experts);
            اگر (sess->mdev.count > 1 && ggml_hexagon_can_صف_تقسیم(dst)) {
                m_فی_expert = (m_فی_expert + (صحیح) sess->mdev.count - 1) / (صحیح) sess->mdev.count;
            }
            m_id_صفیں = hex_باری_اوپر(std::max(m_فی_expert, 32), 32);
        }
        ثابت uint32_t لاگت_m = is_matmul_id ? (uint32_t) m_id_صفیں : (uint32_t) m_for_solver;
        اگر (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_صفیں, ne01_padded, m_for_solver_padded, لاگت_m, n_دھاگے, pipeline, is_matmul_id, aligned_tile_سائز, src2_سائز, vtcm_budget, &m_chunk, &n_chunk, &act_دھاگے_selected, &vtcm_سائز)) {
            واپس جھوٹ;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_دھاگے = n_دھاگے;
    kparams->n_act_دھاگے = act_دھاگے_selected;
    kparams->tile_سائز = is_tiled ? htp_tensor_get_tile_سائز(wtype) : 0;
    kparams->aligned_tile_سائز = aligned_tile_سائز;
    kparams->act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
    kparams->vtcm_سائز = vtcm_سائز;
    kparams->vtcm_src0_سائز = 0;
    kparams->div_n_act_دھاگے = init_fastdiv_اقدار(act_دھاگے_selected);
    kparams->div_ne00_padded   = init_fastdiv_اقدار(ne00_padded);
    kparams->vtcm_act_سائز     = 0;
    kparams->vtcm_bias_سائز    = (int32_t) src2_سائز;
    kparams->vtcm_dst_سائز     = 0;

    اگر (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } ورنہ {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    واپس سچ;
}

ساکن خالی ggml_hexagon_precompute_hvx_mm_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    صحیح wtype,
    منطقی is_tiled,
    صحیح ne02,
    صحیح ne03,
    صحیح ne10,
    صحیح ne11,
    صحیح ne12,
    صحیح ne13,
    منطقی is_matmul_id,
    ثابت size_t src2_صف_سائز,
    size_t vtcm_budget,
    ساخت htp_mm_kernel_params * kparams
) {
    اگر (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    }

    kparams->n_hmx = 0;
    kparams->n_دھاگے = sess->n_دھاگے;

    GGML_غیر_مستعمل(ne02);
    GGML_غیر_مستعمل(ne03);

    ثابت صحیح src1_nrows = ne11 * ne12 * ne13;

    اگر (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_سائز = htp_tensor_get_tile_سائز(wtype);
        kparams->aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(wtype);

        ثابت منطقی k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        اگر (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (صحیح) sess->n_دھاگے)
                                  ? HTP_MM_KERNEL_HVX_QUANT_بلاک : HTP_MM_KERNEL_HVX_QUANT_صف;
            kparams->act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);

            ساخت htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t بہترین_n_prefetch = 2;
            برائے (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_دھاگے,
                    0, src0->nb[1], kparams->act_صف_سائز, 0, d, سچ, جھوٹ
                );
                اگر (L.کل_بائٹیں <= vtcm_budget) {
                    بہترین_n_prefetch = d;
                    توڑو;
                }
            }
            اگر (L.کل_بائٹیں > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                واپس;
            }
            kparams->n_prefetch     = بہترین_n_prefetch;
            kparams->vtcm_سائز      = L.کل_بائٹیں;
            kparams->vtcm_src0_سائز = L.src0_بائٹیں;
            kparams->vtcm_act_سائز  = L.act_بائٹیں;
            kparams->vtcm_bias_سائز = 0;
            kparams->vtcm_dst_سائز  = L.dst_بائٹیں;
            واپس;
        } ورنہ {
            منطقی try_tiled = (k_align && opt_mm_select >= 1);
            اگر (try_tiled) {
                kparams->act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
                اگر (src1_nrows < (صحیح) sess->n_دھاگے) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_بلاک;
                } ورنہ {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_صف;
                }

                ساخت htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t بہترین_n_prefetch = 2;
                برائے (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_دھاگے,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_صف_سائز, d, جھوٹ, جھوٹ
                    );
                    اگر (L.کل_بائٹیں <= vtcm_budget) {
                        بہترین_n_prefetch = d;
                        توڑو;
                    }
                }

                uint32_t m_chunk = 0;
                اگر (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_دھاگے,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_صف_سائز, بہترین_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = بہترین_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_سائز      = L.کل_بائٹیں;
                    kparams->vtcm_src0_سائز = L.src0_بائٹیں;
                    kparams->vtcm_act_سائز  = L.act_بائٹیں;
                    kparams->vtcm_bias_سائز = L.bias_بائٹیں;
                    kparams->vtcm_dst_سائز  = L.dst_بائٹیں;
                    واپس;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            واپس;
        }
    } ورنہ اگر (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    } ورنہ اگر (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ساخت htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_دھاگے,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_صف_سائز, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_صف_سائز = hex_باری_اوپر(ne10 * 2, 128);
            kparams->vtcm_سائز = L.کل_بائٹیں;
            kparams->vtcm_src0_سائز = L.src0_بائٹیں;
            kparams->vtcm_act_سائز  = L.act_بائٹیں;
            kparams->vtcm_bias_سائز = L.bias_بائٹیں;
            kparams->vtcm_dst_سائز = L.dst_بائٹیں;
            kparams->n_prefetch = 16;
            واپس;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    } ورنہ اگر (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ساخت htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_دھاگے,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_صف_سائز, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_صف_سائز = hex_باری_اوپر(ne10 * 4, 128);
            kparams->vtcm_سائز = L.کل_بائٹیں;
            kparams->vtcm_src0_سائز = L.src0_بائٹیں;
            kparams->vtcm_act_سائز  = L.act_بائٹیں;
            kparams->vtcm_bias_سائز = L.bias_بائٹیں;
            kparams->vtcm_dst_سائز = L.dst_بائٹیں;
            kparams->n_prefetch = 16;
            واپس;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    } ورنہ اگر (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ساخت htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_دھاگے,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_صف_سائز, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_صف_سائز = hex_باری_اوپر(ne10 * 4, 128);
            kparams->vtcm_سائز = L.کل_بائٹیں;
            kparams->vtcm_src0_سائز = L.src0_بائٹیں;
            kparams->vtcm_act_سائز  = L.act_بائٹیں;
            kparams->vtcm_bias_سائز = L.bias_بائٹیں;
            kparams->vtcm_dst_سائز = L.dst_بائٹیں;
            kparams->n_prefetch = 16;
            واپس;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    } ورنہ {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    }
}

ساکن خالی ggml_hexagon_precompute_matmul_params_impl(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ثابت size_t src2_صف_سائز,
    ثابت size_t src2_سائز,
    ساخت htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت صحیح ne00 = src0->ne[0];
    ثابت صحیح ne01 = src0->ne[1];
    ثابت صحیح ne02 = src0->ne[2];
    ثابت صحیح ne03 = src0->ne[3];

    ثابت صحیح ne10 = src1->ne[0];
    ثابت صحیح ne11 = src1->ne[1];
    ثابت صحیح ne12 = src1->ne[2];
    ثابت صحیح ne13 = src1->ne[3];

    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحیح ne00_padded = is_tiled ? hex_باری_اوپر(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ثابت صحیح ne11_padded = hex_باری_اوپر(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ثابت صحیح ne01_tiled  = hex_باری_اوپر(ne01, 32);

    ثابت منطقی is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقی is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ثابت size_t vtcm_budget = sess->vtcm_سائز;

    // Check HMX eligibility and try precomputing HMX parameters
    منطقی hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    اگر (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        اگر (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_سائز, vtcm_budget, kparams)) {
            جاؤ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_صف_سائز, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_اقدار(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_اقدار(ne11);
    kparams->div_r2       = init_fastdiv_اقدار(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_اقدار(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_اقدار(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ساکن منطقی ggml_hexagon_صفیں_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride) {
    size_t s = 0, اگلا = 0;
    برائے (صحیح i = 1; i < GGML_MAX_DIMS; i++) {
        اگر (ne[i] == 1) جاری;
        اگر (s == 0) { s = nb[i]; اگلا = s * ne[i]; جاری; }
        اگر (nb[i] != اگلا) واپس جھوٹ;
        اگلا *= ne[i];
    }
    *stride = s ? s : nb[1];
    واپس سچ;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ساکن منطقی ggml_hexagon_matmul_can_collapse(ثابت ساخت ggml_tensor * src0, ثابت ساخت ggml_tensor * src1, ثابت ساخت ggml_tensor * dst) {
    size_t s1, sd;
    واپس (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_جوڑو) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_سائز(src1->type) && ggml_hexagon_صفیں_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_سائز(dst->type) && ggml_hexagon_صفیں_stride(dst->ne, dst->nb, &sd);
}

ساکن منطقی ggml_hexagon_matmul_جوڑو_can_collapse(
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * src2,
    ثابت ساخت ggml_tensor * dst
) {
    اگر (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        واپس جھوٹ;
    }
    اگر (!src2) {
        واپس سچ;
    }
    ثابت int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    اگر (src2_nrows == 1) {
        واپس سچ;
    }
    size_t s2;
    واپس src2->nb[0] == ggml_type_سائز(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_صفیں_stride(src2->ne, src2->nb, &s2);
}

ساکن ggml_tensor ggml_hexagon_tensor_collapse_صفیں(ثابت ساخت ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_صفیں_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    واپس c;
}

ساکن خالی ggml_hexagon_precompute_matmul_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_mm_kernel_params * kparams
) {
    اگر (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_صفیں(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_صفیں(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        واپس;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ساکن خالی ggml_hexagon_precompute_fused_matmul_جوڑو_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * src2,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_mm_kernel_params * kparams
) {
    اگر (ggml_hexagon_matmul_جوڑو_can_collapse(src0, src1, src2, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_صفیں(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_صفیں(dst);
        ثابت ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_صفیں(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ثابت ساخت ggml_tensor * p_src2 = src2 ? &src2_collapsed : خالی_اشاریہ;
        ثابت size_t src2_سائز = p_src2 ? hex_باری_اوپر(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_سائز, kparams);
        kparams->collapse = 1;
        واپس;
    }
    ثابت size_t src2_سائز = src2 ? hex_باری_اوپر(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_سائز, kparams);
}

ساکن منطقی ggml_hexagon_precompute_ثنائی_params(
    ثابت ساخت ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_ثنائی_kernel_params * kparams,
    ثابت ساخت ggml_tensor * src2
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت size_t elem_سائز = ggml_type_سائز(src0->type);
    ثابت size_t src0_صف_سائز = src0->ne[0] * elem_سائز;
    ثابت size_t src1_صف_سائز = src1->ne[0] * elem_سائز;
    ثابت size_t dst_صف_سائز  = dst->ne[0]  * elem_سائز;

    ثابت size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_صف_سائز, 128);
    ثابت size_t src1_صف_سائز_aligned = hex_باری_اوپر(src1_صف_سائز, 128);
    ثابت size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_صف_سائز,  128);

    اگر (!htp_ثنائی_strides_محفوظ(src0, src1, dst, elem_سائز)) {
        واپس جھوٹ;
    }

    ثابت منطقی is_جوڑو_id = op == HTP_OP_جوڑو_ID;
    ثابت منطقی is_scalar = !is_جوڑو_id && src1->ne[0] == 1;
    ثابت منطقی is_صف_bcast = !is_جوڑو_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ثابت منطقی is_same_shape = !is_جوڑو_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ثابت منطقی is_complex   = !is_جوڑو_id && !is_scalar && !is_same_shape && !is_صف_bcast && (src1->ne[0] == src0->ne[0]);
    ثابت منطقی is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ثابت منطقی is_scalar_نشر = !is_جوڑو_id && (ggml_nelements(src1) == 1);

    اگر (!is_جوڑو_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_نشر)) {
        ثابت uint32_t کل_elems = (uint32_t) ggml_nelements(src0);
        ثابت uint32_t n_دھاگے = sess->n_دھاگے;
        ثابت uint32_t max_chunk_elems = 32768 / elem_سائز;
        ثابت uint32_t min_chunk_elems = 256;
        ثابت uint32_t ہدف_chunk_elems = hex_باری_اوپر((کل_elems + (2 * n_دھاگے) - 1) / (2 * n_دھاگے), 32);
        ثابت uint32_t chunk_سائز = (std::min)(max_chunk_elems, (std::max)(ہدف_chunk_elems, min_chunk_elems));
        ثابت uint32_t chunk_بائٹیں = hex_باری_اوپر(chunk_سائز * elem_سائز, 128);

        kparams->kernel_type           = HTP_ثنائی_KERNEL_CHUNKED;
        kparams->n_دھاگے             = n_دھاگے;
        kparams->صفیں_فی_بفر       = 1;
        kparams->src0_صف_سائز_aligned = src0_صف_سائز_aligned;
        kparams->src1_صف_سائز_aligned = is_scalar_نشر ? 0 : src1_صف_سائز_aligned;
        kparams->dst_صف_سائز_aligned  = dst_صف_سائز_aligned;
        kparams->src1_سائز             = 0;
        kparams->chunk_سائز            = chunk_سائز;
        kparams->chunk_بائٹیں           = chunk_بائٹیں;
        kparams->is_scalar             = is_scalar_نشر ? 1 : 0;

        ساخت htp_ثنائی_vtcm_layout L;
        htp_ثنائی_vtcm_layout_build(&L, kparams, sess->vtcm_سائز);
        اگر (L.کل_بائٹیں == 0 || L.کل_بائٹیں > sess->vtcm_سائز) {
            واپس جھوٹ;
        }

        kparams->vtcm_سائز = L.کل_بائٹیں;
        واپس سچ;
    }

    شمار htp_ثنائی_kernel_type kernel_type;
    size_t src1_سائز = 0;
    size_t src2_سائز = 0;

    اگر (is_جوڑو_id) {
        GGML_ASSERT(src2 != خالی_اشاریہ);
        kernel_type = HTP_ثنائی_KERNEL_جوڑو_ID;
        src1_سائز = hex_باری_اوپر(src1->ne[1] * src1_صف_سائز_aligned, 128);
        src2_سائز = hex_باری_اوپر(ggml_nelements(src2) * حجم(int32_t), 128);
        اگر (src1_سائز + src2_سائز > sess->vtcm_سائز) {
            واپس جھوٹ;
        }
    } ورنہ اگر (is_صف_bcast) {
        kernel_type = HTP_ثنائی_KERNEL_صف_BCAST;
        src1_سائز = src1_صف_سائز_aligned;
    } ورنہ اگر (is_scalar) {
        src1_سائز = hex_باری_اوپر(ggml_nelements(src1) * elem_سائز, 128);
        اگر (src1_سائز > sess->vtcm_سائز) {
            واپس جھوٹ;
        }
        kernel_type = HTP_ثنائی_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ورنہ اگر (is_same_shape) {
        kernel_type = HTP_ثنائی_KERNEL_SAME_SHAPE;
    } ورنہ اگر (is_complex) {
        src1_سائز = hex_باری_اوپر(ggml_nelements(src1) * elem_سائز + 128, 128);
        اگر (src1_سائز > sess->vtcm_سائز) {
            واپس جھوٹ;
        }
        kernel_type = HTP_ثنائی_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ورنہ {
        src1_سائز = hex_باری_اوپر(ggml_nelements(src1) * elem_سائز + 128, 128);
        اگر (src1_سائز > sess->vtcm_سائز) {
            واپس جھوٹ;
        }
        kernel_type = HTP_ثنائی_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_دھاگے = sess->n_دھاگے;
    kparams->src0_صف_سائز_aligned = src0_صف_سائز_aligned;
    kparams->src1_صف_سائز_aligned = src1_صف_سائز_aligned;
    kparams->dst_صف_سائز_aligned  = dst_صف_سائز_aligned;
    kparams->src1_سائز             = src1_سائز;
    kparams->src2_سائز             = src2_سائز;

    ثابت منطقی src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ثابت منطقی src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ثابت منطقی dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ثابت منطقی src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ثابت منطقی src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ثابت منطقی dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    منطقی بانٹو_at_ne01 = جھوٹ;
    اگر (src0->ne[2] > 1) {
        اگر (kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE) {
            بانٹو_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } ورنہ اگر (kernel_type == HTP_ثنائی_KERNEL_صف_BCAST) {
            بانٹو_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } ورنہ {
            بانٹو_at_ne01 = سچ;
        }
    }
    kparams->بانٹو_at_ne01 = بانٹو_at_ne01 ? 1 : 0;

    منطقی بانٹو_at_ne02 = جھوٹ;
    اگر (src0->ne[3] > 1) {
        اگر (kernel_type == HTP_ثنائی_KERNEL_SAME_SHAPE) {
            بانٹو_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } ورنہ اگر (kernel_type == HTP_ثنائی_KERNEL_صف_BCAST) {
            بانٹو_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } ورنہ {
            بانٹو_at_ne02 = سچ;
        }
    }
    kparams->بانٹو_at_ne02 = بانٹو_at_ne02 ? 1 : 0;

    ساخت htp_ثنائی_vtcm_layout L;
    htp_ثنائی_vtcm_layout_build(&L, kparams, sess->vtcm_سائز);
    اگر (L.صفیں_فی_بفر == 0 || L.کل_بائٹیں > sess->vtcm_سائز) {
        اگر (!is_جوڑو_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_نشر)) {
            ثابت uint32_t کل_elems = (uint32_t) ggml_nelements(src0);
            ثابت uint32_t n_دھاگے = sess->n_دھاگے;
            ثابت uint32_t max_chunk_elems = 32768 / elem_سائز;
            ثابت uint32_t min_chunk_elems = 256;
            ثابت uint32_t ہدف_chunk_elems = hex_باری_اوپر((کل_elems + (2 * n_دھاگے) - 1) / (2 * n_دھاگے), 32);
            ثابت uint32_t chunk_سائز = (std::min)(max_chunk_elems, (std::max)(ہدف_chunk_elems, min_chunk_elems));
            ثابت uint32_t chunk_بائٹیں = hex_باری_اوپر(chunk_سائز * elem_سائز, 128);

            kparams->kernel_type           = HTP_ثنائی_KERNEL_CHUNKED;
            kparams->n_دھاگے             = n_دھاگے;
            kparams->صفیں_فی_بفر       = 1;
            kparams->src1_صف_سائز_aligned = is_scalar_نشر ? 0 : src1_صف_سائز_aligned;
            kparams->src1_سائز             = 0;
            kparams->chunk_سائز            = chunk_سائز;
            kparams->chunk_بائٹیں           = chunk_بائٹیں;
            kparams->is_scalar             = is_scalar_نشر ? 1 : 0;

            htp_ثنائی_vtcm_layout_build(&L, kparams, sess->vtcm_سائز);
            اگر (L.کل_بائٹیں == 0 || L.کل_بائٹیں > sess->vtcm_سائز) {
                واپس جھوٹ;
            }

            kparams->vtcm_سائز = L.کل_بائٹیں;
            واپس سچ;
        }
        واپس جھوٹ;
    }

    kparams->صفیں_فی_بفر = L.صفیں_فی_بفر;
    kparams->vtcm_سائز       = L.کل_بائٹیں;

    واپس سچ;
}

ساکن خالی ggml_hexagon_precompute_unary_params(
    ثابت ساخت ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t)sess->n_دھاگے, src0_nrows);

    kparams->n_دھاگے = n_دھاگے;

    ثابت size_t elem_سائز = ggml_type_سائز(src0->type);

    ثابت size_t src0_ڈیٹا_صف_سائز = src0->ne[0] * elem_سائز;
    ثابت size_t dst_ڈیٹا_صف_سائز  = dst->ne[0]  * ggml_type_سائز(dst->type);

    ثابت size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_ڈیٹا_صف_سائز, 128);
    ثابت size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_ڈیٹا_صف_سائز,  128);

    kparams->src0_صف_سائز_aligned = src0_صف_سائز_aligned;
    kparams->dst_صف_سائز_aligned  = dst_صف_سائز_aligned;

    size_t src1_ڈیٹا_صف_سائز = 0;
    size_t src1_صف_سائز_aligned = 0;
    منطقی نشر_وزن = جھوٹ;

    اگر (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != خالی_اشاریہ);
        src1_ڈیٹا_صف_سائز = src1->ne[0] * ggml_type_سائز(src1->type);
        src1_صف_سائز_aligned = hex_باری_اوپر(src1_ڈیٹا_صف_سائز, 128);
        نشر_وزن = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_صف_سائز_aligned = src1_صف_سائز_aligned;
    kparams->نشر_وزن      = نشر_وزن;

    ساخت htp_unary_vtcm_layout L;
    uint32_t ستون_tile = 0;
    uint32_t vtcm_صف_فی_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                نشر_وزن, n_دھاگے, sess->vtcm_سائز, elem_سائز,
                                &ستون_tile, &vtcm_صف_فی_thread);

    kparams->ستون_tile = ستون_tile;
    kparams->vtcm_صف_فی_thread = vtcm_صف_فی_thread;
    kparams->vtcm_سائز = L.کل_بائٹیں;

    kparams->vtcm_src0_سائز_فی_thread = L.src0_بائٹیں;
    kparams->vtcm_src1_سائز_فی_thread = L.src1_بائٹیں;
    kparams->vtcm_dst_سائز_فی_thread  = L.dst_بائٹیں;

    kparams->vtcm_src0_سائز = L.src0_بائٹیں * n_دھاگے;
    kparams->vtcm_src1_سائز = L.src1_بائٹیں * n_دھاگے;
    kparams->vtcm_dst_سائز  = L.dst_بائٹیں * n_دھاگے;

    kparams->بلاک = ستون_tile ? 0 : ((L.src0_بائٹیں / 2) / src0_صف_سائز_aligned);

    ثابت uint32_t tiles_فی_صف = ستون_tile > 0 ? (src0->ne[0] + ستون_tile - 1) / ستون_tile : 1;
    kparams->div_ne01  = init_fastdiv_اقدار(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_اقدار(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_اقدار(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_اقدار(tiles_فی_صف);
}

ساکن خالی ggml_hexagon_precompute_get_صفیں_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_get_صفیں_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    ثابت uint32_t ne10 = src1->ne[0];
    ثابت uint32_t ne11 = src1->ne[1];
    ثابت uint32_t ne12 = src1->ne[2];
    ثابت uint32_t nr = ne10 * ne11 * ne12;

    ثابت منطقی tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    اگر (tiled) {
        kparams->kernel_type = HTP_GET_صفیں_KERNEL_TILED;
    } ورنہ اگر (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_صفیں_KERNEL_SAMETYPE;
    } ورنہ {
        kparams->kernel_type = HTP_GET_صفیں_KERNEL_FLAT;
    }

    ثابت uint32_t کل_tasks = nr;

    kparams->n_دھاگے = (std::min)((uint32_t)sess->n_دھاگے, کل_tasks);

    ساخت htp_get_صفیں_vtcm_layout vtcm_layout = {};
    جب (kparams->n_دھاگے > 0) {
        htp_get_صفیں_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, کل_tasks, src1->type, kparams->n_دھاگے);
        اگر (vtcm_layout.کل_بائٹیں <= sess->vtcm_سائز) {
            توڑو;
        }
        --kparams->n_دھاگے;
    }

    اگر (kparams->n_دھاگے == 0 && کل_tasks > 0) {
        htp_get_صفیں_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, کل_tasks, src1->type, 1);
    }

    kparams->vtcm_سائز = (کل_tasks == 0) ? 0 : vtcm_layout.کل_بائٹیں;
    kparams->tasks_فی_thread = kparams->n_دھاگے > 0 ? (کل_tasks + kparams->n_دھاگے - 1) / kparams->n_دھاگے : 0;
    kparams->کل_tasks = کل_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_اقدار(ne10) : fastdiv_اقدار{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_اقدار(ne10 * ne11) : fastdiv_اقدار{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_اقدار(ne02) : fastdiv_اقدار{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_اقدار(ne03) : fastdiv_اقدار{0, 0};
}

ساکن خالی ggml_hexagon_precompute_سیٹ_صفیں_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0, //اقدار
    ثابت ساخت ggml_tensor * src1, // indices
    ثابت ساخت ggml_tensor * dst,  // destination
    ساخت htp_سیٹ_صفیں_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t nr = src0->ne[1];

    kparams->n_دھاگے = (std::min)((uint32_t)sess->n_دھاگے, nr);
    kparams->tasks_فی_thread = (nr + kparams->n_دھاگے - 1) / kparams->n_دھاگے;
    kparams->کل_tasks = nr;

    kparams->div_ne11 = init_fastdiv_اقدار(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_اقدار(src1->ne[2]);
    kparams->div_tasks_فی_thread = init_fastdiv_اقدار(kparams->tasks_فی_thread);
    kparams->div_ne02 = init_fastdiv_اقدار(src0->ne[2]);

    ساخت htp_سیٹ_صفیں_vtcm_layout vtcm_layout;
    htp_سیٹ_صفیں_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_دھاگے);
    kparams->vtcm_سائز = vtcm_layout.کل_بائٹیں;
}

ساکن خالی ggml_hexagon_precompute_softmax_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, src0_nrows);

    اعشاری scale = 1.0f;
    اعشاری max_bias = 0.0f;
    حافظہ_نقل(&scale,    &op->op_params[0], حجم(اعشاری));
    حافظہ_نقل(&max_bias, &op->op_params[1], حجم(اعشاری));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ثابت uint32_t n_سر = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));
    kparams->n_سر      = n_سر;
    kparams->n_سر_log2 = n_سر_log2;

    اگر (max_bias > 0.0f && n_سر_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_سر_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);
    } ورنہ {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != خالی_اشاریہ) ? 1 : 0;
    kparams->use_f16  = (src1 != خالی_اشاریہ && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ساخت htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_دھاگے);

    kparams->n_دھاگے                 = n_دھاگے;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_فی_thread     = (src0_nrows + n_دھاگے - 1) / n_دھاگے;
    kparams->vtcm_سائز                 = (uint32_t) layout.کل_بائٹیں;
    kparams->vtcm_src0_سائز_فی_thread = (uint32_t) layout.src0_بائٹیں_فی_thread;
    kparams->vtcm_src1_سائز_فی_thread = (uint32_t) layout.src1_بائٹیں_فی_thread;
    kparams->vtcm_dst_سائز_فی_thread  = (uint32_t) layout.dst_بائٹیں_فی_thread;
    kparams->src0_صف_سائز_aligned     = (uint32_t) layout.src0_spad_half_سائز;
    kparams->src1_صف_سائز_aligned     = (uint32_t) layout.src1_spad_half_سائز;
    kparams->dst_صف_سائز_aligned      = (uint32_t) layout.dst_spad_half_سائز;
    kparams->src0_spad_half_سائز       = (uint32_t) layout.src0_spad_half_سائز;
    kparams->src1_spad_half_سائز       = (uint32_t) layout.src1_spad_half_سائز;
    kparams->dst_spad_half_سائز        = (uint32_t) layout.dst_spad_half_سائز;
    اگر (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } ورنہ اگر (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_نقاب_F16;
    } ورنہ {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_نقاب_F32;
    }

    اگر (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_اقدار(src0->ne[1]);
    اگر (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_اقدار(src0->ne[2]);
    ثابت uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت uint32_t ne13 = src1 ? src1->ne[3] : 1;
    اگر (ne12 > 0) kparams->div_ne12 = init_fastdiv_اقدار(ne12);
    اگر (ne13 > 0) kparams->div_ne13 = init_fastdiv_اقدار(ne13);
}

ساکن خالی ggml_hexagon_precompute_rope_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * dst  = op;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ساخت htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_دھاگے, n_freq_factors);

    kparams->n_دھاگے              = n_دھاگے;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_فی_thread  = (src0_nrows + n_دھاگے - 1) / n_دھاگے;
    kparams->vtcm_سائز              = (uint32_t) layout.کل_بائٹیں;
    kparams->spad_فی_thread        = (uint32_t) layout.بائٹیں_فی_thread;
    kparams->theta_cache_انحراف     = (uint32_t) layout.theta_cache_سائز_aligned;
    kparams->src0_صف_سائز_aligned  = (uint32_t) layout.src0_صف_سائز_aligned;
    kparams->freq_factors_انحراف    = (uint32_t) (layout.بائٹیں_فی_thread * n_دھاگے);
    kparams->freq_factors_سائز      = (uint32_t) layout.freq_factors_سائز_aligned;

    اگر (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_اقدار(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_اقدار(dst->ne[1]);
    }
}

ساکن خالی ggml_hexagon_precompute_ssm_conv_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t d_conv  = (uint32_t) src1->ne[0];
    ثابت uint32_t d_inner = (uint32_t) src0->ne[1];
    ثابت uint32_t n_t     = (uint32_t) dst->ne[1];
    ثابت uint32_t n_s     = (uint32_t) dst->ne[2];
    ثابت uint32_t ncs     = (uint32_t) src0->ne[0];

    ثابت uint32_t n_دھاگے = (std::min)((uint32_t) sess->n_دھاگے, (d_inner + 31) / 32);

    kparams->n_دھاگے = n_دھاگے;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ثابت uint32_t raw_rpt = (d_inner + n_دھاگے - 1) / n_دھاگے;
    ثابت uint32_t d_inner_فی_thread = hex_باری_اوپر(raw_rpt, 32);

    ثابت uint32_t src1_raw_بائٹیں = hex_باری_اوپر(d_inner_فی_thread * d_conv * حجم(اعشاری), 128) + 128;
    ثابت uint32_t src1_T_بائٹیں   = hex_باری_اوپر(d_conv * d_inner_فی_thread * حجم(اعشاری), 128);
    ثابت uint32_t vtcm_src1_فی_thread = src1_raw_بائٹیں + src1_T_بائٹیں;

    uint32_t vtcm_src0_فی_thread = 0;
    uint32_t vtcm_dst_فی_thread  = 0;

    اگر (n_t == 1) {
        kparams->d_inner_tile = d_inner_فی_thread;

        ثابت uint32_t src0_tile_raw_بائٹیں = hex_باری_اوپر(d_inner_فی_thread * d_conv * حجم(اعشاری), 128);
        ثابت uint32_t src0_T_بائٹیں        = hex_باری_اوپر(d_conv * d_inner_فی_thread * حجم(اعشاری), 128);
        vtcm_src0_فی_thread = 2 * src0_tile_raw_بائٹیں + src0_T_بائٹیں;

        ثابت uint32_t dst_tile_بائٹیں = hex_باری_اوپر(d_inner_فی_thread * حجم(اعشاری), 128);
        vtcm_dst_فی_thread = 2 * dst_tile_بائٹیں;
    } ورنہ {
        ثابت size_t vtcm_budget = (sess->vtcm_سائز > 0 ? sess->vtcm_سائز / n_دھاگے : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ثابت uint32_t src0_بلاک_T    = hex_باری_اوپر(ncs * 32 * حجم(اعشاری), 128);
        ثابت size_t   fixed_بائٹیں     = vtcm_src1_فی_thread + src0_بلاک_T;
        ثابت size_t   avail_for_tiles = vtcm_budget > fixed_بائٹیں ? vtcm_budget - fixed_بائٹیں : (128 * 1024);

        uint32_t ہدف_max_tile = hex_باری_اوپر((d_inner_فی_thread + 3) / 4, 32);
        ہدف_max_tile = (std::max)(ہدف_max_tile, 32u);
        ہدف_max_tile = (std::min)(ہدف_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * حجم(اعشاری)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        اگر (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        اگر (d_inner_tile > ہدف_max_tile) {
            d_inner_tile = ہدف_max_tile;
        }
        اگر (d_inner_tile > d_inner_فی_thread) {
            d_inner_tile = d_inner_فی_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ثابت uint32_t src0_tile_raw = hex_باری_اوپر(d_inner_tile * ncs * حجم(اعشاری), 128);
        vtcm_src0_فی_thread = 2 * src0_tile_raw + src0_بلاک_T;

        vtcm_dst_فی_thread = 2 * hex_باری_اوپر(d_inner_tile * n_t * حجم(اعشاری), 128);
    }

    kparams->vtcm_src0_سائز_فی_thread = vtcm_src0_فی_thread;
    kparams->vtcm_src1_سائز_فی_thread = vtcm_src1_فی_thread;
    kparams->vtcm_dst_سائز_فی_thread  = vtcm_dst_فی_thread;

    kparams->vtcm_src0_سائز = vtcm_src0_فی_thread * n_دھاگے;
    kparams->vtcm_src1_سائز = vtcm_src1_فی_thread * n_دھاگے;
    kparams->vtcm_dst_سائز  = vtcm_dst_فی_thread  * n_دھاگے;
    kparams->vtcm_سائز      = kparams->vtcm_src0_سائز + kparams->vtcm_src1_سائز + kparams->vtcm_dst_سائز;
}

ساکن خالی ggml_hexagon_precompute_gated_delta_net_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت ساخت ggml_tensor * q     = op->src[0];
    ثابت ساخت ggml_tensor * k     = op->src[1];
    ثابت ساخت ggml_tensor * v     = op->src[2];
    ثابت ساخت ggml_tensor * g     = op->src[3];
    ثابت ساخت ggml_tensor * state = op->src[5];

    ثابت uint32_t S_v      = (uint32_t) v->ne[0];
    ثابت uint32_t H        = (uint32_t) v->ne[1];
    ثابت uint32_t n_علامات = (uint32_t) v->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) v->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ثابت uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ثابت uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ثابت uint32_t کل_صفیں = H * n_seqs;
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, کل_صفیں);

    ثابت منطقی can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_علامات >= HTP_GDN_MIN_علامات) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ساخت htp_gdn_hmx_vtcm_layout hmx_layout;
    ساخت htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    اگر (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_سائز, کل_صفیں, sess->vtcm_سائز, n_دھاگے, سچ, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_سائز      = HTP_GDN_CHUNK_سائز;
        kparams->n_chunks        = (n_علامات + HTP_GDN_CHUNK_سائز - 1) / HTP_GDN_CHUNK_سائز;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_سائز       = (uint32_t) hmx_layout.کل_بائٹیں;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_بائٹیں;
        kparams->vtcm_فی_thread = (uint32_t) (hmx_layout.کل_بائٹیں / (n_دھاگے > 0 ? n_دھاگے : 1));
    } ورنہ {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_دھاگے);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_فی_thread = (uint32_t) hvx_layout.بائٹیں_فی_thread;
        kparams->vtcm_سائز       = (uint32_t) hvx_layout.کل_بائٹیں;
    }

    kparams->n_دھاگے           = n_دھاگے;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_علامات            = n_علامات;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->کل_صفیں          = کل_صفیں;
    kparams->صفیں_فی_thread     = (کل_صفیں + kparams->n_دھاگے - 1) / kparams->n_دھاگے;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / حجم(اعشاری));
    kparams->state_سائز_فی_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((اعشاری) S_v);

    اگر (H > 0)                  kparams->div_H         = init_fastdiv_اقدار(H);
    اگر (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_اقدار((uint32_t) q->ne[1]);
    اگر (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_اقدار((uint32_t) k->ne[1]);
    اگر (rq3 > 0)                kparams->div_rq3       = init_fastdiv_اقدار(rq3);
    اگر (rk3 > 0)                kparams->div_rk3       = init_fastdiv_اقدار(rk3);
    اگر (kparams->n_دھاگے > 0) kparams->div_n_دھاگے = init_fastdiv_اقدار(kparams->n_دھاگے);
}

ساکن خالی ggml_hexagon_precompute_ترتیب_دو_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    منطقی is_چوٹی_k,
    ساخت htp_ترتیب_دو_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t ne00       = src0->ne[0];
    ثابت uint32_t k          = dst->ne[0];

    int32_t ترتیب = GGML_ترتیب_دو_ترتیب_DESC;
    اگر (!is_چوٹی_k) {
        ترتیب = ((ثابت int32_t *) op->op_params)[0];
    }

    ثابت uint32_t n_دھاگے_max = sess->n_دھاگے > 0 ? sess->n_دھاگے : 4;
    ثابت size_t vtcm_budget = sess->vtcm_سائز > 0 ? sess->vtcm_سائز : (8 * 1024 * 1024);

    ساخت htp_ترتیب_دو_vtcm_layout layout;
    منطقی ok = htp_ترتیب_دو_solve_layout(&layout, ne00, کل_صفیں, k, n_دھاگے_max, vtcm_budget, is_چوٹی_k);
    GGML_ASSERT(ok);

    kparams->n_دھاگے         = (int32_t) layout.n_دھاگے;
    kparams->کل_صفیں        = (int32_t) کل_صفیں;
    kparams->صف_آغاز         = 0;
    kparams->صف_انتہا           = (int32_t) کل_صفیں;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->ترتیب             = ترتیب;
    kparams->is_چوٹی_k          = is_چوٹی_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_سائز         = (int32_t) layout.کل_بائٹیں;
    kparams->phase1_خانہ_سائز  = (int32_t) layout.phase1_خانہ_سائز;
    kparams->ملاؤ_اقدار_off  = (int32_t) layout.ملاؤ_اقدار_off;
    kparams->ملاؤ_indices_off = (int32_t) layout.ملاؤ_indices_off;
    kparams->ملاؤ_elems       = (int32_t) layout.ملاؤ_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ساکن خالی ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ثابت uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ثابت int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    اگر (numer_hi < 0) {
        *ox_hi = 0;
    } ورنہ {
        ثابت uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    اگر (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ساکن منطقی ggml_hexagon_pool_shape_سماتا(
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * dst
) {
    واپس ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ساکن خالی ggml_hexagon_precompute_pool_2d_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * dst,
    ساخت htp_pool_2d_kernel_params * kparams,
    منطقی is_pool_1d
) {
    memset(kparams, 0, حجم(*kparams));

    ثابت uint32_t src_x = (uint32_t) src0->ne[0];
    ثابت uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ثابت uint32_t dst_x = (uint32_t) dst->ne[0];
    ثابت uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ثابت uint32_t src_plane_بائٹیں = src_x * src_y * حجم(اعشاری);
    ثابت uint32_t dst_plane_بائٹیں = dst_x * dst_y * حجم(اعشاری);
    ثابت uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ثابت uint32_t n_دھاگے = (std::min)((uint32_t) sess->n_دھاگے, planes > 0 ? planes : 1);

    ساخت htp_pool_vtcm_layout layout;
    ثابت منطقی ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_دھاگے, sess->vtcm_سائز);
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
    kparams->src_plane_بائٹیں = src_plane_بائٹیں;
    kparams->dst_plane_بائٹیں = dst_plane_بائٹیں;
    kparams->src_plane_بائٹیں_aligned = (uint32_t) layout.src_spad_half_سائز;
    kparams->dst_plane_بائٹیں_aligned = (uint32_t) layout.dst_spad_half_سائز;
    kparams->n_دھاگے = n_دھاگے;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_راستہ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_راستہ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->بلاک_راستہ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_گنتی = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ثابت منطقی narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_راستہ = (kparams->fast_راستہ && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (اعشاری) (kparams->kernel_x * kparams->kernel_y);
}

ساکن منطقی ggml_hexagon_precompute_concat_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        واپس جھوٹ;
    }

    صحیح dim = ((ثابت int32_t *) op->op_params)[0];
    اگر (dim < 0 || dim >= GGML_MAX_DIMS) {
        واپس جھوٹ;
    }
    kparams->dim = dim;

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }
    اگر (src0->type != dst->type || src1->type != dst->type) {
        واپس جھوٹ;
    }

    ثابت uint32_t type_سائز = ggml_type_سائز(dst->type);

    برائے (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
        ثابت int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        اگر (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            واپس جھوٹ;
        }
    }

    ثابت uint32_t max_gather_ne0 = 128 / type_سائز;
    ثابت uint32_t کل_صفیں = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ثابت منطقی b_dense = (src1->nb[0] == type_سائز);
    ثابت منطقی gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_سائز &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_سائز
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_سائز);

    اگر (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_سائز && dst->nb[1] == (size_t) dst->ne[0] * type_سائز &&
        src0->nb[0] == type_سائز &&
        (src1->nb[0] == type_سائز || src1->nb[1] == type_سائز) &&
        gather_strides_ok &&
        htp_gather_صفیں_tensor_outer_contiguous(dst) &&
        htp_gather_صفیں_tensor_outer_contiguous(src0) &&
        htp_gather_صفیں_tensor_outer_contiguous(src1)) {

        اگر (htp_gather_صفیں_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_سائز, کل_صفیں,
                                         sess->n_دھاگے, sess->mdev.count, sess->vtcm_سائز)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_صفیں;
            kparams->n_دھاگے   = (uint8_t) sess->n_دھاگے;
            kparams->vtcm_سائز   = kparams->u.gather.قاش_2 * sess->n_دھاگے;
            واپس سچ;
        }
    }

    ثابت منطقی dma_strides_ok = (src0->nb[0] == type_سائز && src1->nb[0] == type_سائز && dst->nb[0] == type_سائز);

    اگر (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_دھاگے   = 1;
        واپس سچ;
    }

    ثابت منطقی is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ثابت منطقی is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ثابت منطقی transposed_صفیں_ok = (src0->nb[0] == type_سائز && src1->nb[1] == type_سائز && dst->nb[0] == type_سائز);

    اگر (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_صفیں_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ثابت uint32_t n_دھاگے = sess->n_دھاگے > 0 ? (uint32_t) sess->n_دھاگے : 8;
        ساخت htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_سائز, n_دھاگے);

        اگر (sess->vtcm_سائز > 0 && layout.کل_بائٹیں > sess->vtcm_سائز) {
            واپس جھوٹ;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_دھاگے                         = n_دھاگے;
        kparams->vtcm_سائز                         = layout.کل_بائٹیں;
        kparams->u.transposed.spad0_سائز_فی_thread = layout.src0_spad_سائز_فی_thread;
        kparams->u.transposed.spad1_سائز_فی_thread = layout.src1_spad_سائز_فی_thread;
        واپس سچ;
    }

    واپس جھوٹ;
}

ساکن منطقی ggml_hexagon_precompute_cpy_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!src0 || !dst) {
        واپس جھوٹ;
    }

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }
    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    ثابت int64_t nelem_src = ggml_nelements(src0);
    ثابت int64_t nelem_dst = ggml_nelements(dst);
    اگر (nelem_src != nelem_dst || nelem_src < 0) {
        واپس جھوٹ;
    }

    ثابت uint32_t src_type_سائز = ggml_type_سائز(src0->type);
    ثابت uint32_t dst_type_سائز = ggml_type_سائز(dst->type);

    kparams->src0_type_سائز = (uint8_t) src_type_سائز;
    kparams->dst_type_سائز  = (uint8_t) dst_type_سائز;
    kparams->کل_elems    = (uint32_t) nelem_src;

    اگر (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        واپس سچ;
    }

    اگر (nelem_src == 1) {
        اگر (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            واپس سچ;
        }
        اگر ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            واپس سچ;
        }
        واپس جھوٹ;
    }

    ثابت منطقی sametype = (src0->type == dst->type);

    منطقی same_extents = سچ;
    برائے (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
        اگر (src0->ne[d] != dst->ne[d]) {
            same_extents = جھوٹ;
            توڑو;
        }
    }

    ثابت منطقی transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_سائز) || (dst->nb[0] != dst_type_سائز) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_سائز) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_سائز);
    ثابت منطقی sameshape  = same_extents && !transposed;

    ثابت منطقی src_is_contiguous = ggml_is_contiguous(src0);
    ثابت منطقی dst_is_contiguous = ggml_is_contiguous(dst);

    اگر (sametype) {
        اگر (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            واپس سچ;
        }

        ثابت uint32_t max_gather_ne0 = 128 / src_type_سائز;
        ثابت uint32_t کل_صفیں = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        اگر ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_سائز && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_سائز &&
            کل_صفیں > 1 &&
            htp_gather_صفیں_tensor_outer_contiguous(src0)) {

            اگر (htp_gather_صفیں_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, جھوٹ,
                                             src_type_سائز, کل_صفیں,
                                             sess->n_دھاگے, sess->mdev.count, sess->vtcm_سائز)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_صفیں;
                kparams->n_دھاگے   = (uint8_t) sess->n_دھاگے;
                kparams->vtcm_سائز   = kparams->u.gather.قاش_2 * sess->n_دھاگے;
                واپس سچ;
            }
        }

        اگر (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->کل_صفیں  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            واپس سچ;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_دھاگے   = sess->n_دھاگے > 0 ? (uint8_t) sess->n_دھاگے : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_اقدار((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_اقدار((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_اقدار((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_اقدار((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_اقدار((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_اقدار((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        واپس سچ;
    }

    اگر (!sameshape) {
        واپس جھوٹ;
    }

    ثابت منطقی valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    اگر (!valid_conversion) {
        واپس جھوٹ;
    }

    ثابت uint32_t n_دھاگے = sess->n_دھاگے > 0 ? (uint32_t) sess->n_دھاگے : 4;
    ساخت htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_سائز, dst_type_سائز, n_دھاگے);

    اگر (sess->vtcm_سائز > 0 && layout.کل_بائٹیں > sess->vtcm_سائز) {
        واپس جھوٹ;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->کل_صفیں              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_دھاگے               = (uint8_t) n_دھاگے;
    kparams->vtcm_سائز               = layout.کل_بائٹیں;
    kparams->u.convert.src0_buf_سائز = layout.src0_buf_سائز;
    kparams->u.convert.dst_buf_سائز  = layout.dst_buf_سائز;
    kparams->u.convert.spad0_سائز_فی_thread = layout.spad0_سائز_فی_thread;
    kparams->u.convert.spad1_سائز_فی_thread = layout.spad1_سائز_فی_thread;
    kparams->u.convert.src0_صف_stride = layout.src0_صف_stride;
    kparams->u.convert.dst_صف_stride  = layout.dst_صف_stride;
    kparams->u.convert.blk_صفیں        = layout.blk_صفیں;
    kparams->u.convert.div_ne01      = init_fastdiv_اقدار((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_اقدار((uint32_t) (src0->ne[2] * src0->ne[1]));

    واپس سچ;
}

ساکن خالی ggml_hexagon_precompute_fused_mmnx_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0, // W0
    ثابت ساخت ggml_tensor * act,  // x
    int32_t n_weights,
    ساخت htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->n_دھاگے = sess->n_دھاگے;

    ثابت صحیح ne00 = src0->ne[0];
    ثابت صحیح ne01 = src0->ne[1];
    ثابت صحیح ne02 = src0->ne[2];
    ثابت صحیح ne03 = src0->ne[3];

    ثابت صحیح ne10 = act->ne[0];
    ثابت صحیح ne11 = act->ne[1];
    ثابت صحیح ne12 = act->ne[2];
    ثابت صحیح ne13 = act->ne[3];

    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحیح ne00_padded = is_tiled ? hex_باری_اوپر(ne00, 32) : ne00;
    ثابت صحیح ne11_padded = hex_باری_اوپر(ne11, 32);
    ثابت صحیح ne01_tiled  = hex_باری_اوپر(ne01, 32);

    ثابت size_t vtcm_budget = sess->vtcm_سائز;
    ثابت منطقی is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    منطقی hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    اگر (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, خالی_اشاریہ, جھوٹ, is_batched)) {
        اگر (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, خالی_اشاریہ, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, جھوٹ, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            جاؤ finalize;
        }
    }

    اگر (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        واپس;
    }

    {
        ثابت صحیح act_nrows = ne11 * ne12 * ne13;
        ثابت size_t act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
        ثابت size_t src0_صف_سائز = src0->nb[1];

        ثابت uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t بہترین_n_prefetch = 2;
        برائے (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ساخت htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_صف, wtype, ne10, act_nrows, sess->n_دھاگے,
                0, src0_صف_سائز, act_صف_سائز, 0, d, جھوٹ, سچ
            );
            اگر (L.کل_بائٹیں <= sess->vtcm_سائز) {
                بہترین_n_prefetch = d;
                توڑو;
            }
        }

        ساخت htp_mm_hvx_vtcm_layout L;
        منطقی try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_صف, wtype, ne10, act_nrows, sess->n_دھاگے,
            0, src0_صف_سائز, act_صف_سائز, 0, بہترین_n_prefetch, جھوٹ, سچ
        );

        اگر (try_tiled && L.کل_بائٹیں <= sess->vtcm_سائز) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_صف;
            kparams->tile_سائز         = htp_tensor_get_tile_سائز(wtype);
            kparams->aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(wtype);
            kparams->act_صف_سائز      = act_صف_سائز;
            kparams->vtcm_src0_سائز    = L.src0_بائٹیں;
            kparams->vtcm_act_سائز     = L.act_بائٹیں;
            kparams->vtcm_bias_سائز    = 0;
            kparams->vtcm_dst_سائز     = L.dst_بائٹیں;
            kparams->vtcm_سائز         = L.کل_بائٹیں;
            kparams->n_prefetch        = بہترین_n_prefetch;
            kparams->n_weights         = n_weights;
        } ورنہ {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            واپس;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_اقدار(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_اقدار(ne11);
    kparams->div_r2       = init_fastdiv_اقدار(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_اقدار(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_اقدار(ne12);
}

ساکن خالی ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * src0, // W0
    ثابت ساخت ggml_tensor * act,  // x
    ثابت ساخت ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ساخت htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ساکن منطقی ggml_hexagon_op_mul_mat_use_fwht(ثابت ساخت ggml_tensor * dst) {
    اگر (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        واپس جھوٹ;
    }

    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    ثابت int64_t n = src1->ne[0];
    اگر (n < 32 || (n & (n - 1)) != 0) {
        واپس جھوٹ;
    }

    واپس src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_صفیں(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ساکن منطقی ggml_hexagon_supported_mul_mat(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * dst) {
    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    اگر (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        اگر (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            واپس جھوٹ;
        }
        ساخت htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, خالی_اشاریہ, dst, &ukparams);
        واپس ukparams.vtcm_صف_فی_thread > 0 && (size_t) ukparams.vtcm_سائز <= sess->vtcm_سائز;
    }

    اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q2_K: {
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                واپس جھوٹ;
            }

            ثابت int64_t qk = ggml_blck_سائز(src0->type);
            اگر (src0->ne[0] % qk) {
                واپس جھوٹ;
            }

            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                واپس جھوٹ;
            }
            توڑو;
        }

        حالت GGML_TYPE_F16:
            اگر (src0->nb[1] < src0->nb[0]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                واپس جھوٹ;
            }
            توڑو;

        حالت GGML_TYPE_F32:
            اگر (src1->type != GGML_TYPE_F32) {
                واپس جھوٹ;
            }
            اگر (src0->nb[1] < src0->nb[0]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                واپس جھوٹ;
            }
            توڑو;

        حالت GGML_TYPE_BF16:
            اگر (src0->nb[1] < src0->nb[0]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                واپس جھوٹ;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                واپس جھوٹ;
            }
            توڑو;

        طےشدہ:
            واپس جھوٹ;
    }

    ساخت htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_mul_mat_id(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    اگر (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        واپس جھوٹ;
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q2_K:
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                واپس جھوٹ;
            }

            اگر (src0->ne[0] % ggml_blck_سائز(src0->type) != 0) {
                واپس جھوٹ;
            }
            توڑو;

        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_BF16:
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                واپس جھوٹ;
            }
            اگر (src0->ne[0] % 32 != 0) {
                واپس جھوٹ;
            }
            توڑو;

        طےشدہ:
            واپس جھوٹ;
    }

    ساخت htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_سائز, sess->vtcm_سائز);
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_ثنائی(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type == GGML_TYPE_F32) {
        اگر (src1->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }
        اگر (dst->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }
    }
    ورنہ اگر (src0->type == GGML_TYPE_F16) {
        اگر (src1->type != GGML_TYPE_F16) {
            واپس جھوٹ;
        }
        اگر (dst->type != GGML_TYPE_F16) {
            واپس جھوٹ;
        }
    }
    ورنہ {
        واپس جھوٹ;
    }

    اگر (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        واپس جھوٹ;
    }
    اگر (!ggml_are_same_shape(src0, dst)) {
        واپس جھوٹ;
    }
    اگر (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        واپس جھوٹ;
    }

    ساخت htp_ثنائی_kernel_params kparams;
    واپس ggml_hexagon_precompute_ثنائی_params(sess, op_remap_مقصد_htp(op), src0, src1, dst, &kparams);
}

ساکن منطقی ggml_hexagon_supported_جوڑو_id(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!src2) {
        واپس جھوٹ;
    }
    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }
    اگر (!ggml_are_same_shape(src0, dst)) {
        واپس جھوٹ;
    }
    اگر (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        واپس جھوٹ;
    }
    اگر (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        واپس جھوٹ;
    }
    اگر (src0->nb[0] != حجم(اعشاری) || src1->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری) || src2->nb[0] != حجم(int32_t)) {
        واپس جھوٹ;
    }

    // REVISIT: add support for non-contiguous tensors
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }
    اگر (src2->nb[1] < src2->ne[0] * حجم(int32_t)) {
        واپس جھوٹ;
    }

    ساخت htp_ثنائی_kernel_params kparams;
    واپس ggml_hexagon_precompute_ثنائی_params(sess, HTP_OP_جوڑو_ID, src0, src1, dst, &kparams, src2);
}

ساکن منطقی ggml_hexagon_supported_unary(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }
    اگر (dst->type != src0->type) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous_صفیں(src0)) {
        واپس جھوٹ;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    اگر (src0->type == GGML_TYPE_F16) {
        انتخاب (op->op) {
            حالت GGML_OP_NORM:
            حالت GGML_OP_RMS_NORM:
            حالت GGML_OP_L2_NORM:
            حالت GGML_OP_SCALE:
            حالت GGML_OP_CLAMP:
            حالت GGML_OP_SQR:
            حالت GGML_OP_SQRT:
            حالت GGML_OP_LOG:
                توڑو;
            حالت GGML_OP_UNARY:
                اگر (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_قدم) {
                    واپس جھوٹ;
                }
                توڑو;
            طےشدہ:
                واپس جھوٹ;
        }
    }

    اگر (!ggml_are_same_shape(src0, dst)) {
        واپس جھوٹ;
    }

    // dst must be contiguous; src0 may be non-contiguous
    اگر (!ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_مجموع(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_مجموع_صفیں(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    // TODO: add support for non-contigiuos tensors
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_argmax(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (dst->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_pool_2d(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    اگر (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        واپس جھوٹ;
    }
    اگر (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        واپس جھوٹ;
    }
    اگر (!ggml_hexagon_pool_shape_سماتا(src0, op)) {
        واپس جھوٹ;
    }

    ثابت int32_t kernel_x = params[1];
    ثابت int32_t kernel_y = params[2];
    ثابت int32_t stride_x = params[3];
    ثابت int32_t stride_y = params[4];
    ثابت int32_t pad_x    = params[5];
    ثابت int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    اگر (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        واپس جھوٹ;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_دھاگے = (std::min)((uint32_t) sess->n_دھاگے, planes > 0 ? planes : 1);

    ساخت htp_pool_vtcm_layout layout;
    واپس htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_دھاگے, sess->vtcm_سائز);
}

ساکن منطقی ggml_hexagon_supported_pool_1d(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    اگر (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        واپس جھوٹ;
    }
    اگر (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        واپس جھوٹ;
    }
    اگر (!ggml_hexagon_pool_shape_سماتا(src0, op)) {
        واپس جھوٹ;
    }

    ثابت int32_t kernel = params[1];
    ثابت int32_t stride = params[2];
    ثابت int32_t pad    = params[3];
    اگر (kernel <= 0 || stride <= 0 || pad < 0) {
        واپس جھوٹ;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_دھاگے = (std::min)((uint32_t) sess->n_دھاگے, planes > 0 ? planes : 1);

    ساخت htp_pool_vtcm_layout layout;
    واپس htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_دھاگے, sess->vtcm_سائز);
}

ساکن منطقی ggml_hexagon_supported_activations(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }
    اگر (dst->type != src0->type) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous_1(src0)) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    اگر (src1) {
        اگر (src1->type != src0->type) {
            واپس جھوٹ;
        }
        اگر (!ggml_are_same_shape(src0, src1)) {
            واپس جھوٹ;
        }
        اگر (!ggml_is_contiguous_1(src1)) {
            واپس جھوٹ;
        }
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_softmax(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src2) {
        واپس جھوٹ;  // FIXME: add support for sinks
    }

    اگر (src0->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (src0->ne[2] > 512) {
        واپس جھوٹ;
    }

    اگر (src1) {
        اگر (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            واپس جھوٹ;
        }
        اگر (src0->ne[0] != src1->ne[0]) {
            واپس جھوٹ;
        }
        اگر (src1->ne[1] < src0->ne[1]) {
            واپس جھوٹ;
        }
        اگر (src0->ne[2] % src1->ne[2] != 0) {
            واپس جھوٹ;
        }
        اگر (src0->ne[3] % src1->ne[3] != 0) {
            واپس جھوٹ;
        }
    }

    اگر (src1) {
        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            واپس جھوٹ;
        }
    } ورنہ {
        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            واپس جھوٹ;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ثابت int64_t ne0 = src0->ne[0];
    اگر (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        واپس جھوٹ;
    }

    // HVX vector size constraints for softmax
    #تعریف_کرو SOFTMAX_MAX_صف_سائز 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    اگر (ne0 > SOFTMAX_MAX_صف_سائز) {
        واپس جھوٹ;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے  = (std::min)((uint32_t) sess->n_دھاگے, src0_nrows);
    ساخت htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != خالی_اشاریہ, src1 && src1->type == GGML_TYPE_F16, n_دھاگے);
    اگر (layout.کل_بائٹیں > sess->vtcm_سائز) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_سیٹ_صفیں(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0]; //اقدار
    ثابت ساخت ggml_tensor * src1 = op->src[1]; // indices
    ثابت ساخت ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    اگر (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        واپس جھوٹ;
    }

    اگر (src0->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        واپس جھوٹ;
    }

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_get_صفیں(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0]; //اقدار
    ثابت ساخت ggml_tensor * src1 = op->src[1]; // indices
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous_صفیں(src0) || !ggml_is_contiguous_صفیں(src1) || !ggml_is_contiguous_صفیں(dst)) {
        واپس جھوٹ;
    }

    اگر (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        واپس جھوٹ;
    }

    اگر (src0->type == dst->type) {
        اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            واپس جھوٹ;
        }
    } ورنہ اگر (src0->type == GGML_TYPE_I32) {
        واپس جھوٹ;
    } ورنہ اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    اگر (ggml_is_خالی_خانے(op)) {
        واپس سچ;
    }

    ساخت htp_get_صفیں_kernel_params kparams;
    ggml_hexagon_precompute_get_صفیں_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.n_دھاگے == 0 || (size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_argsort(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0]; //اقدار
    ثابت ساخت ggml_tensor * dst  = op;         // indices

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے_max = sess->n_دھاگے > 0 ? sess->n_دھاگے : 4;
    ثابت size_t vtcm_budget = sess->vtcm_سائز > 0 ? sess->vtcm_سائز : (8 * 1024 * 1024);

    ساخت htp_ترتیب_دو_vtcm_layout layout;
    اگر (!htp_ترتیب_دو_solve_layout(&layout, src0->ne[0], کل_صفیں, dst->ne[0], n_دھاگے_max, vtcm_budget, جھوٹ)) {
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_چوٹی_k(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0]; //اقدار
    ثابت ساخت ggml_tensor * dst  = op;         // indices

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_دھاگے_max = sess->n_دھاگے > 0 ? sess->n_دھاگے : 4;
    ثابت size_t vtcm_budget = sess->vtcm_سائز > 0 ? sess->vtcm_سائز : (8 * 1024 * 1024);

    ساخت htp_ترتیب_دو_vtcm_layout layout;
    اگر (!htp_ترتیب_دو_solve_layout(&layout, src0->ne[0], کل_صفیں, dst->ne[0], n_دھاگے_max, vtcm_budget, سچ)) {
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_rope(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * src2 = op->src[2];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!ggml_are_same_shape(src0, dst)) {
        واپس جھوٹ;
    }

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    اگر (src0->ne[0] <= 0) {
        واپس جھوٹ;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    اگر (src0_nrows == 0) {
        واپس جھوٹ;
    }

    ثابت int32_t * op_params = &op->op_params[0];
    ثابت صحیح n_dims = op_params[1];
    ثابت صحیح mode   = op_params[2];
    ثابت صحیح n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    اگر (n_dims < 0 || n_dims % 2 != 0) {
        واپس جھوٹ;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    اگر (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        واپس جھوٹ;
    }

    اعشاری freq_بنیاد;
    حافظہ_نقل(&freq_بنیاد, op_params + 5, حجم(اعشاری));
    اگر (freq_بنیاد < 0.0f) {
        واپس جھوٹ;
    }

    اگر (mode != GGML_ROPE_TYPE_معمول &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        واپس جھوٹ;
    }

    ثابت منطقی is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    اگر (mode == GGML_ROPE_TYPE_VISION) {
        اگر (n_dims != (صحیح) (src0->ne[0] / 2) || n_offs != 0) {
            واپس جھوٹ;
        }
    }

    اگر (is_mrope) {
        ثابت int32_t * sections = op_params + 11;
        اگر (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            واپس جھوٹ;
        }
    }

    ثابت int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    اگر (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        واپس جھوٹ;
    }

    اگر (src2) {
        اگر (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            واپس جھوٹ;
        }
        اگر (src2->ne[0] < (n_dims / 2)) {
            واپس جھوٹ;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    اگر (src0->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری)) {
        واپس جھوٹ;
    }
    اگر (src0->nb[1] < src0->ne[0] * حجم(اعشاری) || dst->nb[1] < dst->ne[0] * حجم(اعشاری)) {
        واپس جھوٹ;
    }

    ثابت uint32_t n_دھاگے = (std::min)((uint32_t) sess->n_دھاگے, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ساخت htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_دھاگے, n_freq_factors);
    اگر (layout.کل_بائٹیں > sess->vtcm_سائز) {
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_ssm_conv(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    // Only support FP32 for now
    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    // Check IO tensor shapes and dims
    اگر (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        واپس جھوٹ; // src0 should be effectively 3D
    }

    ثابت صحیح d_conv  = src1->ne[0];
    ثابت صحیح d_inner = src0->ne[1];
    ثابت صحیح n_t = dst->ne[1];
    ثابت صحیح n_s = dst->ne[2];

    اگر (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        واپس جھوٹ;
    }
    اگر (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        واپس جھوٹ;
    }
    اگر (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        واپس جھوٹ;
    }
    اگر (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        واپس جھوٹ;
    }
    اگر (src0->nb[0] != حجم(اعشاری) || src1->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری)) {
        واپس جھوٹ;
    }
    اگر (src0->nb[1] != src0->ne[0] * حجم(اعشاری) || src1->nb[1] != src1->ne[0] * حجم(اعشاری)) {
        واپس جھوٹ;
    }

    ساخت htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    اگر ((size_t) kparams.vtcm_سائز > sess->vtcm_سائز) {
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی im2col_3d_calc_output_سائز(uint32_t داخلہ, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    اگر (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        واپس جھوٹ;
    }

    ثابت int64_t numerator = (int64_t) داخلہ + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    اگر (numerator < 0) {
        واپس جھوٹ;
    }

    ثابت uint64_t size = (uint64_t) (numerator / stride) + 1;
    اگر (size == 0 || size > UINT32_MAX) {
        واپس جھوٹ;
    }

    *output = (uint32_t) size;
    واپس سچ;
}

ساکن منطقی ggml_hexagon_precompute_im2col_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_IM2_ستون_KERNEL_UNSUPPORTED;

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        واپس جھوٹ;
    }

    اگر (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        واپس جھوٹ;
    }

    اگر (src1->nb[0] != حجم(اعشاری) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    ثابت int32_t * op_params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = op_params[0];
    ثابت int32_t s1 = op_params[1];
    ثابت int32_t p0 = op_params[2];
    ثابت int32_t p1 = op_params[3];
    ثابت int32_t d0 = op_params[4];
    ثابت int32_t d1 = op_params[5];
    ثابت منطقی is_2D = (op_params[6] == 1);

    اگر (s0 <= 0 || d0 <= 0 || p0 < 0) {
        واپس جھوٹ;
    }
    اگر (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        واپس جھوٹ;
    }

    ثابت uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ثابت uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ثابت uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ثابت uint32_t IW = (uint32_t) src1->ne[0];
    ثابت uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ثابت uint32_t OW = (uint32_t) dst->ne[1];

    اگر (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        واپس جھوٹ;
    }

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ثابت uint64_t کل_صفیں_64   = (uint64_t) N * OH;
    اگر (patch_stride_64 > UINT32_MAX || کل_صفیں_64 > UINT32_MAX) {
        واپس جھوٹ;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t کل_صفیں   = (uint32_t) کل_صفیں_64;

    اگر (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        واپس جھوٹ;
    }
    اگر (is_2D && dst->ne[2] != OH) {
        واپس جھوٹ;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(اعشاری);

    منطقی is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    اگر (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ثابت uint32_t max_دھاگے = (sess && sess->n_دھاگے > 0) ? (uint32_t) sess->n_دھاگے : 8;
    ثابت uint32_t n_دھاگے   = (کل_صفیں < max_دھاگے) ? کل_صفیں : max_دھاگے;
    ثابت uint32_t vtcm_سائز   = (sess && sess->vtcm_سائز > 0) ? sess->vtcm_سائز : 0;

    اگر (is_patchembed) {
        ثابت uint64_t src_صف_بائٹیں_raw = (uint64_t) IC * KH * IW * حجم(اعشاری);
        ثابت uint64_t dst_صف_بائٹیں_raw = (uint64_t) OW * patch_stride * dst_elem;
        اگر (src_صف_بائٹیں_raw <= UINT32_MAX && dst_صف_بائٹیں_raw <= UINT32_MAX) {
            ثابت uint32_t src_صف_بائٹیں = hex_باری_اوپر((uint32_t) src_صف_بائٹیں_raw, 256);
            ثابت uint32_t dst_صف_بائٹیں = hex_باری_اوپر((uint32_t) dst_صف_بائٹیں_raw, 256);

            ساخت htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_صف_بائٹیں, dst_صف_بائٹیں, n_دھاگے);

            اگر (vtcm_سائز == 0 || layout.کل_بائٹیں <= vtcm_سائز) {
                kparams->kernel_type          = HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_دھاگے            = (uint8_t) n_دھاگے;
                kparams->vtcm_سائز            = layout.کل_بائٹیں;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_بائٹیں_فی_thread = layout.src_بائٹیں_فی_thread;
                kparams->dst_بائٹیں_فی_thread = layout.dst_بائٹیں_فی_thread;
                kparams->src_صف_بائٹیں        = src_صف_بائٹیں;
                kparams->dst_صف_بائٹیں        = dst_صف_بائٹیں;
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
                kparams->کل_صفیں           = کل_صفیں;
                واپس سچ;
            }
        }
    }

    برائے (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        اگر (Wb > UINT32_MAX) {
            اگر (owb == 1) توڑو;
            جاری;
        }

        ثابت uint64_t src_صف_بائٹیں_raw = (uint64_t) IC * KH * Wb * حجم(اعشاری);
        ثابت uint64_t dst_صف_بائٹیں_raw = (uint64_t) owb * patch_stride * dst_elem;
        اگر (src_صف_بائٹیں_raw > UINT32_MAX || dst_صف_بائٹیں_raw > UINT32_MAX) {
            اگر (owb == 1) توڑو;
            جاری;
        }

        ثابت uint32_t src_صف_بائٹیں = hex_باری_اوپر((uint32_t) src_صف_بائٹیں_raw, 256);
        ثابت uint32_t dst_صف_بائٹیں = hex_باری_اوپر((uint32_t) dst_صف_بائٹیں_raw, 256);

        ساخت htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_صف_بائٹیں, dst_صف_بائٹیں, n_دھاگے);

        اگر (vtcm_سائز == 0 || layout.کل_بائٹیں <= vtcm_سائز) {
            kparams->kernel_type          = HTP_IM2_ستون_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_دھاگے            = (uint8_t) n_دھاگے;
            kparams->vtcm_سائز            = layout.کل_بائٹیں;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_بائٹیں_فی_thread = layout.src_بائٹیں_فی_thread;
            kparams->dst_بائٹیں_فی_thread = layout.dst_بائٹیں_فی_thread;
            kparams->src_صف_بائٹیں        = src_صف_بائٹیں;
            kparams->dst_صف_بائٹیں        = dst_صف_بائٹیں;
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
            kparams->کل_صفیں           = کل_صفیں;
            واپس سچ;
        }
        اگر (owb == 1) توڑو;
    }

    واپس جھوٹ;
}

ساکن منطقی ggml_hexagon_precompute_im2col_3d_params(
    ثابت ساخت ggml_hexagon_session * sess,
    ثابت ساخت ggml_tensor * op,
    ساخت htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, حجم(*kparams));
    kparams->kernel_type = HTP_IM2_ستون_KERNEL_UNSUPPORTED;

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        واپس جھوٹ;
    }

    اگر (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        واپس جھوٹ;
    }

    اگر (src1->nb[0] != حجم(اعشاری) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    ثابت int32_t * params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ثابت int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ثابت int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ثابت int32_t ic = params[9];

    اگر (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        واپس جھوٹ;
    }

    اگر (src1->ne[3] % ic != 0) {
        واپس جھوٹ;
    }

    ثابت uint32_t IC = (uint32_t) ic;
    ثابت uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    اگر (N == 0) {
        واپس جھوٹ;
    }

    uint32_t OW, OH, OD;
    اگر (!im2col_3d_calc_output_سائز(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_سائز(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_سائز(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        واپس جھوٹ;
    }

    ثابت uint32_t KD = (uint32_t) src0->ne[2];
    ثابت uint32_t KH = (uint32_t) src0->ne[1];
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t ID = (uint32_t) src1->ne[2];
    ثابت uint32_t IH = (uint32_t) src1->ne[1];
    ثابت uint32_t IW = (uint32_t) src1->ne[0];

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ثابت uint64_t کل_صفیں_64   = (uint64_t) N * OD * OH;
    ثابت uint64_t dst_ne3_64      = (uint64_t) OD * N;

    اگر (patch_stride_64 > UINT32_MAX || کل_صفیں_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        واپس جھوٹ;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t کل_صفیں   = (uint32_t) کل_صفیں_64;
    ثابت uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    اگر (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        واپس جھوٹ;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? حجم(ggml_fp16_t) : حجم(اعشاری);

    ثابت منطقی is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ثابت uint32_t max_دھاگے = (sess && sess->n_دھاگے > 0) ? (uint32_t) sess->n_دھاگے : 8;
    ثابت uint32_t n_دھاگے   = (کل_صفیں < max_دھاگے) ? کل_صفیں : max_دھاگے;
    ثابت uint32_t vtcm_سائز   = (sess && sess->vtcm_سائز > 0) ? sess->vtcm_سائز : 0;

    اگر (is_patchembed) {
        ثابت uint64_t src_صف_بائٹیں_raw = (uint64_t) IC * KD * KH * IW * حجم(اعشاری);
        ثابت uint64_t dst_صف_بائٹیں_raw = (uint64_t) OW * patch_stride * dst_elem;
        اگر (src_صف_بائٹیں_raw <= UINT32_MAX && dst_صف_بائٹیں_raw <= UINT32_MAX) {
            ثابت uint32_t src_صف_بائٹیں = hex_باری_اوپر((uint32_t) src_صف_بائٹیں_raw, 256);
            ثابت uint32_t dst_صف_بائٹیں = hex_باری_اوپر((uint32_t) dst_صف_بائٹیں_raw, 256);

            ساخت htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_صف_بائٹیں, dst_صف_بائٹیں, n_دھاگے);

            اگر (vtcm_سائز == 0 || layout.کل_بائٹیں <= vtcm_سائز) {
                kparams->kernel_type          = HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_دھاگے            = (uint8_t) n_دھاگے;
                kparams->vtcm_سائز            = layout.کل_بائٹیں;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_بائٹیں_فی_thread = layout.src_بائٹیں_فی_thread;
                kparams->dst_بائٹیں_فی_thread = layout.dst_بائٹیں_فی_thread;
                kparams->src_صف_بائٹیں        = src_صف_بائٹیں;
                kparams->dst_صف_بائٹیں        = dst_صف_بائٹیں;
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
                kparams->کل_صفیں           = کل_صفیں;
                واپس سچ;
            }
        }
    }

    برائے (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        اگر (Wb > UINT32_MAX) {
            اگر (owb == 1) توڑو;
            جاری;
        }

        ثابت uint64_t src_صف_بائٹیں_raw = (uint64_t) IC * KD * KH * Wb * حجم(اعشاری);
        ثابت uint64_t dst_صف_بائٹیں_raw = (uint64_t) owb * patch_stride * dst_elem;
        اگر (src_صف_بائٹیں_raw > UINT32_MAX || dst_صف_بائٹیں_raw > UINT32_MAX) {
            اگر (owb == 1) توڑو;
            جاری;
        }

        ثابت uint32_t src_صف_بائٹیں = hex_باری_اوپر((uint32_t) src_صف_بائٹیں_raw, 256);
        ثابت uint32_t dst_صف_بائٹیں = hex_باری_اوپر((uint32_t) dst_صف_بائٹیں_raw, 256);

        ساخت htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_صف_بائٹیں, dst_صف_بائٹیں, n_دھاگے);

        اگر (vtcm_سائز == 0 || layout.کل_بائٹیں <= vtcm_سائز) {
            kparams->kernel_type          = HTP_IM2_ستون_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_دھاگے            = (uint8_t) n_دھاگے;
            kparams->vtcm_سائز            = layout.کل_بائٹیں;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_بائٹیں_فی_thread = layout.src_بائٹیں_فی_thread;
            kparams->dst_بائٹیں_فی_thread = layout.dst_بائٹیں_فی_thread;
            kparams->src_صف_بائٹیں        = src_صف_بائٹیں;
            kparams->dst_صف_بائٹیں        = dst_صف_بائٹیں;
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
            kparams->کل_صفیں           = کل_صفیں;
            واپس سچ;
        }
        اگر (owb == 1) توڑو;
    }

    واپس جھوٹ;
}

ساکن منطقی ggml_hexagon_supported_im2col(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ساخت htp_im2col_kernel_params kparams;
    واپس ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ساکن منطقی ggml_hexagon_supported_im2col_3d(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ساخت htp_im2col_kernel_params kparams;
    واپس ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ساکن منطقی ggml_hexagon_supported_pad(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    ثابت int32_t lp0 = ((ثابت int32_t *) op->op_params)[0];
    ثابت int32_t rp0 = ((ثابت int32_t *) op->op_params)[1];
    ثابت int32_t circular = ((ثابت int32_t *) op->op_params)[8];

    اگر (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_cumsum(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_diag(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    // diag only supports F32 currently
    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    // Input must have ne[1] == 1 (vector input)
    اگر (src0->ne[1] != 1) {
        واپس جھوٹ;
    }

    // Output must be square in first two dimensions
    اگر (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_solve_tri(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0]; // A
    ثابت ساخت ggml_tensor * src1 = op->src[1]; // B
    ثابت ساخت ggml_tensor * dst  = op;         // X

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (src0->ne[0] != src0->ne[1]) {
        واپس جھوٹ;
    }

    اگر (src0->ne[1] != src1->ne[1]) {
        واپس جھوٹ;
    }

    اگر (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        واپس جھوٹ;
    }

    اگر (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        واپس جھوٹ;
    }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_tri(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) { واپس جھوٹ; }
    اگر (dst->type  != GGML_TYPE_F32) { واپس جھوٹ; }
    اگر (!ggml_are_same_shape(src0, dst)) { واپس جھوٹ; }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { واپس جھوٹ; }

    واپس سچ;

    GGML_غیر_مستعمل(sess);
}

ساکن ثابت حرف * ggml_backend_hexagon_name(ggml_backend_t backend) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    واپس sess->c_name();
}

ساکن خالی ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    حذف backend;
}

ساکن htp_op_رمز op_remap_مقصد_htp(ثابت ggml_tensor * t) {
    انتخاب (t->op) {
        حالت GGML_OP_FLASH_ATTN_EXT:  واپس HTP_OP_FLASH_ATTN_EXT;
        حالت GGML_OP_MUL_MAT:
            اگر (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                واپس HTP_OP_HADAMARD;
            }
            واپس HTP_OP_MUL_MAT;
        حالت GGML_OP_MUL_MAT_ID:      واپس HTP_OP_MUL_MAT_ID;
        حالت GGML_OP_MUL:             واپس HTP_OP_MUL;
        حالت GGML_OP_جوڑو:             واپس HTP_OP_جوڑو;
        حالت GGML_OP_جوڑو_ID:          واپس HTP_OP_جوڑو_ID;
        حالت GGML_OP_SUB:             واپس HTP_OP_SUB;
        حالت GGML_OP_DIV:             واپس HTP_OP_DIV;
        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:             واپس HTP_OP_CPY;
        حالت GGML_OP_CONT:            واپس HTP_OP_CPY;
        حالت GGML_OP_GET_صفیں:        واپس HTP_OP_GET_صفیں;
        حالت GGML_OP_سیٹ_صفیں:        واپس HTP_OP_سیٹ_صفیں;
        حالت GGML_OP_مجموع:             واپس HTP_OP_مجموع;
        حالت GGML_OP_مجموع_صفیں:        واپس HTP_OP_مجموع_صفیں;
        حالت GGML_OP_POOL_2D:         واپس HTP_OP_POOL_2D;
        حالت GGML_OP_POOL_1D:         واپس HTP_OP_POOL_1D;
        حالت GGML_OP_ARGSORT:         واپس HTP_OP_ARGSORT;
        حالت GGML_OP_چوٹی_K:           واپس HTP_OP_چوٹی_K;
        حالت GGML_OP_ARGMAX:          واپس HTP_OP_ARGMAX;
        حالت GGML_OP_NORM:            واپس HTP_OP_NORM;
        حالت GGML_OP_L2_NORM:         واپس HTP_OP_L2_NORM;
        حالت GGML_OP_RMS_NORM:        واپس HTP_OP_RMS_NORM;
        حالت GGML_OP_CONCAT:          واپس HTP_OP_CONCAT;
        حالت GGML_OP_SCALE:           واپس HTP_OP_SCALE;
        حالت GGML_OP_CLAMP:           واپس HTP_OP_CLAMP;
        حالت GGML_OP_LEAKY_RELU:      واپس HTP_OP_LEAKY_RELU;
        حالت GGML_OP_SQR:             واپس HTP_OP_SQR;
        حالت GGML_OP_SQRT:            واپس HTP_OP_SQRT;
        حالت GGML_OP_LOG:             واپس HTP_OP_UNARY_LOG;
        حالت GGML_OP_SOFT_MAX:        واپس HTP_OP_SOFTMAX;
        حالت GGML_OP_SSM_CONV:        واپس HTP_OP_SSM_CONV;
        حالت GGML_OP_GATED_DELTA_NET: واپس HTP_OP_GATED_DELTA_NET;
        حالت GGML_OP_ROPE:            واپس HTP_OP_ROPE;
        حالت GGML_OP_REPEAT:          واپس HTP_OP_REPEAT;
        حالت GGML_OP_CUMSUM:          واپس HTP_OP_CUMSUM;
        حالت GGML_OP_FILL:            واپس HTP_OP_FILL;
        حالت GGML_OP_DIAG:            واپس HTP_OP_DIAG;
        حالت GGML_OP_SOLVE_TRI:       واپس HTP_OP_SOLVE_TRI;
        حالت GGML_OP_TRI:             واپس HTP_OP_TRI;
        حالت GGML_OP_PAD:             واپس HTP_OP_PAD;
        حالت GGML_OP_IM2_ستون:          واپس HTP_OP_IM2_ستون;
        حالت GGML_OP_IM2_ستون_3_D:       واپس HTP_OP_IM2_ستون_3_D;
        حالت GGML_OP_ROLL:            واپس HTP_OP_ROLL;

        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(t)) {
                حالت GGML_UNARY_OP_SILU:       واپس HTP_OP_UNARY_SILU;
                حالت GGML_UNARY_OP_GELU:       واپس HTP_OP_UNARY_GELU;
                حالت GGML_UNARY_OP_GELU_تیز: واپس HTP_OP_UNARY_GELU_تیز;
                حالت GGML_UNARY_OP_GELU_ERF:   واپس HTP_OP_UNARY_GELU_ERF;
                حالت GGML_UNARY_OP_SIGMOID:    واپس HTP_OP_UNARY_SIGMOID;
                حالت GGML_UNARY_OP_NEG:        واپس HTP_OP_UNARY_NEG;
                حالت GGML_UNARY_OP_EXP:        واپس HTP_OP_UNARY_EXP;
                حالت GGML_UNARY_OP_SOFTPLUS:   واپس HTP_OP_UNARY_SOFTPLUS;
                حالت GGML_UNARY_OP_TANH:       واپس HTP_OP_UNARY_TANH;
                حالت GGML_UNARY_OP_ABS:        واپس HTP_OP_UNARY_ABS;
                حالت GGML_UNARY_OP_RELU:       واپس HTP_OP_UNARY_RELU;
                حالت GGML_UNARY_OP_قدم:       واپس HTP_OP_UNARY_قدم;
            طےشدہ:
                توڑو;
            }
            توڑو;

        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(t)) {
                حالت GGML_GLU_OP_SWIGLU:       واپس HTP_OP_GLU_SWIGLU;
                حالت GGML_GLU_OP_SWIGLU_OAI:   واپس HTP_OP_GLU_SWIGLU_OAI;
                حالت GGML_GLU_OP_SWIGLU_CLAMP: واپس HTP_OP_GLU_SWIGLU_CLAMP;
                حالت GGML_GLU_OP_GEGLU:        واپس HTP_OP_GLU_GEGLU;
                حالت GGML_GLU_OP_GEGLU_تیز:  واپس HTP_OP_GLU_GEGLU_تیز;
                حالت GGML_GLU_OP_GEGLU_ERF:    واپس HTP_OP_GLU_GEGLU_ERF;
                طےشدہ: توڑو;
            }
            توڑو;

        طےشدہ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    واپس HTP_OP_INVALID;
}

ساکن اندرونی منطقی op_is_compute(ggml_tensor *گرہ)
{
    واپس !ggml_op_is_خالی_خانے(گرہ->op) && !ggml_is_خالی_خانے(گرہ) && (گرہ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ساکن منطقی mm_is_hmx_eligible(ثابت ggml_tensor * t) {
    اگر (opt_nhmx == 0) { واپس جھوٹ; }

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];

    اگر (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ثابت ggml_tensor src1_c = ggml_hexagon_tensor_collapse_صفیں(src1);
        واپس ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, جھوٹ, جھوٹ);
    }

    ثابت منطقی is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقی is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    واپس ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ساکن منطقی is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت ساخت htp_mm_kernel_params * kparams) {
    اگر (kparams->n_hmx) {
        واپس kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    اگر (!ggml_hexagon_tensor_is_tiled(src0)) {
        واپس جھوٹ;
    }

    واپس kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف;
}

ساکن منطقی is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت ساخت htp_mm_kernel_params * kparams) {
    اگر (kparams->n_hmx) {
        واپس kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    اگر (!ggml_hexagon_tensor_is_tiled(src0)) {
        واپس جھوٹ;
    }

    واپس kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک;
}

ساکن منطقی is_mergeable_mul_mat(ثابت ggml_tensor * t) {
    اگر (t->op != GGML_OP_MUL_MAT) واپس جھوٹ;
    اگر (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) واپس جھوٹ;

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];
    اگر (src0->ne[2] != 1 || src0->ne[3] != 1) واپس جھوٹ;

    اگر (mm_is_hmx_eligible(t)) {
        واپس htp_is_hmx_وزن_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    اگر (src1->type != GGML_TYPE_F32) واپس جھوٹ;

    واپس ggml_hexagon_tensor_is_tiled(src0);
}

ساکن منطقی is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    اگر (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        واپس جھوٹ;
    }
    اگر (n1->src[1] != n2->src[1]) {
        واپس جھوٹ;
    }
    اگر (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        واپس جھوٹ;
    }
    اگر (n1->src[0]->type != n2->src[0]->type) {
        واپس جھوٹ;
    }
    اگر (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        واپس جھوٹ;
    }
    اگر (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        واپس جھوٹ;
    }
    واپس سچ;
}

ساکن منطقی is_mergeable_mul_mat_id(ثابت ggml_tensor * t) {
    اگر (t->op != GGML_OP_MUL_MAT_ID) واپس جھوٹ;

    ثابت ggml_tensor * src0 = t->src[0];
    اگر (mm_is_hmx_eligible(t)) {
        واپس htp_is_hmx_وزن_type(src0->type);
    }

    واپس ggml_hexagon_tensor_is_tiled(src0);
}

ساکن منطقی is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    اگر (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        واپس جھوٹ;
    }
    اگر (n1->src[1] != n2->src[1]) {
        واپس جھوٹ;
    }
    اگر (n1->src[2] != n2->src[2]) {
        واپس جھوٹ;
    }
    اگر (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        واپس جھوٹ;
    }
    اگر (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        واپس جھوٹ;
    }
    اگر (n1->src[0]->type != n2->src[0]->type) {
        واپس جھوٹ;
    }
    اگر (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        واپس جھوٹ;
    }
    اگر (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        واپس جھوٹ;
    }
    واپس سچ;
}

ساکن ggml_حالت ggml_backend_hexagon_گراف_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);

    اگر (sess->آخری_error > HTP_حالت_OK) {
        واپس GGML_حالت_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ثابت std::vector<htp_opnode> * nodes_ptr = خالی_اشاریہ;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    منطقی cache_کامیابی = (graph->uid != 0 && sess->cached_uid == graph->uid);
    اگر (cache_کامیابی) {
        nodes_ptr = &sess->cached_nodes;
    } ورنہ {
        // Tag fusable tensors in graph
        برائے (صحیح i = 0; i < graph->n_nodes; i++) {
            خودکار * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            اگر (!extra) جاری;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            اگر (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } ورنہ اگر ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                اگر ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_جوڑو && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_جوڑو })) ||
                    ggml_گرہ_has_n_استعمال_کرتا(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        برائے (صحیح i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            اگر (!op_is_compute(n)) {
                جاری;
            }

            htp_opnode گرہ(HTP_OP_INVALID, n);
            گرہ.opcode = op_remap_مقصد_htp(n);
            اگر (گرہ.opcode == HTP_OP_MUL_MAT || گرہ.opcode == HTP_OP_MUL_MAT_ID || گرہ.opcode == HTP_OP_GET_صفیں) {
                ggml_tensor * src0 = گرہ.گرہ->src[0];
                اگر (ggml_hexagon_needs_repack(src0)) {
                    اگر (opt_verbose) {
                        حرف dims_buf[64];
                        حرف strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, حجم(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, حجم(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    حافظہ_نقل(raw.ڈیٹا(), src0->ڈیٹا, raw.size());
                    htp_repack_tensor_tiled(src0, raw.ڈیٹا(), raw.size());
                    خودکار * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            اگر (گرہ.opcode == HTP_OP_MUL_MAT || گرہ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    گرہ.گرہ->src[0], گرہ.گرہ->src[1], گرہ.گرہ,
                    (ساخت htp_mm_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_MUL || گرہ.opcode == HTP_OP_جوڑو || گرہ.opcode == HTP_OP_جوڑو_ID || گرہ.opcode == HTP_OP_SUB || گرہ.opcode == HTP_OP_DIV) {
                ثابت ggml_tensor * src1 = گرہ.گرہ->src[1];
                ثابت ggml_tensor * src2 = (گرہ.opcode == HTP_OP_جوڑو_ID) ? گرہ.گرہ->src[2] : خالی_اشاریہ;
                GGML_ASSERT(ggml_hexagon_precompute_ثنائی_params(sess,
                    گرہ.opcode, گرہ.گرہ->src[0], src1, گرہ.گرہ,
                    (ساخت htp_ثنائی_kernel_params *) گرہ.kernel_params, src2));
            } ورنہ اگر (گرہ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_fa_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (htp_op_is_unary(گرہ.opcode)) {
                خودکار inputs = گرہ.get_inputs();
                ثابت ساخت ggml_tensor * src0 = inputs[0];
                ثابت ساخت ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : خالی_اشاریہ;
                ggml_hexagon_precompute_unary_params(sess,
                    گرہ.opcode, src0, src1, گرہ.dst(),
                    (ساخت htp_unary_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_GET_صفیں) {
                ggml_hexagon_precompute_get_صفیں_params(sess,
                    گرہ.گرہ->src[0], گرہ.گرہ->src[1], گرہ.dst(),
                    (ساخت htp_get_صفیں_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_سیٹ_صفیں) {
                ggml_hexagon_precompute_سیٹ_صفیں_params(sess,
                    گرہ.گرہ->src[0], گرہ.گرہ->src[1], گرہ.dst(),
                    (ساخت htp_سیٹ_صفیں_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_rope_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    گرہ.گرہ->src[0], گرہ.گرہ->src[1], گرہ.dst(),
                    (ساخت htp_ssm_conv_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_softmax_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_gdn_kernel_params *)گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_ARGSORT || گرہ.opcode == HTP_OP_چوٹی_K) {
                ggml_hexagon_precompute_ترتیب_دو_params(sess,
                    گرہ.گرہ,
                    گرہ.opcode == HTP_OP_چوٹی_K,
                    (ساخت htp_ترتیب_دو_kernel_params *) گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_POOL_2D || گرہ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, گرہ.گرہ->src[0], گرہ.dst(),
                    (ساخت htp_pool_2d_kernel_params *)گرہ.kernel_params,
                    گرہ.opcode == HTP_OP_POOL_1D);
            } ورنہ اگر (گرہ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_concat_kernel_params *) گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_CPY || گرہ.opcode == HTP_OP_CPY_FENCE) {
                ثابت منطقی ok = ggml_hexagon_precompute_cpy_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_copy_kernel_params *) گرہ.kernel_params
                );
                ثابت خودکار * kparams = (ثابت ساخت htp_copy_kernel_params *) گرہ.kernel_params;
                اگر (ok && گرہ.opcode == HTP_OP_CPY && kparams->کل_elems == 0) {
                    جاری;
                }
            } ورنہ اگر (گرہ.opcode == HTP_OP_IM2_ستون) {
                ggml_hexagon_precompute_im2col_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_im2col_kernel_params *) گرہ.kernel_params
                );
            } ورنہ اگر (گرہ.opcode == HTP_OP_IM2_ستون_3_D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    گرہ.گرہ,
                    (ساخت htp_im2col_kernel_params *) گرہ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(گرہ));
        }

        اگر (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } ورنہ {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    برائے (ثابت خودکار & گرہ : *nodes_ptr) {
        sess->قطار_میں_ڈالو_op(گرہ);
    }

    اگر (sess->آخری_error > HTP_حالت_OK) {
        واپس GGML_حالت_FAILED;
    }

    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    اگر (sess->آخری_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), حالت_مقصد_str(sess->آخری_error));
    }
}

شمار ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ساخت ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ساخت ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    خالی reset() {
        ranges.clear();
    }

    خالی جوڑو_2(ثابت ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    منطقی check(ثابت ggml_hexagon_mem_range & mr) ثابت {
        برائے (ثابت خودکار & cmp : ranges) {
            اگر (mr.pb != cmp.pb) {
                جاری;
            }
            اگر (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                جاری;
            }
            اگر (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                واپس جھوٹ;
            }
        }
        واپس سچ;
    }
};

ساکن ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ثابت ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ثابت ggml_tensor * بنیاد = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    اگر (tensor->بفر) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->بفر,
            /*.p0 =*/ (uint64_t) tensor->ڈیٹا,
            /*.p1 =*/ (uint64_t) tensor->ڈیٹا + ggml_backend_buft_get_alloc_سائز(tensor->بفر->buft, tensor),
            /*.pt =*/ pt,
        };
    } ورنہ {
        mr = {
            /*.pb =*/ (uint64_t) بنیاد,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    واپس mr;
}

ساکن خالی ggml_hexagon_mem_ranges_جوڑو_گرہ(ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & گرہ) {
    اگر (گرہ.is_خالی_خانے()) واپس;

    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (گرہ.گرہ->src[i]) {
            mrs.جوڑو_2(ggml_hexagon_mem_range_from_tensor(گرہ.گرہ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    برائے (ثابت خودکار * fused : گرہ.fused) {
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (fused->src[i]) {
                mrs.جوڑو_2(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.جوڑو_2(ggml_hexagon_mem_range_from_tensor(گرہ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ساکن منطقی ggml_hexagon_mem_ranges_check_گرہ(ثابت ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & گرہ) {
    اگر (گرہ.is_خالی_خانے()) واپس سچ;

    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (گرہ.گرہ->src[i]) {
            اگر (!mrs.check(ggml_hexagon_mem_range_from_tensor(گرہ.گرہ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                واپس جھوٹ;
            }
        }
    }
    برائے (ثابت خودکار * fused : گرہ.fused) {
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (fused->src[i]) {
                اگر (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    واپس جھوٹ;
                }
            }
        }
    }
    واپس mrs.check(ggml_hexagon_mem_range_from_tensor(گرہ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ساکن std::vector<صحیح> ggml_hexagon_گراف_optimize_reorder(ثابت std::vector<htp_opnode> & nodes) {
    ثابت صحیح n = nodes.size();

    std::vector<صحیح> res;
    res.reserve(n);

    std::vector<منطقی> مستعمل(n, جھوٹ);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    برائے (صحیح i0 = 0; i0 < n; i0++) {
        اگر (مستعمل[i0]) {
            جاری;
        }

        ثابت خودکار & node0 = nodes[i0];

        اگر (!node0.stackable()) {
            res.push_back(i0);
            مستعمل[i0] = سچ;
            جاری;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr صحیح N_FORWARD = 16;

        std::vector<صحیح> ڈھیر;
        ڈھیر.push_back(i0);

        mrs.reset();

        برائے (صحیح i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            اگر (مستعمل[i1]) {
                جاری;
            }

            ثابت خودکار & node1 = nodes[i1];

            اگر (node1.stackable() && node1.same_داخلہ(node0) && ggml_hexagon_mem_ranges_check_گرہ(mrs, node1)) {
                ڈھیر.push_back(i1);
            } ورنہ {
                ggml_hexagon_mem_ranges_جوڑو_گرہ(mrs, node1);
            }
        }

        برائے (صحیح idx : ڈھیر) {
            res.push_back(idx);
            مستعمل[idx] = سچ;
        }
    }

    واپس res;
}

ساکن خالی ggml_backend_hexagon_گراف_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_گراف_optimize_params * params) {
    GGML_غیر_مستعمل(params);

    ثابت صحیح n = gf->n_nodes;

    constexpr صحیح MAX_FUSE = 16;

    شمار ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    برائے (صحیح i = 0; i < n; i++) {
        htp_opnode گرہ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        اگر (گرہ.op() == GGML_OP_جوڑو ||
            گرہ.op() == GGML_OP_NORM ||
            گرہ.op() == GGML_OP_RMS_NORM) {
            ops[0] = گرہ.op();

            صحیح f = i + 1;
            جب (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                اگر (gf->nodes[f]->op != GGML_OP_جوڑو &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    توڑو;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            برائے (; f > 1; f--) {
                اگر (ggml_can_fuse(gf, i, ops, f)) {
                    توڑو;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            برائے (صحیح k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                گرہ.جوڑو_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(گرہ));
    }

    ثابت خودکار ترتیب = ggml_hexagon_گراف_optimize_reorder(nodes);

    // unfuse
    {
        صحیح j = 0;
        برائے (ثابت خودکار i : ترتیب) {
            ثابت خودکار & گرہ = nodes[i];

            gf->nodes[j++] = گرہ.گرہ;

            برائے (خودکار * fused : گرہ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_غیر_مستعمل(backend);
}

ساکن uint64_t ggml_hexagon_session_کلید(ثابت ggml_hexagon_session * sess) {
    واپس ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ساکن منطقی ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    خودکار sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خودکار sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    خودکار sbuf_dst = sess_dst->mmap_tensor(dst);

    اگر (!sess_src->clone_بفر(sbuf_dst)) { واپس جھوٹ; }

    ثابت uint64_t src_کلید = ggml_hexagon_session_کلید(sess_src);
    خودکار & fence_خانہ = sess_dst->cpy_fence_slots[src_کلید];
    اگر (!fence_خانہ) {
        fence_خانہ = (متلون uint32_t *) sess_dst->alloc_fence(1);
    }

    اگر (!sess_src->clone_بفر(sess_dst->fence_buf)) { واپس جھوٹ; }

    اگر (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ساکن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.بفر = &sess_dst->fence_buf->backend_بفر;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.ڈیٹا   = (خالی *) fence_خانہ;
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

    sess_src->قطار_میں_ڈالو_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->قطار_میں_ڈالو_fence(&fence_tensor, fence_seq, /* wait = */ سچ);

    sess_dst->جوڑو_peer(sess_src);

    واپس سچ;
}

ساکن منطقی ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    خودکار sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خودکار sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    خودکار sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    اگر (!sess_dst->clone_بفر(sbuf_src)) { واپس جھوٹ; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->قطار_میں_ڈالو_cpy(src, dst);
    sess_dst->جوڑو_peer(sess_src);

    واپس سچ;
}

ساکن منطقی ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        واپس جھوٹ;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    اگر (!dst->extra) {
        ggml_backend_بفر_init_tensor(dst->بفر, dst);
    }

    خودکار * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ثابت خودکار * src_extra = static_cast<ثابت ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    خودکار sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    خودکار sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    اگر (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->قطار_میں_ڈالو_cpy(src, dst);
        واپس سچ;
    }

    اگر (sess_src->phys_idx != sess_dst->phys_idx)
        واپس ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    واپس ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ساکن ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خودکار sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = نیا ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_خانہ = (متلون uint32_t *) sess->alloc_fence(1);

    ساکن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.بفر = &sess->fence_buf->backend_بفر;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.ڈیٹا   = (خالی *) hex_event->fence_خانہ;
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

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (خالی *)hex_event, (خالی *)hex_event->fence_خانہ);

    واپس نیا ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ساکن خالی ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    اگر (hex_event->seq == 0) {
        واپس;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (خالی *)hex_event, hex_event->seq, (خالی *)hex_event->fence_خانہ);

    خودکار * fence = reinterpret_cast<ثابت متلون std::atomic<uint32_t> *>(hex_event->fence_خانہ);

    اگر ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    جب (سچ) {
        اگر ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t حالت_2 = fence[1].load(std::memory_order_acquire);
            اگر (حالت_2 > HTP_حالت_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), حالت_مقصد_str(حالت_2));
            }
            توڑو;
        }
        std::this_thread::yield();
    }
}

ساکن خالی ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    خودکار * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (خالی *)hex_event);
    hex_event->fence_sess->free_fence((خالی *) hex_event->fence_خانہ, 1);
    حذف hex_event;
    حذف event;
}

ساکن خالی ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    خودکار * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ساکن خالی ggml_backend_hexagon_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    خودکار hex_event = static_cast<ggml_hexagon_event *>(event->context);

    اگر (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->قطار_میں_ڈالو_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ جھوٹ);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (خالی *)hex_event, hex_event->seq, (خالی *)hex_event->fence_خانہ);
}

ساکن خالی ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    خودکار hex_event = static_cast<ggml_hexagon_event *>(event->context);

    اگر (hex_event->seq == 0) {
        واپس;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (خالی *)hex_event, hex_event->seq, (خالی *)hex_event->fence_خانہ);

    // same physical NPU runs sequentially in FIFO order
    اگر (sess->phys_idx == hex_event->sess->phys_idx) {
        اگر (sess != hex_event->sess) {
            sess->جوڑو_peer(hex_event->sess);
        }
        واپس;
    }

    sess->clone_بفر(hex_event->fence_sess->fence_buf);
    sess->جوڑو_peer(hex_event->sess);
    sess->قطار_میں_ڈالو_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ سچ);
}

ساکن خالی ggml_backend_hexagon_سیٹ_tensor_async(ggml_backend_t backend, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, tensor->بفر ? (صحیح) tensor->بفر->usage : -1);
    ggml_backend_tensor_سیٹ(tensor, ڈیٹا, انحراف, size);
}

ساکن خالی ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, tensor->بفر ? (صحیح) tensor->بفر->usage : -1);
    sess->flush_sync();
    اگر (sess->آخری_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), حالت_مقصد_str(sess->آخری_error));
    }
    ggml_backend_tensor_get(tensor, ڈیٹا, انحراف, size);
}

ساکن خالی ggml_backend_hexagon_سیٹ_tensor_2d_async(ggml_backend_t backend,
                                                     ساخت ggml_tensor * tensor,
                                                     ثابت خالی * ڈیٹا,
                                                     size_t انحراف,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ڈیٹا) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا, tensor->بفر ? (صحیح) tensor->بفر->usage : -1);
    ggml_backend_tensor_سیٹ_2d(tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

ساکن خالی ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ثابت ساخت ggml_tensor * tensor,
                                                     خالی * ڈیٹا,
                                                     size_t انحراف,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ڈیٹا) {
    خودکار sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا, tensor->بفر ? (صحیح) tensor->بفر->usage : -1);
    sess->flush_sync();
    اگر (sess->آخری_error > HTP_حالت_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), حالت_مقصد_str(sess->آخری_error));
    }
    ggml_backend_tensor_get_2d(tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
}

ساکن ساخت ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_سیٹ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_hexagon_گراف_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_گراف_optimize,
};

ساکن ggml_guid_t ggml_backend_hexagon_guid() {
    ساکن ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    واپس &guid;
}

منطقی ggml_backend_is_hexagon(ggml_backend_t backend) {
    واپس backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ساکن ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خودکار sess    = dev_ctx->session();

    واپس نیا ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_غیر_مستعمل(params);
}

ساکن ثابت حرف * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    واپس dev_ctx->c_name();

    GGML_غیر_مستعمل(dev);
}

ساکن ثابت حرف * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    واپس "Hexagon";
    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_hexagon_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    *آزاد  = opt_mbuf_کل;
    *کل = opt_mbuf_کل;

    GGML_غیر_مستعمل(dev);
}

ساکن شمار ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ جھوٹ,
        /* .events                = */ سچ,
        /* .mmap_support          = */ جھوٹ,
    };
}

ساکن ggml_backend_بفر_type_t ggml_backend_hexagon_device_get_بفر_type(ggml_backend_dev_t dev) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    واپس &dev_ctx->بفر_type;
}

ساکن ggml_backend_بفر_type_t ggml_backend_hexagon_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    اگر (!opt_hostbuf) {
        واپس تہی;
    }
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    واپس &dev_ctx->host_بفر_type;
}

ساکن منطقی ggml_hexagon_supported_cpy(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ساخت htp_copy_kernel_params kparams;
    واپس ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ساکن منطقی ggml_hexagon_supported_cont(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ساخت htp_copy_kernel_params kparams;
    واپس ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ساکن منطقی ggml_hexagon_supported_repeat(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    GGML_غیر_مستعمل(sess);
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    // Support f32 and f16
    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) واپس جھوٹ;

    // src and dst must be the same type
    اگر (src0->type != dst->type) واپس جھوٹ;

    // dst dims must be multiples of src dims
    اگر (dst->ne[0] % src0->ne[0] != 0) واپس جھوٹ;
    اگر (dst->ne[1] % src0->ne[1] != 0) واپس جھوٹ;
    اگر (dst->ne[2] % src0->ne[2] != 0) واپس جھوٹ;
    اگر (dst->ne[3] % src0->ne[3] != 0) واپس جھوٹ;

    // require contiguous tensors (no transposition)
    اگر (ggml_is_transposed(src0) || ggml_is_transposed(dst)) واپس جھوٹ;

    واپس سچ;
}

ساکن منطقی ggml_hexagon_supported_concat(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ساخت htp_concat_kernel_params kparams;
    واپس ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ساکن منطقی ggml_hexagon_supported_fill(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * dst = op;

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }

    واپس سچ;
    GGML_غیر_مستعمل(sess);
}

ساکن منطقی ggml_hexagon_supported_roll(ثابت ساخت ggml_hexagon_session * sess, ثابت ساخت ggml_tensor * op) {
    GGML_غیر_مستعمل(sess);

    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    اگر (!ggml_are_same_shape(src0, dst)) {
        واپس جھوٹ;
    }

    اگر (src0->nb[0] != ggml_type_سائز(src0->type) || dst->nb[0] != ggml_type_سائز(dst->type)) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    واپس سچ;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ساکن منطقی ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    خودکار sess    = dev_ctx->session();

    // reject ops that match the filter
    اگر (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        واپس جھوٹ;
    }

    منطقی supp = جھوٹ;
    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
            supp = سچ;
            توڑو;

        حالت GGML_OP_MUL:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_SUB:
        حالت GGML_OP_DIV:
            supp = ggml_hexagon_supported_ثنائی(sess, op);
            توڑو;

        حالت GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            توڑو;

        حالت GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            توڑو;

        حالت GGML_OP_جوڑو_ID:
            supp = ggml_hexagon_supported_جوڑو_id(sess, op);
            توڑو;

        حالت GGML_OP_NORM:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            توڑو;

        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            توڑو;

        حالت GGML_OP_مجموع:
            supp = ggml_hexagon_supported_مجموع(sess, op);
            توڑو;

        حالت GGML_OP_مجموع_صفیں:
            supp = ggml_hexagon_supported_مجموع_صفیں(sess, op);
            توڑو;

        حالت GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            توڑو;

        حالت GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            توڑو;

        حالت GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            توڑو;

        حالت GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            توڑو;

        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(op)) {
                حالت GGML_UNARY_OP_NEG:
                حالت GGML_UNARY_OP_EXP:
                حالت GGML_UNARY_OP_SIGMOID:
                حالت GGML_UNARY_OP_SOFTPLUS:
                حالت GGML_UNARY_OP_TANH:
                حالت GGML_UNARY_OP_ABS:
                حالت GGML_UNARY_OP_SILU:
                حالت GGML_UNARY_OP_GELU:
                حالت GGML_UNARY_OP_GELU_تیز:
                حالت GGML_UNARY_OP_GELU_ERF:
                حالت GGML_UNARY_OP_RELU:
                حالت GGML_UNARY_OP_قدم:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    توڑو;
                طےشدہ:
                    supp = جھوٹ;
                    توڑو;
            }
            توڑو;

        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(op)) {
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_SWIGLU_OAI:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                حالت GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU_تیز:
                حالت GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    توڑو;
                طےشدہ:
                    supp = جھوٹ;
                    توڑو;
            }
            توڑو;

        حالت GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            توڑو;

        حالت GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            توڑو;

        حالت GGML_OP_سیٹ_صفیں:
            supp = ggml_hexagon_supported_سیٹ_صفیں(sess, op);
            توڑو;

        حالت GGML_OP_GET_صفیں:
            supp = ggml_hexagon_supported_get_صفیں(sess, op);
            توڑو;

        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            توڑو;

        حالت GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            توڑو;

        حالت GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            توڑو;

        حالت GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            توڑو;

        حالت GGML_OP_چوٹی_K:
            supp = ggml_hexagon_supported_چوٹی_k(sess, op);
            توڑو;

        حالت GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            توڑو;

        حالت GGML_OP_IM2_ستون:
            supp = ggml_hexagon_supported_im2col(sess, op);
            توڑو;

        حالت GGML_OP_IM2_ستون_3_D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            توڑو;

        حالت GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            توڑو;

        حالت GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            توڑو;

        حالت GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            توڑو;

        حالت GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            توڑو;

        حالت GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            توڑو;

        حالت GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            توڑو;

        حالت GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            توڑو;

        حالت GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            توڑو;

        حالت GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            توڑو;

        طےشدہ:
            توڑو;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    واپس supp;
}

ساکن منطقی ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    منطقی supp = (buft == &dev_ctx->host_بفر_type) || (buft == &dev_ctx->بفر_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    واپس supp;
}

ساکن ثابت ساخت ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ تہی,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ تہی,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    برائے (size_t i = 0; i < opt_ndev; i++) {
        ثابت خودکار & cfg = opt_device_configs[i];
        اگر (cfg.mdev_گروہ.خالی_خانے()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } ورنہ {
            std::string peers_str;
            برائے (ثابت خودکار & p : cfg.mdev_گروہ) {
                اگر (!peers_str.خالی_خانے()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = نیا ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    برائے (size_t i = 0; i < opt_ndev; i++) {
        خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        حذف dev_ctx;
    }
}

ساکن ثابت حرف * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    واپس "HTP";
    GGML_غیر_مستعمل(reg);
}

ساکن size_t ggml_backend_hexagon_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    واپس opt_ndev;
    GGML_غیر_مستعمل(reg);
}

ساکن ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    خودکار hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    اگر (اشاریہ >= opt_ndev || !hreg->devices[اشاریہ].context) {
        واپس خالی_اشاریہ;
    }

    واپس &hreg->devices[اشاریہ];
}

// ** communication context for tensor-split allreduce

ساکن خالی * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    اگر (n_backends < 2 || n_backends > 4) {
        واپس خالی_اشاریہ;
    }

    برائے (size_t i = 0; i < n_backends; ++i) {
        اگر (!ggml_backend_is_hexagon(backends[i])) {
            واپس خالی_اشاریہ;
        }
    }

    برائے (size_t i = 0; i < n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        برائے (size_t j = i + 1; j < n_backends; j++) {
            خودکار sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            اگر (sess_i->phys_idx == sess_j->phys_idx) {
                واپس خالی_اشاریہ;
            }
        }
    }

    خودکار * ctx = نیا ggml_backend_hexagon_comm_context();
    ctx->backends.تفویض(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ساکن ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    برائے (size_t i = 0; i < n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (متلون uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].بفر = &sess_i->fence_buf->backend_بفر;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].ڈیٹا   = (خالی *) ctx->fence_slots[i];
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

    واپس ctx;
}

ساکن خالی ggml_backend_hexagon_comm_free(خالی * comm_ctx_v) {
    اگر (!comm_ctx_v) واپس;
    خودکار * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    برائے (size_t i = 0; i < ctx->n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((خالی *) ctx->fence_slots[i], 1);
    }
    حذف ctx;
}

ساکن منطقی ggml_backend_hexagon_comm_allreduce_tensor(خالی * comm_ctx_v, ساخت ggml_tensor ** tensors) {
    اگر (opt_ar_select == 0 || !comm_ctx_v) واپس جھوٹ;
    خودکار * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ثابت size_t n_backends = comm_ctx->n_backends;

    اگر (n_backends < 2 || n_backends > 4) واپس جھوٹ;

    برائے (size_t i = 0; i < n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        برائے (size_t j = i + 1; j < n_backends; j++) {
            خودکار sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            اگر (sess_i->phys_idx == sess_j->phys_idx) {
                واپس جھوٹ;
            }
        }
    }

    برائے (size_t i = 0; i < n_backends; i++) {
        اگر (!tensors[i] || !tensors[i]->بفر || !ggml_backend_بفر_is_hexagon(tensors[i]->بفر)) {
            واپس جھوٹ;
        }
        اگر (tensors[i]->type != tensors[0]->type) {
            واپس جھوٹ;
        }
        اگر (!ggml_is_contiguous(tensors[i])) {
            واپس جھوٹ;
        }
        اگر (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            واپس جھوٹ;
        }
    }

    اگر (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    برائے (size_t r = 0; r < n_backends; r++) {
        خودکار sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ساخت htp_allreduce_kernel_params kparams;
        اگر (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, جھوٹ, جھوٹ, /*is_shard_ok=*/ جھوٹ, &kparams)) {
            واپس جھوٹ;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    برائے (size_t i = 1; i < n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        اگر ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    اگر (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_اندراج = max_seq;
    اگر (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    برائے (size_t i = 0; i < n_backends; i++) {
        خودکار sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ثابت ggml_tensor *> ڈیٹا_tensors(n_backends);
    std::vector<ثابت ggml_tensor *> sync_tensors(n_backends);
    برائے (size_t i = 0; i < n_backends; i++) {
        ڈیٹا_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    برائے (size_t r = 0; r < n_backends; r++) {
        خودکار sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->قطار_میں_ڈالو_allreduce(tensors[r], ڈیٹا_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_اندراج, fence_seq_exit);
        برائے (size_t j = 0; j < n_backends; j++) {
            اگر (r != j) {
                sess->جوڑو_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    واپس سچ;
}

ساکن ggml_backend_بفر_type_t ggml_backend_hexagon_بانٹو_بفر_type(صحیح main_device, ثابت اعشاری * tensor_بانٹو) {
    GGML_غیر_مستعمل(tensor_بانٹو);
    خودکار reg = ggml_backend_hexagon_reg();
    خودکار dev = ggml_backend_reg_dev_get(reg, main_device);
    اگر (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    اگر (!dev) واپس خالی_اشاریہ;
    خودکار dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    واپس &dev_ctx->بفر_type;
}

ساکن خالی * ggml_backend_hexagon_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_غیر_مستعمل(reg);
    اگر (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        واپس (خالی *) ggml_backend_hexagon_بانٹو_بفر_type;
    }
    اگر (strcmp(name, "ggml_backend_comm_init") == 0) {
        واپس (خالی *) ggml_backend_hexagon_comm_init;
    }
    اگر (strcmp(name, "ggml_backend_comm_free") == 0) {
        واپس (خالی *) ggml_backend_hexagon_comm_free;
    }
    اگر (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        واپس (خالی *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    واپس تہی;
}

سانچہ<قسم_نام T> std::vector<T> str_مقصد_vec(ثابت حرف* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    جب (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, خالی_اشاریہ, 0));
    }

    واپس v;
}

سانچہ<قسم_نام T, صحیح بنیاد_2=10> std::string vec_مقصد_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(بنیاد_2) << std::showbase;
    برائے (خودکار i : v) { ss << i << ','; }
    خودکار str = ss.str(); str.pop_back(); // drop last comma
    واپس str;
}

ساکن خالی ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, منطقی discovery_supported, ثابت std::unordered_map<صحیح, fastrpc_domain> & cdsp_map) {
    اگر (discovery_supported) {
        خودکار it = cdsp_map.find(cfg.physical_idx);
        اگر (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } ورنہ {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } ورنہ {
        انتخاب (cfg.physical_idx) {
            حالت 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                توڑو;
            حالت 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                توڑو;
            طےشدہ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                توڑو;
        }
    }
    برائے (خودکار & sub_cfg : cfg.mdev_گروہ) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ساکن خالی ggml_hexagon_discover_devices() {
    std::unordered_map<صحیح, fastrpc_domain> cdsp_map;
    منطقی discovery_supported = جھوٹ;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = خالی_اشاریہ;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_سیٹ_TYPE(0, FASTRPC_NSP);

    صحیح err = remote_system_request(&domain_info);
    اگر (err == AEE_SUCCESS && domain_info.sys.عدد_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.عدد_domains);
        domain_info.sys.domains     = domains.ڈیٹا();
        domain_info.sys.max_domains = (صحیح) domains.size();

        err = remote_system_request(&domain_info);
        اگر (err == AEE_SUCCESS) {
            discovery_supported = سچ;
            ثابت صحیح n_domains = std::min(domain_info.sys.عدد_domains, (صحیح) domains.size());
            برائے (صحیح i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (صحیح) domains[i].type, domains[i].id, domains[i].name, domains[i].حالت_2, domains[i].instance_id);
                اگر (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (صحیح) domains[i].type);
                    جاری;
                }
                اگر (!domains[i].حالت_2) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    جاری;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } ورنہ {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (غیرعلامتی) err);
        }
    } ورنہ اگر (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (غیرعلامتی) err);
    }

    // Populate domain IDs and names for all configured devices
    برائے (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ساکن خالی ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q4_0 == (غیرعلامتی صحیح) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q4_1 == (غیرعلامتی صحیح) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q8_0 == (غیرعلامتی صحیح) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_MXFP4 == (غیرعلامتی صحیح) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_IQ4_NL == (غیرعلامتی صحیح) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q4_K == (غیرعلامتی صحیح) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q5_K == (غیرعلامتی صحیح) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q6_K == (غیرعلامتی صحیح) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q3_K == (غیرعلامتی صحیح) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q2_K == (غیرعلامتی صحیح) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q1_0 == (غیرعلامتی صحیح) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_Q2_0 == (غیرعلامتی صحیح) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((غیرعلامتی صحیح) HTP_TYPE_BF16 == (غیرعلامتی صحیح) GGML_TYPE_BF16,
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
    ثابت حرف * str_fa_سر_بانٹو = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
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
    ثابت حرف * str_power_منوال = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    اگر (!str_arch) {
        صحیح err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } ورنہ {
            اگر (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } ورنہ اگر (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } ورنہ {
        اگر (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, تہی, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    خودکار RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? نیا std::regex(str_opfilter, RE_ICASE) : تہی;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, تہی, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, تہی, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, تہی, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  تہی, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    اگر (str_power_منوال) {
        صحیح m = htp_power_منوال_from_str(str_power_منوال);
        اگر (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_منوال);
        } ورنہ {
            opt_power_منوال = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, تہی, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_سر_بانٹو = str_fa_سر_بانٹو ? atoi(str_fa_سر_بانٹو)       : opt_fa_سر_بانٹو;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    اگر (str_mbuf) {
        ثابت حرف * p = str_mbuf;
        برائے (صحیح idx = 0; idx < 3 && p && *p; idx++) {
            جب (*p == ' ') p++;
            ثابت حرف * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : لڑی_لمبائی(p);
            جب (len > 0 && p[len - 1] == ' ') len--;
            اگر (len > 0) {
                std::string token(p, len);
                اگر (idx == 0) opt_mbuf_dyn    = parse_سائز(token.c_str());
                اگر (idx == 1) opt_mbuf_static = parse_سائز(token.c_str());
                اگر (idx == 2) opt_mbuf_کل  = parse_سائز(token.c_str());
            }
            اگر (!comma) توڑو;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_سائز(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ثابت حرف * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    اگر (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    اگر (str_devices && str_devices[0] != '\0') {
        منطقی is_single_نمبر = سچ;
        برائے (صحیح i = 0; str_devices[i] != '\0'; i++) {
            اگر (!ہندسہ_ہے((غیرعلامتی حرف)str_devices[i])) {
                is_single_نمبر = جھوٹ;
                توڑو;
            }
        }
        اگر (is_single_نمبر) {
            صحیح n = atoi(str_devices);
            اگر (n < 1) n = 1;
            اگر (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            برائے (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (صحیح)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_گروہ.clear();
            }
        } ورنہ {
            std::string s_devices(str_devices);
            std::vector<std::string> اشیاء;
            std::string curr_شے;
            صحیح bracket_گہرائی = 0;
            برائے (حرف ch : s_devices) {
                اگر (ch == '[') {
                    bracket_گہرائی++;
                    curr_شے += ch;
                } ورنہ اگر (ch == ']') {
                    اگر (bracket_گہرائی > 0) bracket_گہرائی--;
                    curr_شے += ch;
                } ورنہ اگر (ch == ',' && bracket_گہرائی == 0) {
                    size_t s = curr_شے.find_first_not_of(" \t\r\n");
                    size_t e = curr_شے.find_last_not_of(" \t\r\n");
                    اگر (s != std::string::npos) {
                        اشیاء.push_back(curr_شے.substr(s, e - s + 1));
                    }
                    curr_شے.clear();
                } ورنہ {
                    curr_شے += ch;
                }
            }
            size_t s = curr_شے.find_first_not_of(" \t\r\n");
            size_t e = curr_شے.find_last_not_of(" \t\r\n");
            اگر (s != std::string::npos) {
                اشیاء.push_back(curr_شے.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            برائے (ثابت خودکار & شے : اشیاء) {
                size_t b_open  = شے.find('[');
                size_t b_close = شے.rfind(']');

                اگر (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = شے.substr(0, b_open);
                    std::string content  = شے.substr(b_open + 1, b_close - b_open - 1);

                    صحیح virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    اگر (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        کوشش {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } پکڑو (...) {
                            virt = 0;
                        }
                    } ورنہ {
                        size_t dev_colon = dev_name.find(':');
                        اگر (dev_colon != std::string::npos) {
                            کوشش {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } پکڑو (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<صحیح> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    جب (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        اگر (ps == std::string::npos) جاری;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        اگر (dash_pos != std::string::npos) {
                            کوشش {
                                صحیح p_آغاز = std::stoi(p_part.substr(0, dash_pos));
                                صحیح p_انتہا   = std::stoi(p_part.substr(dash_pos + 1));
                                برائے (صحیح p = p_آغاز; p <= p_انتہا; p++) {
                                    اگر (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } پکڑو (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } ورنہ {
                            کوشش {
                                صحیح p = std::stoi(p_part);
                                اگر (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } پکڑو (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    اگر (phys_list.خالی_خانے()) {
                        phys_list.push_back(0);
                    }

                    اگر (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        خودکار & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_گروہ.clear();

                        برائے (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_گروہ.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } ورنہ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, شے.c_str());
                    }
                } ورنہ اگر (شے.rfind("HTP", 0) == 0) {
                    std::string rest = شے.substr(3);
                    size_t colon_pos = rest.find(':');
                    صحیح phys = 0;
                    صحیح virt = 0;
                    کوشش {
                        اگر (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } ورنہ {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } پکڑو (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", شے.c_str());
                        جاری;
                    }

                    اگر (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_گروہ.clear();
                        opt_ndev++;
                    } ورنہ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, شے.c_str());
                    }
                } ورنہ {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", شے.c_str());
                }
            }
        }
    } ورنہ {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_گروہ.clear();
    }

#اگر defined(__ANDROID__)
    اگر (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#اگر_ختم

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    اگر (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            خودکار v  = str_مقصد_vec<uint32_t>(str_profile);
            انتخاب (v.size()) {
                حالت 1:  opt_profile = v[0]; واپس opt_pmu_evt; // mode with default pmu events
                حالت 8:  opt_profile = 2;    واپس v;           // mode with custom  pmu events
                طےشدہ: opt_profile = 0;    واپس {};          // garbage input
            }}();
        اگر (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_مقصد_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = نیا ggml_hexagon_registry(reg);
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(خالی) {
    ساکن منطقی initialized = جھوٹ;

    ساکن ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_نسخہ,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ تہی };

    {
        ساکن std::mutex           mutex;
        std::lock_guard<std::mutex> تالا(mutex);
        اگر (!initialized) {
            خودکار nErr = htpdrv_init();
            اگر (nErr != AEE_SUCCESS) {
                واپس تہی;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = سچ;
    }

    واپس &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
