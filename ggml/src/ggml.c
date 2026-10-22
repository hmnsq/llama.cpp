#تعریف_کرو _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#تعریف_کرو USE_MATH_متعین_کرتا // For M_PI on MSVC

#شامل "ggml-version.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "ggml-threading.h"
#شامل "ggml-cpu.h"
#شامل "ggml.h"

// FIXME: required here for quantization functions
#شامل "ggml-quants.h"

#اگر_تعریف_شدہ GGML_USE_CPU_HBM
#شامل <hbwmalloc.h>
#اگر_ختم

#اگر defined(_MSC_VER) || defined(__MINGW32__)
#شامل <malloc.h> // using malloc.h with MSC/MINGW
#وگرنہ_اگر !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#شامل <alloca.h>
#اگر_ختم

#شامل <assert.h>
#شامل <errno.h>
#شامل <time.h>
#شامل <math.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <stdint.h>
#شامل <inttypes.h>
#شامل <stdio.h>
#شامل <float.h>
#شامل <limits.h>
#شامل <stdarg.h>
#شامل <signal.h>
#اگر defined(__gnu_linux__)
#شامل <syscall.h>
#اگر_ختم

#اگر defined(__APPLE__)
#شامل <unistd.h>
#شامل <mach/mach.h>
#شامل <TargetConditionals.h>
#اگر_ختم

#اگر defined(_WIN32)
#تعریف_کرو WIN32_LEAN_AND_اوسط
#اگر_غیر_تعریف_شدہ NOMINMAX
    #تعریف_کرو NOMINMAX
#اگر_ختم
#شامل <windows.h>
#اگر_ختم

#تعریف_کرو غیر_مستعمل GGML_غیر_مستعمل

uint64_t ggml_گراف_اگلا_uid(خالی) {
#اگر_تعریف_شدہ _MSC_VER
#اگر defined(_WIN32)
    ساکن متلون LONG شمار_2 = 1;
    واپس (uint64_t) InterlockedIncrement(&شمار_2) - 1;
#ورنہ
    ساکن متلون طویل طویل شمار_2 = 1;
    واپس (uint64_t) _InterlockedIncrement64(&شمار_2) - 1;
#اگر_ختم
#ورنہ
    ساکن uint64_t شمار_2 = 1;
    واپس __atomic_fetch_add(&شمار_2, 1, __ATOMIC_RELAXED);
#اگر_ختم
}

// Needed for ggml_fp32_to_bf16_row()
#اگر defined(__AVX512BF16__)
#اگر defined(_MSC_VER)
#تعریف_کرو m512i(p) p
#ورنہ
#شامل <immintrin.h>
#تعریف_کرو m512i(p) (__m512i)(p)
#اگر_ختم // defined(_MSC_VER)
#اگر_ختم // defined(__AVX512BF16__)

#اگر defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !ہدف_OS_TV && !ہدف_OS_WATCH)

#شامل <unistd.h>
#شامل <sys/types.h>
#شامل <sys/stat.h>
#شامل <sys/wait.h>
#اگر defined(__linux__)
#شامل <sys/prctl.h>
#اگر_ختم

#اگر defined(__ANDROID__)
#شامل <unwind.h>
#شامل <dlfcn.h>
#شامل <stdio.h>

ساخت backtrace_state {
    خالی ** current;
    خالی ** انتہا;
};

ساکن Unwind_Reason_رمز unwind_callback(ساخت _Unwind_Context* context, خالی* arg) {
    ساخت backtrace_state * state = (ساخت backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    اگر (pc) {
        اگر (state->current == state->انتہا) {
            واپس URC_انتہا_کا_ڈھیر;
        } ورنہ {
            *state->current++ = (خالی*)pc;
        }
    }
    واپس _URC_NO_REASON;
}

ساکن خالی ggml_print_backtrace_symbols(خالی) {
    ثابت صحیح max = 100;
    خالی* بفر[max];

    ساخت backtrace_state state = {بفر, بفر + max};
    _Unwind_Backtrace(unwind_callback, &state);

    صحیح گنتی = state.current - بفر;

    برائے (صحیح idx = 0; idx < گنتی; ++idx) {
        ثابت خالی * addr = بفر[idx];
        ثابت حرف * علامت = "";

        Dl_info info;
        اگر (dladdr(addr, &info) && info.dli_sname) {
            علامت = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, علامت);
    }
}
#وگرنہ_اگر defined(__linux__) && defined(__GLIBC__)
#شامل <execinfo.h>
ساکن خالی ggml_print_backtrace_symbols(خالی) {
    خالی * trace[100];
    صحیح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وگرنہ_اگر defined(__APPLE__)
#شامل <execinfo.h>
ساکن خالی ggml_print_backtrace_symbols(خالی) {
    خالی * trace[100];
    صحیح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ورنہ
ساکن خالی ggml_print_backtrace_symbols(خالی) {
    // platform not supported
}
#اگر_ختم

خالی ggml_print_backtrace(خالی) {
    ثابت حرف * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    اگر (GGML_NO_BACKTRACE) {
        واپس;
    }
#اگر defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ثابت حرف * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    اگر (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        واپس;
    }
#اگر_ختم
#اگر defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t سائز = 0;
    حرف * سطر = تہی;
    ssize_t لمبائی = 0;
    جب ((لمبائی = getline(&سطر, &سائز, f)) > 0) {
        اگر (!لڑی_موازنہ(سطر, "TracerPid:", حجم("TracerPid:") - 1) &&
            (لمبائی != حجم("TracerPid:\t0\n") - 1 || سطر[لمبائی - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            آزاد(سطر);
            fclose(f);
            واپس;
        }
    }
    آزاد(سطر);
    fclose(f);
    صحیح تالا[2] = { -1, -1 };
    (خالی) !pipe(تالا); // Don't start gdb until after PR_SET_PTRACER
#اگر_ختم
    ثابت صحیح والد_pid = getpid();
    ثابت صحیح بچہ_pid = fork();
    اگر (بچہ_pid < 0) { // error
#اگر defined(__linux__)
        close(تالا[1]);
        close(تالا[0]);
#اگر_ختم
        واپس;
    } ورنہ اگر (بچہ_pid == 0) { //بچہ
        حرف attach[32];
        snprintf(attach, حجم(attach), "attach %d", والد_pid);
#اگر defined(__linux__)
        close(تالا[1]);
        (خالی) !اقرأ(تالا[0], تالا, 1);
        close(تالا[0]);
#اگر_ختم
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (حرف *) تہی);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[حجم("attach ") - 1],
            (حرف *) تہی);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } ورنہ { //والد
#اگر defined(__linux__)
        prctl(PR_سیٹ_PTRACER, بچہ_pid);
        close(تالا[1]);
        close(تالا[0]);
#اگر_ختم
        waitpid(بچہ_pid, تہی, 0);
    }
}
#ورنہ
خالی ggml_print_backtrace(خالی) {
    // platform not supported
}
#اگر_ختم

ساکن ggml_abort_callback_t g_abort_callback = تہی;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_سیٹ_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    واپس ret_val;
}

خالی ggml_abort(ثابت حرف * file, صحیح سطر, ثابت حرف * fmt, ...) {
    fflush(stdout);

    حرف پیغام[2048];
    صحیح انحراف = snprintf(پیغام, حجم(پیغام), "%s:%d: ", file, سطر);

    va_list دلائل_2;
    va_start(دلائل_2, fmt);
    vsnprintf(پیغام + انحراف, حجم(پیغام) - انحراف, fmt, دلائل_2);
    va_end(دلائل_2);

    اگر (g_abort_callback) {
        g_abort_callback(پیغام);
    } ورنہ {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", پیغام);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

ساخت ggml_logger_state {
    ggml_log_callback log_callback;
    خالی * log_callback_user_ڈیٹا;
};
ساکن ساخت ggml_logger_state g_logger_state = {ggml_log_callback_default, تہی};

ساکن خالی ggml_log_اندرونی_v(شمار ggml_log_level level, ثابت حرف * format, va_list دلائل_2) {
    اگر (format == تہی) {
        واپس;
    }
    va_list دلائل_copy;
    va_copy(دلائل_copy, دلائل_2);
    حرف بفر[128];
    صحیح len = vsnprintf(بفر, 128, format, دلائل_2);
    اگر (len < 128) {
        g_logger_state.log_callback(level, بفر, g_logger_state.log_callback_user_ڈیٹا);
    } ورنہ {
        حرف * buffer2 = (حرف *) صفر_حافظہ_دو(len + 1, حجم(حرف));
        vsnprintf(buffer2, len + 1, format, دلائل_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_ڈیٹا);
        آزاد(buffer2);
    }
    va_end(دلائل_copy);
}

خالی ggml_log_اندرونی(شمار ggml_log_level level, ثابت حرف * format, ...) {
    va_list دلائل_2;
    va_start(دلائل_2, format);
    ggml_log_اندرونی_v(level, format, دلائل_2);
    va_end(دلائل_2);
}

خالی ggml_log_callback_default(شمار ggml_log_level level, ثابت حرف * متن_2, خالی * user_ڈیٹا) {
    (خالی) level;
    (خالی) user_ڈیٹا;
    fputs(متن_2, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#اگر_تعریف_شدہ GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#اگر_ختم


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#اگر defined(__has_feature)
#اگر __has_feature(thread_sanitizer)
#تعریف_کرو GGML_TSAN_ENABLED 1
#اگر_ختم
#ورنہ
#اگر defined(__SANITIZE_THREAD__)
#تعریف_کرو GGML_TSAN_ENABLED 1
#اگر_ختم
#اگر_ختم

خالی * ggml_aligned_malloc(size_t سائز) {
#اگر defined(__s390x__)
    ثابت صحیح alignment = 256;
#ورنہ
    ثابت صحیح alignment = 64;
#اگر_ختم

#اگر defined(_MSC_VER) || defined(__MINGW32__)
    واپس _aligned_malloc(سائز, alignment);
#ورنہ
    اگر (سائز == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        واپس تہی;
    }
    خالی * aligned_حافظہ = تہی;
  #اگر_تعریف_شدہ GGML_USE_CPU_HBM
    صحیح نتیجہ = hbw_posix_memalign(&aligned_حافظہ, alignment, سائز);
  #وگرنہ_اگر ہدف_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_غیر_مستعمل(alignment);
    kern_return_t alloc_حالت = vm_allocate((vm_map_t) mach_task_self(), (vm_پتہ_t *) &aligned_حافظہ, سائز, VM_FLAGS_ANYWHERE);
    صحیح نتیجہ = EFAULT;
    انتخاب (alloc_حالت) {
        حالت KERN_SUCCESS:
            نتیجہ = 0;
            توڑو;
        حالت KERN_INVALID_پتہ:
            نتیجہ = EINVAL;
            توڑو;
        حالت KERN_NO_SPACE:
            نتیجہ = ENOMEM;
            توڑو;
        طےشدہ:
            نتیجہ = EFAULT;
            توڑو;
    }
  #ورنہ
    صحیح نتیجہ = posix_memalign(&aligned_حافظہ, alignment, سائز);
  #اگر_ختم
    اگر (نتیجہ != 0) {
        // Handle allocation failure
        ثابت حرف *error_desc = "unknown allocation error";
        انتخاب (نتیجہ) {
            حالت EINVAL:
                error_desc = "invalid alignment value";
                توڑو;
            حالت ENOMEM:
                error_desc = "insufficient memory";
                توڑو;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, سائز/(1024.0*1024.0));
        واپس تہی;
    }
    واپس aligned_حافظہ;
#اگر_ختم
}

خالی ggml_aligned_free(خالی * ptr, size_t سائز) {
    GGML_غیر_مستعمل(سائز);
#اگر defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#وگرنہ_اگر GGML_USE_CPU_HBM
    اگر (ptr != تہی) {
        hbw_free(ptr);
    }
#وگرنہ_اگر ہدف_OS_OSX && !defined(GGML_TSAN_ENABLED)
    اگر (ptr != تہی) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_پتہ_t)ptr, سائز);
    }
#ورنہ
    آزاد(ptr);
#اگر_ختم
}


اندرونی ساکن خالی * ggml_malloc(size_t سائز) {
    اگر (سائز == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        واپس تہی;
    }
    خالی * نتیجہ = حافظہ_دو(سائز);
    اگر (نتیجہ == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, سائز/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    واپس نتیجہ;
}

// calloc
اندرونی ساکن خالی * ggml_calloc(size_t عدد, size_t سائز) {
    اگر (عدد == 0 || سائز == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        واپس تہی;
    }
    خالی * نتیجہ = صفر_حافظہ_دو(عدد, سائز);
    اگر (نتیجہ == تہی) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, سائز/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    واپس نتیجہ;
}

#تعریف_کرو GGML_MALLOC(سائز)      ggml_malloc(سائز)
#تعریف_کرو GGML_CALLOC(عدد, سائز) ggml_calloc(عدد, سائز)

#تعریف_کرو GGML_FREE(ptr) آزاد(ptr)

ثابت حرف * ggml_حالت_مقصد_string(شمار ggml_حالت حالت_2) {
    انتخاب (حالت_2) {
        حالت GGML_حالت_ALLOC_FAILED: واپس "GGML status: error (failed to allocate memory)";
        حالت GGML_حالت_FAILED:       واپس "GGML status: error (operation failed)";
        حالت GGML_حالت_SUCCESS:      واپس "GGML status: success";
        حالت GGML_حالت_ABORTED:      واپس "GGML status: warning (operation aborted)";
    }

    واپس "GGML status: unknown";
}

اعشاری ggml_fp16_مقصد_fp32(ggml_fp16_t x) {
#تعریف_کرو ggml_fp16_مقصد_fp32 do_not_use_ggml_fp16_مقصد_fp32_اندر_ggml
    واپس GGML_FP16_مقصد_FP32(x);
}

ggml_fp16_t ggml_fp32_مقصد_fp16(اعشاری x) {
#تعریف_کرو ggml_fp32_مقصد_fp16 do_not_use_ggml_fp32_مقصد_fp16_اندر_ggml
    واپس GGML_FP32_مقصد_FP16(x);
}

اعشاری ggml_bf16_مقصد_fp32(ggml_bf16_t x) {
#تعریف_کرو ggml_bf16_مقصد_fp32 do_not_use_ggml_bf16_مقصد_fp32_اندر_ggml
    واپس GGML_BF16_مقصد_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_مقصد_bf16(اعشاری x) {
#تعریف_کرو ggml_fp32_مقصد_bf16 do_not_use_ggml_fp32_مقصد_bf16_اندر_ggml
    واپس GGML_FP32_مقصد_BF16(x);
}

خالی ggml_fp16_مقصد_fp32_صف(ثابت ggml_fp16_t * x, اعشاری * y, int64_t n) {
    برائے (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_مقصد_FP32(x[i]);
    }
}

خالی ggml_fp32_مقصد_fp16_صف(ثابت اعشاری * x, ggml_fp16_t * y, int64_t n) {
    صحیح i = 0;
    برائے (; i < n; ++i) {
        y[i] = GGML_FP32_مقصد_FP16(x[i]);
    }
}

خالی ggml_bf16_مقصد_fp32_صف(ثابت ggml_bf16_t * x, اعشاری * y, int64_t n) {
    صحیح i = 0;
    برائے (; i < n; ++i) {
        y[i] = GGML_BF16_مقصد_FP32(x[i]);
    }
}

خالی ggml_fp32_مقصد_bf16_صف_ref(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
    برائے (صحیح i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_مقصد_bf16(x[i]);
    }
}

خالی ggml_fp32_مقصد_bf16_صف(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
  صحیح i = 0;
#اگر defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  برائے (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#اگر_ختم
    برائے (; i < n; i++) {
        y[i] = GGML_FP32_مقصد_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    واپس memcmp(guid_a, guid_b, حجم(ggml_guid)) == 0;
}

ثابت حرف * ggml_نسخہ(خالی) {
    واپس GGML_نسخہ;
}

ثابت حرف * ggml_commit(خالی) {
    واپس GGML_COMMIT;
}

//
// timing
//

#اگر defined(_MSC_VER) || defined(__MINGW32__)
ساکن int64_t timer_freq, timer_آغاز;
ساکن BOOL CALLBACK ggml_وقت_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    غیر_مستعمل(once);
    غیر_مستعمل(param);
    غیر_مستعمل(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    Query_Performance_شمار(&t);
    timer_آغاز = t.QuadPart;

    واپس TRUE;
}
خالی ggml_وقت_init(خالی) {
    ساکن INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_وقت_init_once, تہی, تہی);
}
int64_t ggml_وقت_ms(خالی) {
    LARGE_INTEGER t;
    Query_Performance_شمار(&t);
    واپس ((t.QuadPart-timer_آغاز) * 1000) / timer_freq;
}
int64_t ggml_وقت_us(خالی) {
    LARGE_INTEGER t;
    Query_Performance_شمار(&t);
    واپس ((t.QuadPart-timer_آغاز) * 1000000) / timer_freq;
}
#ورنہ
خالی ggml_وقت_init(خالی) {}
int64_t ggml_وقت_ms(خالی) {
    ساخت timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    واپس (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_وقت_us(خالی) {
    ساخت timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    واپس (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#اگر_ختم

int64_t ggml_ادوار(خالی) {
    واپس clock();
}

int64_t ggml_ادوار_فی_ms(خالی) {
    واپس CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#اگر_تعریف_شدہ _WIN32
ساکن wchar_t * ggml_mbstowcs(ثابت حرف * mbs) {
    صحیح wlen = Multi_Byte_مقصد_Wide_Char(CP_UTF8, 0, mbs, -1, تہی, 0);
    اگر (!wlen) {
        errno = EINVAL;
        واپس تہی;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * حجم(wchar_t));
    wlen = Multi_Byte_مقصد_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    اگر (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        واپس تہی;
    }

    واپس wbuf;
}
#اگر_ختم

FILE * ggml_fopen(ثابت حرف * fname, ثابت حرف * منوال) {
#اگر_تعریف_شدہ _WIN32
    FILE * file = تہی;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    اگر (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(منوال);
        اگر (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    واپس file;
#ورنہ
    واپس fopen(fname, منوال);
#اگر_ختم

}

ساکن ثابت ساخت ggml_type_traits type_traits[GGML_TYPE_گنتی] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_سائز                = 1,
        .type_سائز                = حجم(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_سائز                = 1,
        .type_سائز                = حجم(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_سائز                = 1,
        .type_سائز                = حجم(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_سائز                = 1,
        .type_سائز                = حجم(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_سائز                = 1,
        .type_سائز                = حجم(دوہرا),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_سائز                = 1,
        .type_سائز                = حجم(اعشاری),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_سائز                = 1,
        .type_سائز                = حجم(ggml_fp16_t),
        .is_quantized             = false,
        .مقصد_float                 = (ggml_مقصد_float_t) ggml_fp16_مقصد_fp32_صف,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_مقصد_fp16_صف,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_سائز                = QK1_0,
        .type_سائز                = حجم(بلاک_q1_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_سائز                = QK2_0,
        .type_سائز                = حجم(بلاک_q2_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_سائز                = QK4_0,
        .type_سائز                = حجم(بلاک_q4_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_سائز                = QK4_1,
        .type_سائز                = حجم(بلاک_q4_1),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_سائز                = QK5_0,
        .type_سائز                = حجم(بلاک_q5_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_سائز                = QK5_1,
        .type_سائز                = حجم(بلاک_q5_1),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_سائز                = QK8_0,
        .type_سائز                = حجم(بلاک_q8_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_سائز                = QK8_1,
        .type_سائز                = حجم(بلاک_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_سائز                = QK_MXFP4,
        .type_سائز                = حجم(بلاک_mxfp4),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_سائز                = QK_NVFP4,
        .type_سائز                = حجم(بلاک_nvfp4),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q2_K),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q3_K),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q4_K),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q5_K),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q6_K),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq2_xxs),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq2_xxs,
        .from_float_ref           = تہی,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq2_xs),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq2_xs,
        .from_float_ref           = تہی,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq3_xxs),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq3_s),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq2_s),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq1_s),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq1_s,
        .from_float_ref           = تہی,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq1_m),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq1_m,
        .from_float_ref           = تہی,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_سائز                = QK4_NL,
        .type_سائز                = حجم(بلاک_iq4_nl),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_iq4_xs),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_سائز                = 1,
        .type_سائز                = حجم(ggml_bf16_t),
        .is_quantized             = false,
        .مقصد_float                 = (ggml_مقصد_float_t) ggml_bf16_مقصد_fp32_صف,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_مقصد_bf16_صف_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_tq1_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_سائز                = QK_K,
        .type_سائز                = حجم(بلاک_tq2_0),
        .is_quantized             = true,
        .مقصد_float                 = (ggml_مقصد_float_t) dequantize_صف_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_سائز                = 0,
        .type_سائز                = 0,
        .is_quantized             = false,
    },
};

ثابت ساخت ggml_type_traits * ggml_get_type_traits(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس &type_traits[type];
}

//
// ggml object
//

ساخت ggml_object {
    size_t offs;
    size_t سائز;

    ساخت ggml_object * اگلا;

    شمار ggml_object_type type;

    حرف padding[4];
};

ساکن ثابت size_t GGML_OBJECT_سائز = حجم(ساخت ggml_object);

//
// ggml context
//

ساخت ggml_context {
    size_t mem_سائز;
    خالی * mem_بفر;
    bool   mem_بفر_owned;
    bool   no_alloc;

    صحیح    n_objects;

    ساخت ggml_object * objects_begin;
    ساخت ggml_object * objects_انتہا;
};

//
// data types
//

ساکن ثابت حرف * GGML_OP_NAME[GGML_OP_گنتی] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "CUMSUM",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",
    "L2_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "SET_ROWS",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "IM2COL_3D",
    "COL2IM_1D",
    "CONV_2D",
    "CONV_3D",
    "CONV_2D_DW",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "PAD_REFLECT_1D",
    "ROLL",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "TOP_K",
    "LEAKY_RELU",
    "TRI",
    "FILL",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",
    "GATED_LINEAR_ATTN",
    "RWKV_WKV7",
    "SOLVE_TRI",
    "GATED_DELTA_NET",
    "LIGHTNING_INDEXER",
    "DSV4_HC_COMB",
    "DSV4_HC_PRE",
    "DSV4_HC_POST",
    "TOP_K_MASK",

    "UNARY",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CUSTOM",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",

    "GLU",
};

static_assert(GGML_OP_گنتی == 102, "GGML_OP_COUNT != 102");

ساکن ثابت حرف * GGML_OP_علامت[GGML_OP_گنتی] = {
    "none",

    "x",
    "x+y",
    "x[i]+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "sin(x)",
    "cos(x)",
    "Σx",
    "Σx_k",
    "cumsum(x)",
    "Σx/n",
    "argmax(x)",
    "count_equal(x)",
    "repeat(x)",
    "repeat_back(x)",
    "concat(x, y)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",
    "group_norm(x)",
    "l2_norm(x)",

    "X*Y",
    "X[i]*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "set_rows(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "clamp(x)",
    "conv_transpose_1d(x)",
    "im2col(x)",
    "im2col_back(x)",
    "im2col_3d(x)",
    "col2im_1d(x)",
    "conv_2d(x)",
    "conv_3d(x)",
    "conv_2d_dw(x)",
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
    "pool_2d_back(x)",
    "upscale(x)",
    "pad(x)",
    "pad_reflect_1d(x)",
    "roll(x)",
    "arange(start, stop, step)",
    "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",
    "top_k(x)",
    "leaky_relu(x)",
    "tri(x)",
    "fill(x, c)",

    "flash_attn_ext(x)",
    "flash_attn_back(x)",
    "ssm_conv(x)",
    "ssm_scan(x)",
    "win_part(x)",
    "win_unpart(x)",
    "get_rel_pos(x)",
    "add_rel_pos(x)",
    "rwkv_wkv6(k, v, r, tf, td, s)",
    "gated_linear_attn(k, v, q, gate, s)",
    "rwkv_wkv7(r, w, k, v, a, b, s)",
    "A X = B, A triangular, solve X",
    "gated_delta_net(q, k, v, g, beta, s)",
    "lightning_indexer(q, k, weights, mask)",
    "dsv4_hc_comb(mixes, scale, base)",
    "dsv4_hc_pre(x, weights)",
    "dsv4_hc_post(x, residual, post, comb)",
    "top_k_mask(mask, top_k, score, blocks, extra)",

    "unary(x)",

    "map_custom(x)",
    "map_custom(x,y)",
    "map_custom(x,y,z)",

    "custom(x)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
    "adamw(x)",
    "sgd(x)",

    "glu(x)",
};

static_assert(GGML_OP_گنتی == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_گنتی == 2, "GGML_OP_POOL_COUNT != 2");

ساکن ثابت حرف * GGML_UNARY_OP_NAME[GGML_UNARY_OP_گنتی] = {
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "TANH",
    "ELU",
    "RELU",
    "SIGMOID",
    "GELU",
    "GELU_QUICK",
    "SILU",
    "HARDSWISH",
    "HARDSIGMOID",
    "EXP",
    "EXPM1",
    "SOFTPLUS",
    "GELU_ERF",
    "XIELU",
    "FLOOR",
    "CEIL",
    "ROUND",
    "TRUNC",
};

static_assert(GGML_UNARY_OP_گنتی == 22, "GGML_UNARY_OP_COUNT != 22");

ساکن ثابت حرف * GGML_GLU_OP_NAME[GGML_GLU_OP_گنتی] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_گنتی == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(حجم(ساخت ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(حجم(ساخت ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

خالی ggml_print_object(ثابت ساخت ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->سائز, (ثابت خالی *) obj->اگلا);
}

خالی ggml_print_objects(ثابت ساخت ggml_context * ctx) {
    ساخت ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ثابت خالی *) ctx);

    جب (obj != تہی) {
        ggml_print_object(obj);
        obj = obj->اگلا;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ثابت ساخت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        اگر (tensor->ne[i] <= 0) {
            واپس 0;
        }
    }

    size_t nbytes;
    ثابت size_t blck_سائز = ggml_blck_سائز(tensor->type);
    اگر (blck_سائز == 1) {
        nbytes = ggml_type_سائز(tensor->type);
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    ورنہ {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_سائز;
        برائے (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    واپس nbytes;
}

size_t ggml_nbytes_pad(ثابت ساخت ggml_tensor * tensor) {
    واپس GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_سائز(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس type_traits[type].blck_سائز;
}

size_t ggml_type_سائز(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس type_traits[type].type_سائز;
}

size_t ggml_صف_سائز(شمار ggml_type type, int64_t ne) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أكد(ne % ggml_blck_سائز(type) == 0);
    واپس ggml_type_سائز(type)*ne/ggml_blck_سائز(type);
}

دوہرا ggml_type_sizef(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس ((دوہرا)(type_traits[type].type_سائز))/type_traits[type].blck_سائز;
}

ثابت حرف * ggml_type_name(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس type_traits[type].type_name;
}

bool ggml_is_quantized(شمار ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    واپس type_traits[type].is_quantized;
}

ثابت حرف * ggml_op_name(شمار ggml_op op) {
    واپس GGML_OP_NAME[op];
}

ثابت حرف * ggml_op_علامت(شمار ggml_op op) {
    واپس GGML_OP_علامت[op];
}

ثابت حرف * ggml_unary_op_name(شمار ggml_unary_op op) {
    واپس GGML_UNARY_OP_NAME[op];
}

ثابت حرف * ggml_glu_op_name(شمار ggml_glu_op op) {
    واپس GGML_GLU_OP_NAME[op];
}

ثابت حرف * ggml_op_desc(ثابت ساخت ggml_tensor * t) {
    اگر (t->op == GGML_OP_UNARY) {
        شمار ggml_unary_op uop = ggml_get_unary_op(t);
        واپس ggml_unary_op_name(uop);
    }
    اگر (t->op == GGML_OP_GLU) {
        شمار ggml_glu_op gop = ggml_get_glu_op(t);
        واپس ggml_glu_op_name(gop);
    }
    واپس ggml_op_name(t->op);
}

size_t ggml_element_سائز(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_type_سائز(tensor->type);
}

bool ggml_is_scalar(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_سمتیہ(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_میٹرکس(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ثابت ساخت ggml_tensor * tensor) {
    واپس tensor->ne[3] == 1;
}

صحیح ggml_n_dims(ثابت ساخت ggml_tensor * tensor) {
    برائے (صحیح i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        اگر (tensor->ne[i] > 1) {
            واپس i + 1;
        }
    }
    واپس 1;
}

شمار ggml_type ggml_ftype_مقصد_ggml_type(شمار ggml_ftype ftype) {
    شمار ggml_type wtype = GGML_TYPE_گنتی;

    انتخاب (ftype) {
        حالت GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   توڑو;
        حالت GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   توڑو;
        حالت GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  توڑو;
        حالت GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; توڑو;
        حالت GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; توڑو;
        حالت GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  توڑو;
        حالت GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    توڑو;
        حالت GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    توڑو;
        حالت GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_گنتی; توڑو;
        حالت GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_گنتی; توڑو;
    }

    GGML_ASSERT(wtype != GGML_TYPE_گنتی);

    واپس wtype;
}

size_t ggml_tensor_overhead(خالی) {
    واپس GGML_OBJECT_سائز + GGML_TENSOR_سائز;
}

bool ggml_is_transposed(ثابت ساخت ggml_tensor * tensor) {
    واپس tensor->nb[0] > tensor->nb[1];
}

ساکن bool ggml_is_contiguous_m_n(ثابت ساخت ggml_tensor * tensor, صحیح m, صحیح n) {
    size_t اگلا_nb = ggml_type_سائز(tensor->type);
    اگر (tensor->ne[0] != ggml_blck_سائز(tensor->type) && tensor->nb[0] != اگلا_nb) {
        واپس false;
    }
    اگلا_nb *= tensor->ne[0]/ggml_blck_سائز(tensor->type);
    برائے (صحیح i = 1; i < n; i++) {
        اگر (i > m) {
            اگر (tensor->ne[i] != 1 && tensor->nb[i] != اگلا_nb) {
                واپس false;
            }
            اگلا_nb *= tensor->ne[i];
        } ورنہ {
            // this dimension does not need to be contiguous
            اگلا_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    واپس true;
}

bool ggml_is_contiguous(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_مقصد_1(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_مقصد_2(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_مقصد_3(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_سائز(tensor->type)/ggml_blck_سائز(tensor->type);
}

bool ggml_is_permuted(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ثابت ساخت ggml_tensor * tensor) {
    واپس
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_سائز(tensor->type);
}

bool ggml_is_contiguous_صفیں(ثابت ساخت ggml_tensor * tensor) {
    واپس
        tensor->ne[0] == ggml_blck_سائز(tensor->type) ||
        tensor->nb[0] == ggml_type_سائز(tensor->type);
}

ساکن اندرونی bool ggml_is_padded_1d(ثابت ساخت ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس
        tensor->nb[0] == ggml_type_سائز(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_خالی_خانے(ثابت ساخت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        اگر (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            واپس true;
        }
    }
    واپس false;
}

bool ggml_are_same_shape(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ثابت ساخت ggml_tensor * t) {
    واپس ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس ggml_is_خالی_خانے(t0) ? ggml_is_خالی_خانے(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ساکن اندرونی bool ggml_can_repeat_صفیں(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#تعریف_کرو GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ساخت ggml_context * ggml_init(ساخت ggml_init_params params) {
    ساکن bool is_پہلا_call = true;

    ggml_critical_section_آغاز();

    اگر (is_پہلا_call) {
        // initialize time system (required on Windows)
        ggml_وقت_init();

        is_پہلا_call = false;
    }

    ggml_critical_section_انتہا();

    ساخت ggml_context * ctx = GGML_MALLOC(حجم(ساخت ggml_context));

    // allow to call ggml_init with 0 size
    اگر (params.mem_سائز == 0) {
        params.mem_سائز = GGML_MEM_ALIGN;
    }

    ثابت size_t mem_سائز = params.mem_بفر ? params.mem_سائز : GGML_PAD(params.mem_سائز, GGML_MEM_ALIGN);

    *ctx = (ساخت ggml_context) {
        /*.mem_size           =*/ mem_سائز,
        /*.mem_buffer         =*/ params.mem_بفر ? params.mem_بفر : ggml_aligned_malloc(mem_سائز),
        /*.mem_buffer_owned   =*/ params.mem_بفر ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ تہی,
        /*.objects_end        =*/ تہی,
    };

    GGML_ASSERT(ctx->mem_بفر != تہی);

    GGML_ASSERT_ALIGNED(ctx->mem_بفر);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    واپس ctx;
}

خالی ggml_reset(ساخت ggml_context * ctx) {
    اگر (ctx == تہی) {
        واپس;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = تہی;
    ctx->objects_انتہا   = تہی;
}

خالی ggml_free(ساخت ggml_context * ctx) {
    اگر (ctx == تہی) {
        واپس;
    }

    اگر (ctx->mem_بفر_owned) {
        ggml_aligned_free(ctx->mem_بفر, ctx->mem_سائز);
    }

    GGML_FREE(ctx);
}

size_t ggml_مستعمل_mem(ثابت ساخت ggml_context * ctx) {
    واپس ctx->objects_انتہا == تہی ? 0 : ctx->objects_انتہا->offs + ctx->objects_انتہا->سائز;
}

bool ggml_get_no_alloc(ساخت ggml_context * ctx) {
    واپس ctx->no_alloc;
}

خالی ggml_سیٹ_no_alloc(ساخت ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

خالی * ggml_get_mem_بفر(ثابت ساخت ggml_context * ctx) {
    واپس ctx->mem_بفر;
}

size_t ggml_get_mem_سائز(ثابت ساخت ggml_context * ctx) {
    واپس ctx->mem_سائز;
}

size_t ggml_get_max_tensor_سائز(ثابت ساخت ggml_context * ctx) {
    size_t max_سائز = 0;

    برائے (ساخت ggml_tensor * tensor = ggml_get_پہلا_tensor(ctx); tensor != تہی; tensor = ggml_get_اگلا_tensor(ctx, tensor)) {
        size_t بائٹیں = ggml_nbytes(tensor);
        max_سائز = MAX(max_سائز, بائٹیں);
    }

    واپس max_سائز;
}

////////////////////////////////////////////////////////////////////////////////

ساکن ساخت ggml_object * ggml_new_object(ساخت ggml_context * ctx, شمار ggml_object_type type, size_t سائز) {
    // always insert objects at the end of the context's memory pool
    ساخت ggml_object * obj_cur = ctx->objects_انتہا;

    ثابت size_t cur_offs = obj_cur == تہی ? 0 : obj_cur->offs;
    ثابت size_t cur_سائز = obj_cur == تہی ? 0 : obj_cur->سائز;
    ثابت size_t cur_انتہا  = cur_offs + cur_سائز;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(سائز <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t سائز_needed = GGML_PAD(سائز, GGML_MEM_ALIGN);

    حرف * ثابت mem_بفر = ctx->mem_بفر;
    ساخت ggml_object * ثابت obj_new = (ساخت ggml_object *)(mem_بفر + cur_انتہا);

    // integer overflow checks
    اگر (cur_انتہا > SIZE_MAX - سائز_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_انتہا, سائز_needed);
        واپس تہی;
    }
    اگر (cur_انتہا + سائز_needed > SIZE_MAX - GGML_OBJECT_سائز) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_انتہا, سائز_needed, (size_t) GGML_OBJECT_سائز);
        واپس تہی;
    }

    اگر (cur_انتہا + سائز_needed + GGML_OBJECT_سائز > ctx->mem_سائز) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_انتہا + سائز_needed + GGML_OBJECT_سائز, ctx->mem_سائز);
#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#اگر_ختم
        واپس تہی;
    }

    *obj_new = (ساخت ggml_object) {
        .offs = cur_انتہا + GGML_OBJECT_سائز,
        .سائز = سائز_needed,
        .اگلا = تہی,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_بفر + obj_new->offs);

    اگر (obj_cur != تہی) {
        obj_cur->اگلا = obj_new;
    } ورنہ {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_انتہا = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    واپس obj_new;
}

ساکن ساخت ggml_tensor * ggml_new_tensor_impl(
        ساخت ggml_context * ctx,
        شمار   ggml_type      type,
        صحیح                   n_dims,
        ثابت int64_t       * ne,
        ساخت ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_گنتی);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    اگر (view_src != تہی && view_src->view_src != تہی) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    برائے (صحیح i = 1; i < n_dims; i++) {
        اگر (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t ڈیٹا_سائز = ggml_صف_سائز(type, ne[0]);
    برائے (صحیح i = 1; i < n_dims; i++) {
        ڈیٹا_سائز *= ne[i];
    }

    GGML_ASSERT(view_src == تہی || ڈیٹا_سائز == 0 || ڈیٹا_سائز + view_offs <= ggml_nbytes(view_src));

    خالی * ڈیٹا = view_src != تہی ? view_src->ڈیٹا : تہی;
    اگر (ڈیٹا != تہی) {
        ڈیٹا = (حرف *) ڈیٹا + view_offs;
    }

    size_t obj_alloc_سائز = 0;

    اگر (view_src == تہی && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_سائز = ڈیٹا_سائز;
    }

    GGML_ASSERT(GGML_TENSOR_سائز <= SIZE_MAX - obj_alloc_سائز);

    ساخت ggml_object * ثابت obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_سائز + obj_alloc_سائز);
    GGML_ASSERT(obj_new);

    ساخت ggml_tensor * ثابت نتیجہ = (ساخت ggml_tensor *)((حرف *)ctx->mem_بفر + obj_new->offs);

    *نتیجہ = (ساخت ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ تہی,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { تہی },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_سائز > 0 ? (خالی *)(نتیجہ + 1) : ڈیٹا,
        /*.name         =*/ { 0 },
        /*.extra        =*/ تہی,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    برائے (صحیح i = 0; i < n_dims; i++) {
        نتیجہ->ne[i] = ne[i];
    }

    نتیجہ->nb[0] = ggml_type_سائز(type);
    نتیجہ->nb[1] = نتیجہ->nb[0]*(نتیجہ->ne[0]/ggml_blck_سائز(type));
    برائے (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = نتیجہ->nb[i - 1]*نتیجہ->ne[i - 1];
    }

    ctx->n_objects++;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_new_tensor(
        ساخت ggml_context * ctx,
        شمار   ggml_type      type,
        صحیح                   n_dims,
        ثابت int64_t       * ne) {
    واپس ggml_new_tensor_impl(ctx, type, n_dims, ne, تہی, 0);
}

ساخت ggml_tensor * ggml_new_tensor_1d(
        ساخت ggml_context * ctx,
        شمار   ggml_type      type,
        int64_t ne0) {
    واپس ggml_new_tensor(ctx, type, 1, &ne0);
}

ساخت ggml_tensor * ggml_new_tensor_2d(
        ساخت ggml_context * ctx,
        شمار   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ثابت int64_t ne[2] = { ne0, ne1 };
    واپس ggml_new_tensor(ctx, type, 2, ne);
}

ساخت ggml_tensor * ggml_new_tensor_3d(
        ساخت ggml_context * ctx,
        شمار   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    واپس ggml_new_tensor(ctx, type, 3, ne);
}

ساخت ggml_tensor * ggml_new_tensor_4d(
        ساخت ggml_context * ctx,
        شمار   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    واپس ggml_new_tensor(ctx, type, 4, ne);
}

خالی * ggml_new_بفر(ساخت ggml_context * ctx, size_t nbytes) {
    ساخت ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_بفر, nbytes);

    واپس (uint8_t *)ctx->mem_بفر + obj->offs;
}

ساخت ggml_tensor * ggml_dup_tensor(ساخت ggml_context * ctx, ثابت ساخت ggml_tensor * src) {
    واپس ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

خالی ggml_unravel_اشاریہ(ثابت ساخت ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ثابت int64_t ne2 = tensor->ne[2];
    ثابت int64_t ne1 = tensor->ne[1];
    ثابت int64_t ne0 = tensor->ne[0];

    ثابت int64_t i3_ = (i/(ne2*ne1*ne0));
    ثابت int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ثابت int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ثابت int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    اگر (i0) {
        * i0 = i0_;
    }
    اگر (i1) {
        * i1 = i1_;
    }
    اگر (i2) {
        * i2 = i2_;
    }
    اگر (i3) {
        * i3 = i3_;
    }
}

خالی * ggml_get_ڈیٹا(ثابت ساخت ggml_tensor * tensor) {
    واپس tensor->ڈیٹا;
}

اعشاری * ggml_get_ڈیٹا_f32(ثابت ساخت ggml_tensor * tensor) {
    أكد(tensor->type == GGML_TYPE_F32);
    واپس (اعشاری *)(tensor->ڈیٹا);
}

شمار ggml_unary_op ggml_get_unary_op(ثابت ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    واپس (شمار ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

شمار ggml_glu_op ggml_get_glu_op(ثابت ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    واپس (شمار ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ثابت حرف * ggml_get_name(ثابت ساخت ggml_tensor * tensor) {
    واپس tensor->name;
}

ساخت ggml_tensor * ggml_سیٹ_name(ساخت ggml_tensor * tensor, ثابت حرف * name) {
    size_t i;
    برائے (i = 0; i < حجم(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    واپس tensor;
}

ساخت ggml_tensor * ggml_format_name(ساخت ggml_tensor * tensor, ثابت حرف * fmt, ...) {
    va_list دلائل_2;
    va_start(دلائل_2, fmt);
    vsnprintf(tensor->name, حجم(tensor->name), fmt, دلائل_2);
    va_end(دلائل_2);
    واپس tensor;
}

ساخت ggml_tensor * ggml_view_tensor(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * src) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(نتیجہ, "%s (view)", src->name);

    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = src->nb[i];
    }

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_get_پہلا_tensor(ثابت ساخت ggml_context * ctx) {
    ساخت ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    جب (obj != تہی) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            واپس (ساخت ggml_tensor *)(mem_بفر + obj->offs);
        }

        obj = obj->اگلا;
    }

    واپس تہی;
}

ساخت ggml_tensor * ggml_get_اگلا_tensor(ثابت ساخت ggml_context * ctx, ساخت ggml_tensor * tensor) {
    ساخت ggml_object * obj = (ساخت ggml_object *) ((حرف *)tensor - GGML_OBJECT_سائز);
    obj = obj->اگلا;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    جب (obj != تہی) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            واپس (ساخت ggml_tensor *)(mem_بفر + obj->offs);
        }

        obj = obj->اگلا;
    }

    واپس تہی;
}

ساخت ggml_tensor * ggml_get_tensor(ساخت ggml_context * ctx, ثابت حرف * name) {
    ساخت ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    جب (obj != تہی) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ساخت ggml_tensor * cur = (ساخت ggml_tensor *)(mem_بفر + obj->offs);
            اگر (strcmp(cur->name, name) == 0) {
                واپس cur;
            }
        }

        obj = obj->اگلا;
    }

    واپس تہی;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ساکن ساخت ggml_tensor * ggml_dup_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_DUP;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_dup(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_dup_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_dup_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_dup_impl(ctx, a, true);
}

// ggml_add

ساکن ساخت ggml_tensor * ggml_جوڑو_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_جوڑو;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_جوڑو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_جوڑو_impl(ctx, a, b, false);
}

ساخت ggml_tensor * ggml_جوڑو_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_جوڑو_impl(ctx, a, b, true);
}

// ggml_add_cast

ساکن ساخت ggml_tensor * ggml_جوڑو_cast_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        شمار   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_صفیں(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    نتیجہ->op     = GGML_OP_جوڑو;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_جوڑو_cast(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        شمار   ggml_type      type) {
    واپس ggml_جوڑو_cast_impl(ctx, a, b, type);
}

ساخت ggml_tensor * ggml_جوڑو_id(
            ساخت ggml_context * ctx,
            ساخت ggml_tensor  * a,
            ساخت ggml_tensor  * b,
            ساخت ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_جوڑو_ID;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = ids;

    واپس نتیجہ;
}

// ggml_add1

ساکن ساخت ggml_tensor * ggml_add1_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_ADD1;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_add1(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_add1_impl(ctx, a, b, false);
}

ساخت ggml_tensor * ggml_add1_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ساکن ساخت ggml_tensor * ggml_acc_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(انحراف < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, انحراف, inplace ? 1 : 0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_ACC;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_acc(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف) {
    واپس ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, انحراف, false);
}

ساخت ggml_tensor * ggml_acc_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف) {
    واپس ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, انحراف, true);
}

// ggml_sub

ساکن ساخت ggml_tensor * ggml_sub_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SUB;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_sub(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_sub_impl(ctx, a, b, false);
}

ساخت ggml_tensor * ggml_sub_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ساکن ساخت ggml_tensor * ggml_mul_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_MUL;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_mul(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_mul_impl(ctx, a, b, false);
}

ساخت ggml_tensor * ggml_mul_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ساکن ساخت ggml_tensor * ggml_div_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_DIV;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_div(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_div_impl(ctx, a, b, false);
}

ساخت ggml_tensor * ggml_div_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ساکن ساخت ggml_tensor * ggml_sqr_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SQR;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_sqr(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sqr_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_sqr_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ساکن ساخت ggml_tensor * ggml_sqrt_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SQRT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_sqrt(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sqrt_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_sqrt_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ساکن ساخت ggml_tensor * ggml_log_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_LOG;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_log(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_log_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_log_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_log_impl(ctx, a, true);
}

ساخت ggml_tensor * ggml_expm1(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ساخت ggml_tensor * ggml_expm1_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ساخت ggml_tensor * ggml_softplus(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ساخت ggml_tensor * ggml_softplus_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ساکن ساخت ggml_tensor * ggml_sin_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SIN;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_sin(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sin_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_sin_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ساکن ساخت ggml_tensor * ggml_cos_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_COS;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_cos(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_cos_impl(ctx, a, false);
}

ساخت ggml_tensor * ggml_cos_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ساخت ggml_tensor * ggml_مجموع(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, a->type, 1);

    نتیجہ->op     = GGML_OP_مجموع;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_sum_rows

ساخت ggml_tensor * ggml_مجموع_صفیں(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    برائے (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    نتیجہ->op     = GGML_OP_مجموع_صفیں;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_cumsum

ساخت ggml_tensor * ggml_cumsum(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_CUMSUM;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_mean

ساخت ggml_tensor * ggml_اوسط(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_اوسط;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_argmax

ساخت ggml_tensor * ggml_argmax(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_میٹرکس(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    نتیجہ->op     = GGML_OP_ARGMAX;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_count_equal

ساخت ggml_tensor * ggml_گنتی_equal(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    نتیجہ->op     = GGML_OP_گنتی_EQUAL;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_repeat

ساخت ggml_tensor * ggml_repeat(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    نتیجہ->op     = GGML_OP_REPEAT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_repeat_4d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ثابت bool can_repeat = ggml_is_خالی_خانے(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    نتیجہ->op     = GGML_OP_REPEAT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_repeat_back

ساخت ggml_tensor * ggml_repeat_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    نتیجہ->op     = GGML_OP_REPEAT_عقب;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_concat

ساخت ggml_tensor * ggml_concat(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    ساخت ggml_tensor  * b,
    صحیح                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    برائے (صحیح d = 0; d < GGML_MAX_DIMS; ++d) {
        اگر (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            جاری;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, dim);

    نتیجہ->op     = GGML_OP_CONCAT;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_abs

ساخت ggml_tensor * ggml_abs(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ساخت ggml_tensor * ggml_abs_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ساخت ggml_tensor * ggml_sgn(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ساخت ggml_tensor * ggml_sgn_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ساخت ggml_tensor * ggml_neg(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ساخت ggml_tensor * ggml_neg_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ساخت ggml_tensor * ggml_قدم(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_قدم);
}

ساخت ggml_tensor * ggml_قدم_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_قدم);
}

// ggml_tanh

ساخت ggml_tensor * ggml_tanh(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ساخت ggml_tensor * ggml_tanh_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ساخت ggml_tensor * ggml_elu(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ساخت ggml_tensor * ggml_elu_inplace(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ساخت ggml_tensor * ggml_relu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ساخت ggml_tensor * ggml_relu_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ساخت ggml_tensor * ggml_leaky_relu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 negative_ڈھلان,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &negative_ڈھلان, حجم(negative_ڈھلان));

    نتیجہ->op     = GGML_OP_LEAKY_RELU;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_sigmoid

ساخت ggml_tensor * ggml_sigmoid(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ساخت ggml_tensor * ggml_sigmoid_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ساخت ggml_tensor * ggml_gelu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ساخت ggml_tensor * ggml_gelu_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ساخت ggml_tensor * ggml_gelu_erf(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ساخت ggml_tensor * ggml_gelu_erf_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ساخت ggml_tensor * ggml_gelu_تیز(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_GELU_تیز);
}

ساخت ggml_tensor * ggml_gelu_تیز_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_تیز);
}

// ggml_silu

ساخت ggml_tensor * ggml_silu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ساخت ggml_tensor * ggml_silu_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ساخت ggml_tensor * ggml_xielu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری alpha_n,
        اعشاری alpha_p,
        اعشاری beta,
        اعشاری eps) {
    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_سیٹ_op_params_f32(نتیجہ, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_سیٹ_op_params_f32(نتیجہ, 3, beta);
    ggml_سیٹ_op_params_f32(نتیجہ, 4, eps);

    نتیجہ->op     = GGML_OP_UNARY;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_silu_back

ساخت ggml_tensor * ggml_silu_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SILU_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml hardswish

ساخت ggml_tensor * ggml_hardswish(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ساخت ggml_tensor * ggml_hardsigmoid(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ساخت ggml_tensor * ggml_exp(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ساخت ggml_tensor * ggml_exp_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ساکن ساخت ggml_tensor * ggml_glu_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        شمار ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    اگر (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; برائے (صحیح i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, تہی, 0);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) op);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, (int32_t) swapped);

    نتیجہ->op     = GGML_OP_GLU;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_floor

ساخت ggml_tensor * ggml_floor(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ساخت ggml_tensor * ggml_floor_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ساخت ggml_tensor * ggml_ceil(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ساخت ggml_tensor * ggml_ceil_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ساخت ggml_tensor * ggml_باری(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_باری);
}

ساخت ggml_tensor * ggml_باری_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_باری);
}

//ggml_trunc

ساخت ggml_tensor * ggml_trunc(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ساخت ggml_tensor * ggml_trunc_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ساخت ggml_tensor * ggml_glu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_glu_op      op,
        bool                  swapped) {
    واپس ggml_glu_impl(ctx, a, تہی, op, swapped);
}

ساخت ggml_tensor * ggml_glu_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        شمار ggml_glu_op      op) {
    واپس ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ساخت ggml_tensor * ggml_reglu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_REGLU, false);
}

ساخت ggml_tensor * ggml_reglu_swapped(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_REGLU, true);
}

ساخت ggml_tensor * ggml_reglu_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ساخت ggml_tensor * ggml_geglu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU, false);
}

ساخت ggml_tensor * ggml_geglu_swapped(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU, true);
}

ساخت ggml_tensor * ggml_geglu_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ساخت ggml_tensor * ggml_swiglu(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_SWIGLU, false);
}

ساخت ggml_tensor * ggml_swiglu_swapped(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_SWIGLU, true);
}

ساخت ggml_tensor * ggml_swiglu_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ساخت ggml_tensor * ggml_geglu_erf(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU_ERF, false);
}

ساخت ggml_tensor * ggml_geglu_erf_swapped(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU_ERF, true);
}

ساخت ggml_tensor * ggml_geglu_erf_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ساخت ggml_tensor * ggml_geglu_تیز(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU_تیز, false);
}

ساخت ggml_tensor * ggml_geglu_تیز_swapped(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_glu_impl(ctx, a, تہی, GGML_GLU_OP_GEGLU_تیز, true);
}

ساخت ggml_tensor * ggml_geglu_تیز_بانٹو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_تیز, false);
}

ساخت ggml_tensor * ggml_swiglu_oai(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 alpha,
        اعشاری                 حد_بندی) {
    ساخت ggml_tensor * نتیجہ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_سیٹ_op_params_f32(نتیجہ, 2, alpha);
    ggml_سیٹ_op_params_f32(نتیجہ, 3, حد_بندی);

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_swiglu_clamp(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 حد_بندی) {
    ساخت ggml_tensor * نتیجہ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_سیٹ_op_params_f32(نتیجہ, 3, حد_بندی);

    واپس نتیجہ;
}

// ggml_norm

ساکن ساخت ggml_tensor * ggml_norm_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_NORM;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_norm(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_norm_impl(ctx, a, eps, false);
}

ساخت ggml_tensor * ggml_norm_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ساکن ساخت ggml_tensor * ggml_rms_norm_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_RMS_NORM;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_rms_norm(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_rms_norm_impl(ctx, a, eps, false);
}

ساخت ggml_tensor * ggml_rms_norm_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ساخت ggml_tensor * ggml_rms_norm_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 eps) {
    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_RMS_NORM_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_group_norm

ساکن ساخت ggml_tensor * ggml_گروہ_norm_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, n_groups);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, eps);

    نتیجہ->op     = GGML_OP_گروہ_NORM;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_گروہ_norm(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps) {
    واپس ggml_گروہ_norm_impl(ctx, a, n_groups, eps, false);
}

ساخت ggml_tensor * ggml_گروہ_norm_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps) {
    واپس ggml_گروہ_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ساکن ساخت ggml_tensor * ggml_l2_norm_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, eps);

    نتیجہ->op     = GGML_OP_L2_NORM;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_l2_norm(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_l2_norm_impl(ctx, a, eps, false);
}

ساخت ggml_tensor * ggml_l2_norm_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 eps) {
    واپس ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_سیٹ_acc(
        ساخت ggml_tensor * a,
        شمار ggml_prec       prec) {
    انتخاب (a->op) {
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 0, prec_i32);
            }
            توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 3, prec_i32);
            }
            توڑو;
        طےشدہ:
            واپس false;
    };

    واپس true;
}

bool ggml_prec_سیٹ_src(
        ساخت ggml_tensor * a,
        شمار ggml_prec       prec,
        صحیح                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    انتخاب (a->op) {
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
            {
                اگر (idx != 1) {
                    واپس false;
                }

                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 2 + idx, prec_i32);
            }
            توڑو;
        طےشدہ:
            واپس false;
    };

    واپس true;
}

// ggml_mul_mat

ساکن اندرونی bool ggml_can_mul_mat(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ساخت ggml_tensor * ggml_mul_mat(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ثابت int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_MUL_MAT;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

خالی ggml_mul_mat_سیٹ_prec(
        ساخت ggml_tensor * a,
        شمار ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_سیٹ_op_params_i32(a, 0, prec_i32);
}

خالی ggml_mul_mat_سیٹ_hint(
        ساخت ggml_tensor * a,
        شمار ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t hint_i32 = (int32_t) hint;

    ggml_سیٹ_op_params_i32(a, 1, hint_i32);
}

// ggml_mul_mat_id

/*
    c = ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    b   -> [cols, n_expert_used, n_tokens]
    ids -> [n_expert_used, n_tokens] (i32)
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_expert_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
ساخت ggml_tensor * ggml_mul_mat_id(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * as,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ثابت int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_MUL_MAT_ID;
    نتیجہ->src[0] = as;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = ids;

    واپس نتیجہ;
}

// ggml_out_prod

ساکن اندرونی bool ggml_can_برآمد_prod(ثابت ساخت ggml_tensor * t0, ثابت ساخت ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ساخت ggml_tensor * ggml_برآمد_prod(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_برآمد_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ثابت int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_برآمد_PROD;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_scale

ساکن ساخت ggml_tensor * ggml_scale_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[2] = { s, b };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_SCALE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_scale(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 s) {
    واپس ggml_scale_impl(ctx, a, s, 0.0, false);
}

ساخت ggml_tensor * ggml_scale_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 s) {
    واپس ggml_scale_impl(ctx, a, s, 0.0, true);
}

ساخت ggml_tensor * ggml_scale_bias(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b) {
    واپس ggml_scale_impl(ctx, a, s, b, false);
}

ساخت ggml_tensor * ggml_scale_bias_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b) {
    واپس ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ساکن ساخت ggml_tensor * ggml_سیٹ_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(انحراف < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, انحراف, inplace ? 1 : 0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_سیٹ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_سیٹ(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, nb1, nb2, nb3, انحراف, false);
}

ساخت ggml_tensor * ggml_سیٹ_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, nb1, nb2, nb3, انحراف, true);
}

ساخت ggml_tensor * ggml_سیٹ_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], انحراف, false);
}

ساخت ggml_tensor * ggml_سیٹ_1d_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], انحراف, true);
}

ساخت ggml_tensor * ggml_سیٹ_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], انحراف, false);
}

ساخت ggml_tensor * ggml_سیٹ_2d_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        size_t                nb1,
        size_t                انحراف) {
    واپس ggml_سیٹ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], انحراف, true);
}

// ggml_cpy

ساکن ساخت ggml_tensor * ggml_cpy_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, b);
    اگر (لڑی_لمبائی(b->name) > 0) {
        ggml_format_name(نتیجہ, "%s (copy of %s)", b->name, a->name);
    } ورنہ {
        ggml_format_name(نتیجہ, "%s (copy)", a->name);
    }

    نتیجہ->op     = GGML_OP_CPY;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_cpy(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor * a,
        ساخت ggml_tensor * b) {
    واپس ggml_cpy_impl(ctx, a, b);
}

ساخت ggml_tensor * ggml_cast(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار   ggml_type      type) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(نتیجہ, "%s (copy)", a->name);

    نتیجہ->op     = GGML_OP_CPY;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = نتیجہ; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    واپس نتیجہ;
}

// ggml_cont

ساکن ساخت ggml_tensor * ggml_cont_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);
    ggml_format_name(نتیجہ, "%s (cont)", a->name);

    نتیجہ->op     = GGML_OP_CONT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_cont(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor * a) {
    واپس ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ساخت ggml_tensor * ggml_cont_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0) {
    واپس ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ساخت ggml_tensor * ggml_cont_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    واپس ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ساخت ggml_tensor * ggml_cont_3d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    واپس ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ساخت ggml_tensor * ggml_cont_4d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(نتیجہ, "%s (cont)", a->name);

    نتیجہ->op     = GGML_OP_CONT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_reshape

ساخت ggml_tensor * ggml_reshape(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor * a,
        ساخت ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_reshape_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ثابت int64_t ne[1] = { ne0 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_reshape_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ثابت int64_t ne[2] = { ne0, ne1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_reshape_3d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_reshape_4d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساکن ساخت ggml_tensor * ggml_view_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_dims,
        ثابت int64_t       * ne,
        size_t                انحراف) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, انحراف);
    ggml_format_name(نتیجہ, "%s (view)", a->name);

    ggml_سیٹ_op_params(نتیجہ, &انحراف, حجم(انحراف));

    نتیجہ->op     = GGML_OP_VIEW;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_view_1d

ساخت ggml_tensor * ggml_view_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        size_t                انحراف) {
    ساخت ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 1, &ne0, انحراف);

    واپس نتیجہ;
}

// ggml_view_2d

ساخت ggml_tensor * ggml_view_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                انحراف) {
    ثابت int64_t ne[2] = { ne0, ne1 };

    ساخت ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 2, ne, انحراف);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = نتیجہ->nb[1]*ne1;
    نتیجہ->nb[3] = نتیجہ->nb[2];

    واپس نتیجہ;
}

// ggml_view_3d

ساخت ggml_tensor * ggml_view_3d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                انحراف) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };

    ساخت ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 3, ne, انحراف);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = nb2;
    نتیجہ->nb[3] = نتیجہ->nb[2]*ne2;

    واپس نتیجہ;
}

// ggml_view_4d

ساخت ggml_tensor * ggml_view_4d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                انحراف) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ساخت ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 4, ne, انحراف);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = nb2;
    نتیجہ->nb[3] = nb3;

    واپس نتیجہ;
}

// ggml_permute

ساخت ggml_tensor * ggml_ترتیب_بدلو(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   axis0,
        صحیح                   axis1,
        صحیح                   axis2,
        صحیح                   axis3) {
    GGML_ASSERT(axis0 >= 0 && axis0 < GGML_MAX_DIMS);
    GGML_ASSERT(axis1 >= 0 && axis1 < GGML_MAX_DIMS);
    GGML_ASSERT(axis2 >= 0 && axis2 < GGML_MAX_DIMS);
    GGML_ASSERT(axis3 >= 0 && axis3 < GGML_MAX_DIMS);

    GGML_ASSERT(axis0 != axis1);
    GGML_ASSERT(axis0 != axis2);
    GGML_ASSERT(axis0 != axis3);
    GGML_ASSERT(axis1 != axis2);
    GGML_ASSERT(axis1 != axis3);
    GGML_ASSERT(axis2 != axis3);

    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);
    ggml_format_name(نتیجہ, "%s (permuted)", a->name);

    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    نتیجہ->ne[0] = ne[0];
    نتیجہ->ne[1] = ne[1];
    نتیجہ->ne[2] = ne[2];
    نتیجہ->ne[3] = ne[3];

    نتیجہ->nb[0] = nb[0];
    نتیجہ->nb[1] = nb[1];
    نتیجہ->nb[2] = nb[2];
    نتیجہ->nb[3] = nb[3];

    نتیجہ->op     = GGML_OP_ترتیب_بدلو;
    نتیجہ->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    واپس نتیجہ;
}

// ggml_transpose

ساخت ggml_tensor * ggml_transpose(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);
    ggml_format_name(نتیجہ, "%s (transposed)", a->name);

    نتیجہ->ne[0] = a->ne[1];
    نتیجہ->ne[1] = a->ne[0];

    نتیجہ->nb[0] = a->nb[1];
    نتیجہ->nb[1] = a->nb[0];

    نتیجہ->op     = GGML_OP_TRANSPOSE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_get_rows

ساخت ggml_tensor * ggml_get_صفیں(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    شمار ggml_type type = GGML_TYPE_F32;
    اگر (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    نتیجہ->op     = GGML_OP_GET_صفیں;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_get_rows_back

ساخت ggml_tensor * ggml_get_صفیں_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_میٹرکس(a) && ggml_is_سمتیہ(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_میٹرکس(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    نتیجہ->op     = GGML_OP_GET_صفیں_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_set_rows

ساخت ggml_tensor * ggml_سیٹ_صفیں(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_صفیں(a));
    GGML_ASSERT(ggml_is_contiguous_صفیں(b));

    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_سیٹ_صفیں;
    نتیجہ->src[0] = b;
    نتیجہ->src[1] = c;
    نتیجہ->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    واپس نتیجہ;
}

// ggml_diag

ساخت ggml_tensor * ggml_diag(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ثابت int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, 4, ne);

    نتیجہ->op     = GGML_OP_DIAG;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_diag_mask_inf

ساکن ساخت ggml_tensor * ggml_diag_نقاب_inf_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_DIAG_نقاب_INF;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_diag_نقاب_inf(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past) {
    واپس ggml_diag_نقاب_inf_impl(ctx, a, n_past, false);
}

ساخت ggml_tensor * ggml_diag_نقاب_inf_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past) {
    واپس ggml_diag_نقاب_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ساکن ساخت ggml_tensor * ggml_diag_نقاب_zero_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_DIAG_نقاب_ZERO;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_diag_نقاب_zero(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past) {
    واپس ggml_diag_نقاب_zero_impl(ctx, a, n_past, false);
}

ساخت ggml_tensor * ggml_diag_نقاب_zero_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   n_past) {
    واپس ggml_diag_نقاب_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ساکن ساخت ggml_tensor * ggml_clamp_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        اعشاری                 min,
        اعشاری                 max,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[] = { min, max };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CLAMP;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_clamp(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    اعشاری                 min,
    اعشاری                 max) {
    واپس ggml_clamp_impl(ctx, a, min, max, false);
}

ساخت ggml_tensor * ggml_clamp_inplace(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    اعشاری                 min,
    اعشاری                 max) {
    واپس ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ساکن ساخت ggml_tensor * ggml_soft_max_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * نقاب,
        اعشاری                 scale,
        اعشاری                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    اگر (نقاب) {
        GGML_ASSERT(نقاب->type == GGML_TYPE_F16 || نقاب->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(نقاب));
        GGML_ASSERT(نقاب->ne[0] == a->ne[0]);
        GGML_ASSERT(نقاب->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%نقاب->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%نقاب->ne[3] == 0);
    }

    اگر (max_bias > 0.0f) {
        GGML_ASSERT(نقاب);
    }

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[] = { scale, max_bias };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_SOFT_MAX;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = نقاب;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_soft_max(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_soft_max_impl(ctx, a, تہی, 1.0f, 0.0f, false);
}

ساخت ggml_tensor * ggml_soft_max_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a) {
    واپس ggml_soft_max_impl(ctx, a, تہی, 1.0f, 0.0f, true);
}

ساخت ggml_tensor * ggml_soft_max_ext(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * نقاب,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    واپس ggml_soft_max_impl(ctx, a, نقاب, scale, max_bias, false);
}

ساخت ggml_tensor * ggml_soft_max_ext_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * نقاب,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    واپس ggml_soft_max_impl(ctx, a, نقاب, scale, max_bias, true);
}

خالی ggml_soft_max_جوڑو_sinks(
        ساخت ggml_tensor * a,
        ساخت ggml_tensor * sinks) {
    اگر (!sinks) {
        a->src[2] = تہی;
        واپس;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == تہی);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ساکن ساخت ggml_tensor * ggml_soft_max_ext_عقب_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias,
        bool                  inplace) {
    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SOFT_MAX_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    حافظہ_نقل((اعشاری *) نتیجہ->op_params + 0, &scale,    حجم(اعشاری));
    حافظہ_نقل((اعشاری *) نتیجہ->op_params + 1, &max_bias, حجم(اعشاری));

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_soft_max_ext_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    واپس ggml_soft_max_ext_عقب_impl(ctx, a, b, scale, max_bias, false);
}

ساخت ggml_tensor * ggml_soft_max_ext_عقب_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    واپس ggml_soft_max_ext_عقب_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ساکن ساخت ggml_tensor * ggml_rope_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((منوال & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_سمتیہ(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_مستعمل = منوال & GGML_ROPE_TYPE_MROPE;
    اگر (mrope_مستعمل) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } ورنہ {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    اگر (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, منوال, /*n_ctx*/ 0, n_ctx_orig };
    حافظہ_نقل(params +  5, &freq_بنیاد,    حجم(اعشاری));
    حافظہ_نقل(params +  6, &freq_scale,   حجم(اعشاری));
    حافظہ_نقل(params +  7, &ext_ضربی_عدد,   حجم(اعشاری));
    حافظہ_نقل(params +  8, &attn_ضربی_عدد,  حجم(اعشاری));
    حافظہ_نقل(params +  9, &beta_fast,    حجم(اعشاری));
    حافظہ_نقل(params + 10, &beta_slow,    حجم(اعشاری));
    اگر (mrope_مستعمل && sections) {
        حافظہ_نقل(params + 11, sections,  حجم(int32_t) * GGML_MROPE_SECTIONS);
    } ورنہ {
        memset(params + 11, 0,         حجم(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_ROPE;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_rope(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   منوال) {
    واپس ggml_rope_impl(
        ctx, a, b, تہی, n_dims, تہی, منوال, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ساخت ggml_tensor * ggml_rope_multi(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, false
    );
}

ساخت ggml_tensor * ggml_rope_multi_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, true
    );
}

ساخت ggml_tensor * ggml_rope_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   منوال) {
    واپس ggml_rope_impl(
        ctx, a, b, تہی, n_dims, تہی, منوال, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ساخت ggml_tensor * ggml_rope_ext(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, c, n_dims, تہی, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, false
    );
}

ساخت ggml_tensor * ggml_rope_ext_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, c, n_dims, تہی, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, true
    );
}

ساخت ggml_tensor * ggml_rope_custom(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, تہی, n_dims, تہی, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, false
    );
}

ساخت ggml_tensor * ggml_rope_custom_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    واپس ggml_rope_impl(
        ctx, a, b, تہی, n_dims, تہی, منوال, n_ctx_orig, freq_بنیاد, freq_scale,
        ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ساکن اعشاری ggml_rope_yarn_corr_dim(صحیح n_dims, صحیح n_ctx_orig, اعشاری n_rot, اعشاری بنیاد) {
    واپس n_dims * logf(n_ctx_orig / (n_rot * 2 * (اعشاری)M_PI)) / (2 * logf(بنیاد));
}

خالی ggml_rope_yarn_corr_dims(
    صحیح n_dims, صحیح n_ctx_orig, اعشاری freq_بنیاد, اعشاری beta_fast, اعشاری beta_slow, اعشاری dims[2]
) {
    // start and end correction dims
    اعشاری آغاز = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_بنیاد));
    اعشاری انتہا   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_بنیاد));
    dims[0] = MAX(0, آغاز);
    dims[1] = MIN(n_dims - 1, انتہا);
}

// ggml_rope_back

ساخت ggml_tensor * ggml_rope_ext_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    ساخت ggml_tensor * نتیجہ = ggml_rope_ext(
        ctx, a, b, c, n_dims, منوال, n_ctx_orig, freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow);
    نتیجہ->op = GGML_OP_ROPE_عقب;
    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_rope_multi_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[4],
        صحیح                   منوال,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_بنیاد,
        اعشاری                 freq_scale,
        اعشاری                 ext_ضربی_عدد,
        اعشاری                 attn_ضربی_عدد,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    ساخت ggml_tensor * نتیجہ = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow);
    نتیجہ->op = GGML_OP_ROPE_عقب;
    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_rope_سیٹ_انحراف(
        ساخت ggml_tensor  * a,
        صحیح                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_عقب);
    GGML_ASSERT(n_offs >= 0);

    ثابت int32_t منوال = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(منوال != GGML_ROPE_TYPE_VISION);

    ggml_سیٹ_op_params_i32(a, 15, n_offs);
    واپس a;
}

ساکن int64_t ggml_calc_conv_output_سائز(int64_t ins, int64_t ks, صحیح s, صحیح p, صحیح d) {
    واپس (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ساخت ggml_tensor * ggml_im2col(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1,
        bool                  is_2D,
        شمار ggml_type        dst_type) {
    اگر (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } ورنہ {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ثابت int64_t OH = is_2D ? ggml_calc_conv_output_سائز(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ثابت int64_t OW =         ggml_calc_conv_output_سائز(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ثابت int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_ستون;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_im2col_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        int64_t             * ne,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1,
        bool                  is_2D) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_ستون_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_conv_1d

ساخت ggml_tensor * ggml_conv_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    ساخت ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ساخت ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    نتیجہ = ggml_reshape_3d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 2, 1, 3));                // [N, OC, OL]

    واپس نتیجہ;
}

// ggml_conv_1d_ph

ساخت ggml_tensor* ggml_conv_1d_ph(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s,
        صحیح                   d) {
    واپس ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ساخت ggml_tensor * ggml_conv_1d_dw(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    ساخت ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ساخت ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ساخت ggml_tensor * نتیجہ = ggml_mul_mat(ctx, im2col, a);

    نتیجہ = ggml_reshape_3d(ctx, نتیجہ, نتیجہ->ne[0], نتیجہ->ne[2], 1);

    واپس نتیجہ;
}

// ggml_conv_1d_dw_ph

ساخت ggml_tensor * ggml_conv_1d_dw_ph(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   d0) {
    واپس ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ساخت ggml_tensor * ggml_col2im_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   s0,
        صحیح                   oc,
        صحیح                   p0) {
    GGML_ASSERT(ggml_is_میٹرکس(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ثابت int64_t K_OC = a->ne[0];
    ثابت int64_t T_اندر = a->ne[1];
    ثابت int64_t K = K_OC / oc;
    ثابت int64_t T_برآمد = (T_اندر - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_برآمد > 0);

    ثابت int64_t ne[4] = { T_برآمد, oc, 1, 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_COL2IM_1D;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_conv_transpose_1d

ساکن int64_t ggml_calc_conv_transpose_1d_output_سائز(int64_t ins, int64_t ks, صحیح s, صحیح p, صحیح d) {
    واپس (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ساخت ggml_tensor * ggml_conv_transpose_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    GGML_ASSERT(ggml_is_میٹرکس(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_سائز(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CONV_TRANSPOSE_1D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ساخت ggml_tensor * ggml_conv_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1) {
    ساخت ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ساخت ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 1, 3, 2)); // [N, OC, OH, OW]


    واپس نتیجہ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ساخت ggml_tensor * ggml_im2col_3d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        int64_t               IC,
        صحیح                   s0, // stride width
        صحیح                   s1, // stride height
        صحیح                   s2, // stride depth
        صحیح                   p0, // padding width
        صحیح                   p1, // padding height
        صحیح                   p2, // padding depth
        صحیح                   d0, // dilation width
        صحیح                   d1, // dilation height
        صحیح                   d2, // dilation depth
        شمار ggml_type        dst_type) {
    ثابت int64_t N = b->ne[3] / IC;
    ثابت int64_t ID = b->ne[2];
    ثابت int64_t IH = b->ne[1];
    ثابت int64_t IW = b->ne[0];

    ثابت int64_t OC = a->ne[3] / IC;
    غیر_مستعمل(OC);
    ثابت int64_t KD = a->ne[2];
    ثابت int64_t KH = a->ne[1];
    ثابت int64_t KW = a->ne[0];
    ثابت int64_t OD = ggml_calc_conv_output_سائز(ID, KD, s2, p2, d2);
    ثابت int64_t OH = ggml_calc_conv_output_سائز(IH, KH, s1, p1, d1);
    ثابت int64_t OW = ggml_calc_conv_output_سائز(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ثابت int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_ستون_3_D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ساخت ggml_tensor * ggml_conv_3d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        int64_t               IC,
        صحیح                   s0, // stride width
        صحیح                   s1, // stride height
        صحیح                   s2, // stride depth
        صحیح                   p0, // padding width
        صحیح                   p1, // padding height
        صحیح                   p2, // padding depth
        صحیح                   d0, // dilation width
        صحیح                   d1, // dilation height
        صحیح                   d2  // dilation depth
        ) {
    ساخت ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ساخت ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    واپس نتیجہ;
}

// ggml_conv_2d_sk_p0

ساخت ggml_tensor * ggml_conv_2d_sk_p0(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ساخت ggml_tensor * ggml_conv_2d_s1_ph(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    واپس ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ساخت ggml_tensor * ggml_conv_2d_dw(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1) {
    ساخت ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ساخت ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ساخت ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ساخت ggml_tensor * نتیجہ = ggml_mul_mat(ctx, new_a, new_b);
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    واپس نتیجہ;
}

// ggml_conv_2d_dw_direct

ساخت ggml_tensor * ggml_conv_2d_dw_براہ_راست(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   stride0,
        صحیح                   stride1,
        صحیح                   pad0,
        صحیح                   pad1,
        صحیح                   dilation0,
        صحیح                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_سائز(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_سائز(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, b->type, 4, ne);

    اگر (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ثابت int64_t type_سائز = ggml_type_سائز(نتیجہ->type);
        GGML_ASSERT(ggml_blck_سائز(نتیجہ->type) == 1);
        نتیجہ->nb[0] = نتیجہ->ne[2] * type_سائز;
        نتیجہ->nb[1] = نتیجہ->ne[0] * نتیجہ->nb[0];
        نتیجہ->nb[2] = type_سائز;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CONV_2D_DW;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    واپس نتیجہ;
}

// ggml_conv_2d_direct

ساخت ggml_tensor * ggml_conv_2d_براہ_راست(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ساخت ggml_tensor  * b,   // input data [W, H, C, N]
        صحیح                   s0,  // stride dimension 0
        صحیح                   s1,  // stride dimension 1
        صحیح                   p0,  // padding dimension 0
        صحیح                   p1,  // padding dimension 1
        صحیح                   d0,  // dilation dimension 0
        صحیح                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_سائز(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_سائز(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, s0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, s1);
    ggml_سیٹ_op_params_i32(نتیجہ, 2, p0);
    ggml_سیٹ_op_params_i32(نتیجہ, 3, p1);
    ggml_سیٹ_op_params_i32(نتیجہ, 4, d0);
    ggml_سیٹ_op_params_i32(نتیجہ, 5, d1);

    نتیجہ->op = GGML_OP_CONV_2D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_conv_3d_direct

ساخت ggml_tensor * ggml_conv_3d_براہ_راست(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   s2,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   d0,
        صحیح                   d1,
        صحیح                   d2,
        صحیح                   c,
        صحیح                   n,
        صحیح                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_سائز(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_سائز(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_سائز(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0,  s0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1,  s1);
    ggml_سیٹ_op_params_i32(نتیجہ, 2,  s2);
    ggml_سیٹ_op_params_i32(نتیجہ, 3,  p0);
    ggml_سیٹ_op_params_i32(نتیجہ, 4,  p1);
    ggml_سیٹ_op_params_i32(نتیجہ, 5,  p2);
    ggml_سیٹ_op_params_i32(نتیجہ, 6,  d0);
    ggml_سیٹ_op_params_i32(نتیجہ, 7,  d1);
    ggml_سیٹ_op_params_i32(نتیجہ, 8,  d2);
    ggml_سیٹ_op_params_i32(نتیجہ, 9,  c);
    ggml_سیٹ_op_params_i32(نتیجہ, 10, n);
    ggml_سیٹ_op_params_i32(نتیجہ, 11, oc);

    نتیجہ->op = GGML_OP_CONV_3D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_conv_transpose_2d_p0

ساکن int64_t ggml_calc_conv_transpose_output_سائز(int64_t ins, int64_t ks, صحیح s, صحیح p) {
    واپس (ins - 1) * s - 2 * p + ks;
}

ساخت ggml_tensor * ggml_conv_transpose_2d_p0(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        صحیح                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_output_سائز(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_سائز(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ساخت ggml_tensor* نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, stride);

    نتیجہ->op     = GGML_OP_CONV_TRANSPOSE_2D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_pool_*

ساکن int64_t ggml_calc_pool_output_سائز(int64_t ins, صحیح ks, صحیح s, اعشاری p) {
    واپس (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ساخت ggml_tensor * ggml_pool_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   s0,
        صحیح                   p0) {
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_سائز(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_POOL_1D;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_pool_2d

ساخت ggml_tensor * ggml_pool_2d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   k1,
        صحیح                   s0,
        صحیح                   s1,
        اعشاری                 p0,
        اعشاری                 p1) {
    ساخت ggml_tensor * نتیجہ;
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_سائز(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_سائز(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_POOL_2D;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_pool_2d_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * af,
        شمار ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   k1,
        صحیح                   s0,
        صحیح                   s1,
        اعشاری                 p0,
        اعشاری                 p1) {
    ساخت ggml_tensor * نتیجہ;
    نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_POOL_2_D_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = af;

    واپس نتیجہ;
}

// ggml_upscale / ggml_interpolate

ساکن ساخت ggml_tensor * ggml_interpolate_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              منوال) {
    GGML_ASSERT((منوال & 0xFF) < GGML_SCALE_منوال_گنتی);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(منوال & GGML_SCALE_FLAG_ANTIALIAS) || (منوال & 0xFF) == GGML_SCALE_منوال_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t)منوال);

    نتیجہ->op     = GGML_OP_UPSCALE;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_upscale(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   scale_ضربی_عدد,
        شمار ggml_scale_منوال  منوال) {
    GGML_ASSERT(scale_ضربی_عدد > 1);
    واپس ggml_interpolate_impl(ctx, a, a->ne[0] * scale_ضربی_عدد, a->ne[1] * scale_ضربی_عدد, a->ne[2], a->ne[3], منوال);
}

ساخت ggml_tensor * ggml_upscale_ext(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   ne0,
        صحیح                   ne1,
        صحیح                   ne2,
        صحیح                   ne3,
        شمار ggml_scale_منوال  منوال) {
    واپس ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, منوال);
}

ساخت ggml_tensor * ggml_interpolate(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              منوال) {
    واپس ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, منوال);
}

// ggml_pad

ساخت ggml_tensor * ggml_pad(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   p3) {
    واپس ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ساخت ggml_tensor * ggml_pad_circular(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   p3) {
    واپس ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ساخت ggml_tensor * ggml_pad_ext(
            ساخت ggml_context * ctx,
            ساخت ggml_tensor  * a,
            صحیح                  lp0,
            صحیح                  rp0,
            صحیح                  lp1,
            صحیح                  rp1,
            صحیح                  lp2,
            صحیح                  rp2,
            صحیح                  lp3,
            صحیح                  rp3
            ) {
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, lp0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, rp0);
    ggml_سیٹ_op_params_i32(نتیجہ, 2, lp1);
    ggml_سیٹ_op_params_i32(نتیجہ, 3, rp1);
    ggml_سیٹ_op_params_i32(نتیجہ, 4, lp2);
    ggml_سیٹ_op_params_i32(نتیجہ, 5, rp2);
    ggml_سیٹ_op_params_i32(نتیجہ, 6, lp3);
    ggml_سیٹ_op_params_i32(نتیجہ, 7, rp3);
    ggml_سیٹ_op_params_i32(نتیجہ, 8, 0); // not circular by default


    نتیجہ->op     = GGML_OP_PAD;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_pad_ext_circular

ساخت ggml_tensor * ggml_pad_ext_circular(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                  lp0,
        صحیح                  rp0,
        صحیح                  lp1,
        صحیح                  rp1,
        صحیح                  lp2,
        صحیح                  rp2,
        صحیح                  lp3,
        صحیح                  rp3
        ) {
    ساخت ggml_tensor * نتیجہ = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_سیٹ_op_params_i32(نتیجہ, 8, 1); // circular
    واپس نتیجہ;
}

// ggml_pad_reflect_1d

ساخت ggml_tensor * ggml_pad_reflect_1d(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_PAD_REFLECT_1D;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_roll

ساخت ggml_tensor * ggml_roll(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   shift0,
        صحیح                   shift1,
        صحیح                   shift2,
        صحیح                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_سائز(a->type));
    GGML_ASSERT(مطلق_قدر(shift0) < a->ne[0]);
    GGML_ASSERT(مطلق_قدر(shift1) < a->ne[1]);
    GGML_ASSERT(مطلق_قدر(shift2) < a->ne[2]);
    GGML_ASSERT(مطلق_قدر(shift3) < a->ne[3]);

    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, shift0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, shift1);
    ggml_سیٹ_op_params_i32(نتیجہ, 2, shift2);
    ggml_سیٹ_op_params_i32(نتیجہ, 3, shift3);

    نتیجہ->op     = GGML_OP_ROLL;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_timestep_embedding

ساخت ggml_tensor * ggml_timestep_embedding(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * timesteps,
        صحیح                   dim,
        صحیح                   max_period) {

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, dim);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, max_period);

    نتیجہ->op     = GGML_OP_TIMESTEP_EMBEDDING;
    نتیجہ->src[0] = timesteps;

    واپس نتیجہ;
}

// ggml_tri

ساخت ggml_tensor * ggml_tri(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    شمار ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, type);

    نتیجہ->op = GGML_OP_TRI;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_fill

ساکن ساخت ggml_tensor * ggml_fill_impl(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    اعشاری                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, c);

    نتیجہ->op = GGML_OP_FILL;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_fill(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    اعشاری                 c) {
    واپس ggml_fill_impl(ctx, a, c, false);
}

ساخت ggml_tensor * ggml_fill_inplace(
    ساخت ggml_context * ctx,
    ساخت ggml_tensor  * a,
    اعشاری                 c) {
    واپس ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ساخت ggml_tensor * ggml_argsort(
        ساخت ggml_context  * ctx,
        ساخت ggml_tensor   * a,
        شمار ggml_ترتیب_دو_ترتیب   ترتیب) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) ترتیب);

    نتیجہ->op     = GGML_OP_ARGSORT;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_argsort_top_k

ساخت ggml_tensor * ggml_argsort_چوٹی_k(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ساخت ggml_tensor * نتیجہ = ggml_argsort(ctx, a, GGML_ترتیب_دو_ترتیب_DESC);

    نتیجہ = ggml_view_4d(ctx, نتیجہ,
                k, نتیجہ->ne[1], نتیجہ->ne[2], نتیجہ->ne[3],
                   نتیجہ->nb[1], نتیجہ->nb[2], نتیجہ->nb[3],
                0);

    واپس نتیجہ;
}

// ggml_top_k

ساخت ggml_tensor * ggml_چوٹی_k(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    نتیجہ->op     = GGML_OP_چوٹی_K;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_arange

ساخت ggml_tensor * ggml_arange(
        ساخت ggml_context * ctx,
        اعشاری                 آغاز,
        اعشاری                 stop,
        اعشاری                 قدم) {
    GGML_ASSERT(stop > آغاز);

    ثابت int64_t اقدام = (int64_t) ceilf((stop - آغاز) / قدم);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, اقدام);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, آغاز);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, stop);
    ggml_سیٹ_op_params_f32(نتیجہ, 2, قدم);

    نتیجہ->op = GGML_OP_ARANGE;

    واپس نتیجہ;
}

// ggml_flash_attn_ext

ساخت ggml_tensor * ggml_flash_attn_ext(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * q,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * نقاب,
        اعشاری                 scale,
        اعشاری                 max_bias,
        اعشاری                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    اگر (نقاب) {
        GGML_ASSERT(نقاب->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(نقاب));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % نقاب->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % نقاب->ne[3] == 0);
    }

    اگر (max_bias > 0.0f) {
        GGML_ASSERT(نقاب);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    اعشاری params[] = { scale, max_bias, logit_softcap };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_FLASH_ATTN_EXT;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = نقاب;

    واپس نتیجہ;
}


خالی ggml_flash_attn_ext_سیٹ_prec(
        ساخت ggml_tensor * a,
        شمار ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_سیٹ_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

شمار ggml_prec ggml_flash_attn_ext_get_prec(
        ثابت ساخت ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    واپس (شمار ggml_prec) prec_i32;
}

خالی ggml_flash_attn_ext_سیٹ_n_kv_max(
        ساخت ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_سیٹ_op_params_i32(a, 4, n_kv_max);
}

خالی ggml_flash_attn_ext_جوڑو_sinks(
        ساخت ggml_tensor * a,
        ساخت ggml_tensor * sinks) {
    اگر (!sinks) {
        a->src[4] = تہی;
        واپس;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == تہی);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ساخت ggml_tensor * ggml_flash_attn_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * q,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ثابت int64_t     D = q->ne[0];
    ثابت int64_t     N = q->ne[1];
    ثابت int64_t     M = k->ne[1];
    ثابت int64_t   ne2 = q->ne[2];
    ثابت int64_t   ne3 = q->ne[3];
    ثابت int64_t kvne2 = k->ne[2];

    GGML_ASSERT(k->ne[0] == D);
    GGML_ASSERT(v->ne[0] == M);
    GGML_ASSERT(v->ne[1] == D);
    GGML_ASSERT(d->ne[0] == D);
    GGML_ASSERT(d->ne[1] == N);
    GGML_ASSERT(k->ne[2] == kvne2);
    GGML_ASSERT(k->ne[3] == ne3);
    GGML_ASSERT(v->ne[2] == kvne2);
    GGML_ASSERT(v->ne[3] == ne3);
    GGML_ASSERT(d->ne[2] == ne2);
    GGML_ASSERT(d->ne[3] == ne3);

    GGML_ASSERT(ne2 % kvne2 == 0);

    // store gradients of q, k and v as continuous tensors concatenated in result.
    // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
    ثابت int64_t elem_q = ggml_nelements(q);
    ثابت int64_t elem_k = ggml_nelements(k);
    ثابت int64_t elem_v = ggml_nelements(v);

    شمار ggml_type نتیجہ_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_سائز(نتیجہ_type) == 1);
    ثابت size_t tsize = ggml_type_سائز(نتیجہ_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ثابت size_t انتہا    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ثابت size_t nelements = (انتہا + tsize - 1)/tsize;

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_سیٹ_op_params(نتیجہ, &masked_i, حجم(masked_i));

    نتیجہ->op     = GGML_OP_FLASH_ATTN_عقب;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = d;

    واپس نتیجہ;
}

// ggml_ssm_conv

ساخت ggml_tensor * ggml_ssm_conv(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * sx,
        ساخت ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_میٹرکس(c));

    ثابت int64_t d_conv  = c->ne[0];
    ثابت int64_t d_inner = c->ne[1];
    ثابت int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ثابت int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    نتیجہ->op     = GGML_OP_SSM_CONV;
    نتیجہ->src[0] = sx;
    نتیجہ->src[1] = c;

    واپس نتیجہ;
}

// ggml_ssm_scan

ساخت ggml_tensor * ggml_ssm_جائزہ(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * s,
        ساخت ggml_tensor  * x,
        ساخت ggml_tensor  * dt,
        ساخت ggml_tensor  * A,
        ساخت ggml_tensor  * B,
        ساخت ggml_tensor  * C,
        ساخت ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_سائز(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_سائز(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_سائز(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ثابت int64_t d_state      = s->ne[0];
        ثابت int64_t سر_dim     = x->ne[0];
        ثابت int64_t n_سر       = x->ne[1];
        ثابت int64_t n_seq_علامات = x->ne[2];
        ثابت int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_سر);
        GGML_ASSERT(dt->ne[1] == n_seq_علامات);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == سر_dim);
        GGML_ASSERT(s->ne[2] == n_سر);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_علامات);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_سمتیہ(ids));
        GGML_ASSERT(A->ne[1] == n_سر);
        GGML_ASSERT(ggml_is_میٹرکس(A));

        اگر (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    نتیجہ->op   = GGML_OP_SSM_جائزہ;
    نتیجہ->src[0] = s;
    نتیجہ->src[1] = x;
    نتیجہ->src[2] = dt;
    نتیجہ->src[3] = A;
    نتیجہ->src[4] = B;
    نتیجہ->src[5] = C;
    نتیجہ->src[6] = ids;

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) K);

    واپس نتیجہ;
}

// ggml_win_part

ساخت ggml_tensor * ggml_win_part(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ثابت صحیح px = (w - a->ne[1]%w)%w;
    ثابت صحیح py = (w - a->ne[2]%w)%w;

    ثابت صحیح npx = (px + a->ne[1])/w;
    ثابت صحیح npy = (py + a->ne[2])/w;
    ثابت صحیح np  = npx*npy;

    ثابت int64_t ne[4] = { a->ne[0], w, w, np, };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_WIN_PART;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_win_unpart

ساخت ggml_tensor * ggml_win_unpart(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   w0,
        صحیح                   h0,
        صحیح                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ثابت int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_WIN_UNPART;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_get_rel_pos

ساخت ggml_tensor * ggml_get_rel_pos(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        صحیح                   qh,
        صحیح                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ثابت int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    نتیجہ->op     = GGML_OP_GET_REL_POS;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

// ggml_add_rel_pos

ساکن ساخت ggml_tensor * ggml_جوڑو_rel_pos_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * pw,
        ساخت ggml_tensor  * ph,
        bool                  inplace) {
    GGML_ASSERT(ggml_are_same_shape(pw, ph));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(pw));
    GGML_ASSERT(ggml_is_contiguous(ph));
    GGML_ASSERT(ph->type == GGML_TYPE_F32);
    GGML_ASSERT(pw->type == GGML_TYPE_F32);
    GGML_ASSERT(pw->ne[3] == a->ne[2]);
    GGML_ASSERT(pw->ne[0]*pw->ne[0] == a->ne[0]);
    GGML_ASSERT(pw->ne[1]*pw->ne[2] == a->ne[1]);

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_سیٹ_op_params_i32(نتیجہ, 0, inplace ? 1 : 0);

    نتیجہ->op     = GGML_OP_جوڑو_REL_POS;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = pw;
    نتیجہ->src[2] = ph;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_جوڑو_rel_pos(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * pw,
        ساخت ggml_tensor  * ph) {
    واپس ggml_جوڑو_rel_pos_impl(ctx, a, pw, ph, false);
}

ساخت ggml_tensor * ggml_جوڑو_rel_pos_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * pw,
        ساخت ggml_tensor  * ph) {
    واپس ggml_جوڑو_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ساخت ggml_tensor * ggml_rwkv_wkv6(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * r,
        ساخت ggml_tensor  * tf,
        ساخت ggml_tensor  * td,
        ساخت ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_علامات = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_علامات);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_علامات);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_علامات);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_علامات + S * n_seqs, 1, 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_RWKV_WKV6;
    نتیجہ->src[0] = k;
    نتیجہ->src[1] = v;
    نتیجہ->src[2] = r;
    نتیجہ->src[3] = tf;
    نتیجہ->src[4] = td;
    نتیجہ->src[5] = state;

    واپس نتیجہ;
}

// ggml_gated_linear_attn

ساخت ggml_tensor * ggml_gated_خطی_attn(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * q,
        ساخت ggml_tensor  * g,
        ساخت ggml_tensor  * state,
        اعشاری scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_علامات = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_علامات);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_علامات);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_علامات);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_علامات + S * n_seqs, 1, 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, scale);

    نتیجہ->op     = GGML_OP_GATED_خطی_ATTN;
    نتیجہ->src[0] = k;
    نتیجہ->src[1] = v;
    نتیجہ->src[2] = q;
    نتیجہ->src[3] = g;
    نتیجہ->src[4] = state;

    واپس نتیجہ;
}

// ggml_rwkv_wkv7

ساخت ggml_tensor * ggml_rwkv_wkv7(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * r,
        ساخت ggml_tensor  * w,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_علامات = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_علامات);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_علامات);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_علامات);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_علامات);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_علامات);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_علامات + S * n_seqs, 1, 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_RWKV_WKV7;
    نتیجہ->src[0] = r;
    نتیجہ->src[1] = w;
    نتیجہ->src[2] = k;
    نتیجہ->src[3] = v;
    نتیجہ->src[4] = a;
    نتیجہ->src[5] = b;
    نتیجہ->src[6] = state;

    واپس نتیجہ;
}

// ggml_unary

ساکن ساخت ggml_tensor * ggml_unary_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_صفیں(a));

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) op);

    نتیجہ->op     = GGML_OP_UNARY;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_unary(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_unary_op    op) {
    واپس ggml_unary_impl(ctx, a, op, false);
}

ساخت ggml_tensor * ggml_unary_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        شمار ggml_unary_op    op) {
    واپس ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ساکن ساخت ggml_tensor * ggml_map_custom1_impl(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساخت ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM1;
    نتیجہ->src[0] = a;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_map_custom1(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ساخت ggml_tensor * ggml_map_custom1_inplace(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ساکن ساخت ggml_tensor * ggml_map_custom2_impl(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساخت ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM2;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_map_custom2(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ساخت ggml_tensor * ggml_map_custom2_inplace(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ساکن ساخت ggml_tensor * ggml_map_custom3_impl(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ساخت ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساخت ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساخت ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM3;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_map_custom3(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ساخت ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ساخت ggml_tensor * ggml_map_custom3_inplace(
        ساخت ggml_context      * ctx,
        ساخت ggml_tensor       * a,
        ساخت ggml_tensor       * b,
        ساخت ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        خالی                     * userdata) {
    واپس ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ساخت ggml_tensor * ggml_custom_4d(
        ساخت ggml_context * ctx,
        شمار ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ساخت ggml_tensor ** دلائل_2,
        صحیح                   n_دلائل,
        ggml_custom_op_t      fun,
        صحیح                   n_tasks,
        خالی                * userdata) {

    GGML_ASSERT(n_دلائل < GGML_MAX_SRC);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ساخت ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op = GGML_OP_CUSTOM;
    برائے (صحیح i = 0; i < n_دلائل; i++) {
        نتیجہ->src[i] = دلائل_2[i];
    }

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_custom_inplace(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor ** دلائل_2,
        صحیح                   n_دلائل,
        ggml_custom_op_t      fun,
        صحیح                   n_tasks,
        خالی                * userdata) {

    GGML_ASSERT(n_دلائل < GGML_MAX_SRC - 1);

    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    ساخت ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op = GGML_OP_CUSTOM;
    نتیجہ->src[0] = a;
    برائے (صحیح i = 0; i < n_دلائل; i++) {
        نتیجہ->src[i + 1] = دلائل_2[i];
    }

    واپس نتیجہ;
}
// ggml_cross_entropy_loss

ساخت ggml_tensor * ggml_cross_entropy_loss(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, a->type, 1);

    نتیجہ->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_cross_entropy_loss_back

ساخت ggml_tensor * ggml_cross_entropy_loss_عقب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        ساخت ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ساخت ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, b);

    نتیجہ->op     = GGML_OP_CROSS_ENTROPY_LOSS_عقب;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    واپس نتیجہ;
}

// opt_step_adamw

ساخت ggml_tensor * ggml_opt_قدم_adamw(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * grad,
        ساخت ggml_tensor  * m,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_OPT_قدم_ADAMW;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = grad;
    نتیجہ->src[2] = m;
    نتیجہ->src[3] = v;
    نتیجہ->src[4] = adamw_params;

    واپس نتیجہ;
}

// opt_step_sgd

ساخت ggml_tensor * ggml_opt_قدم_sgd(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * grad,
        ساخت ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ساخت ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_OPT_قدم_SGD;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = grad;
    نتیجہ->src[2] = params;

    واپس نتیجہ;
}

// solve_tri

ساخت ggml_tensor * ggml_solve_tri(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * a,
        ساخت ggml_tensor  * b,
        bool                  بایاں,
        bool                  زیریں,
        bool                  uni) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    // A must be square and lower diagonal
    GGML_ASSERT(a->ne[0] == a->ne[1]);
    // B must have same outer dimension as A
    GGML_ASSERT(a->ne[1] == b->ne[1]);

    // batch dimensions must be equal
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));

    GGML_ASSERT(زیریں && بایاں && !uni); // TODO: support other variants

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    نتیجہ->op     = GGML_OP_SOLVE_TRI;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    واپس نتیجہ;
}

// ggml_gated_delta_net

ساخت ggml_tensor * ggml_gated_delta_net(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * q,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * v,
        ساخت ggml_tensor  * g,
        ساخت ggml_tensor  * beta,
        ساخت ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_صفیں(q));
    GGML_ASSERT(ggml_is_contiguous_صفیں(k));
    GGML_ASSERT(ggml_is_contiguous_صفیں(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    ثابت int64_t S_v      = v->ne[0];
    ثابت int64_t H        = v->ne[1];
    ثابت int64_t n_علامات = v->ne[2];
    ثابت int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ثابت int64_t state_صفیں = K * S_v * n_seqs;
    ثابت int64_t ne[4] = { S_v * H, n_علامات * n_seqs + state_صفیں, 1, 1 };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) K);

    نتیجہ->op     = GGML_OP_GATED_DELTA_NET;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = g;
    نتیجہ->src[4] = beta;
    نتیجہ->src[5] = state;

    واپس نتیجہ;
}

// ggml_lightning_indexer

ساخت ggml_tensor * ggml_lightning_indexer(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * q,
        ساخت ggml_tensor  * k,
        ساخت ggml_tensor  * weights,
        ساخت ggml_tensor  * نقاب) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    نقاب->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   نقاب->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   نقاب->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   نقاب->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % نقاب->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    ساخت ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op   = GGML_OP_LIGHTNING_INDEXER;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = weights;
    نتیجہ->src[3] = نقاب;

    واپس نتیجہ;
}

// ggml_dsv4_hc_comb

ساخت ggml_tensor * ggml_dsv4_hc_comb(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * mixes,
        ساخت ggml_tensor  * scale,
        ساخت ggml_tensor  * بنیاد,
        اعشاری                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(بنیاد->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ثابت int64_t hc_mix_dim = mixes->ne[0];
    ثابت int64_t n_علامات   = mixes->ne[1];

    int64_t hc = 0;
    برائے (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        اگر ((2 + i)*i == hc_mix_dim) {
            hc = i;
            توڑو;
        }
    }

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(hc == 4);
    GGML_ASSERT(mixes->ne[2] == 1);
    GGML_ASSERT(mixes->ne[3] == 1);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(scale->ne[1] == 1);
    GGML_ASSERT(scale->ne[2] == 1);
    GGML_ASSERT(scale->ne[3] == 1);
    GGML_ASSERT(بنیاد->ne[0] == hc_mix_dim);
    GGML_ASSERT(بنیاد->ne[1] == 1);
    GGML_ASSERT(بنیاد->ne[2] == 1);
    GGML_ASSERT(بنیاد->ne[3] == 1);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_علامات);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, eps);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, n_iter);

    نتیجہ->op     = GGML_OP_DSV4_HC_COMB;
    نتیجہ->src[0] = mixes;
    نتیجہ->src[1] = scale;
    نتیجہ->src[2] = بنیاد;

    واپس نتیجہ;
}

// ggml_dsv4_hc_pre

ساکن ساخت ggml_tensor * ggml_dsv4_hc_pre_impl(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * x,
        ساخت ggml_tensor  * weights,
        اعشاری                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_علامات = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    اگر (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_علامات);
    } ورنہ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_علامات);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_علامات);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, scale);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, gated ? 1 : 0);

    نتیجہ->op     = GGML_OP_DSV4_HC_PRE;
    نتیجہ->src[0] = x;
    نتیجہ->src[1] = weights;

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_dsv4_hc_pre(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * x,
        ساخت ggml_tensor  * weights) {
    واپس ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ساخت ggml_tensor * ggml_dsv4_hc_pre_gated(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * x,
        ساخت ggml_tensor  * gate,
        اعشاری                 scale) {
    واپس ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ساخت ggml_tensor * ggml_dsv4_hc_post(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * x,
        ساخت ggml_tensor  * residual,
        ساخت ggml_tensor  * post,
        ساخت ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t n_علامات = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_علامات);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_علامات);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    اگر (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_علامات);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_علامات);

    نتیجہ->op     = GGML_OP_DSV4_HC_POST;
    نتیجہ->src[0] = x;
    نتیجہ->src[1] = residual;
    نتیجہ->src[2] = post;
    نتیجہ->src[3] = comb;

    واپس نتیجہ;
}

// ggml_top_k_mask

ساخت ggml_tensor * ggml_چوٹی_k_نقاب(
        ساخت ggml_context * ctx,
        ساخت ggml_tensor  * نقاب,
        ساخت ggml_tensor  * چوٹی_k,
        ساخت ggml_tensor  * score,
        ساخت ggml_tensor  * بلاکیں,
        ساخت ggml_tensor  * extra) {
    GGML_ASSERT(نقاب->type == GGML_TYPE_F16 || نقاب->type == GGML_TYPE_F32);
    GGML_ASSERT(نقاب->nb[0] == ggml_type_سائز(نقاب->type));
    GGML_ASSERT(چوٹی_k->type == GGML_TYPE_I32);
    GGML_ASSERT(چوٹی_k->ne[1] % نقاب->ne[1] == 0 && چوٹی_k->ne[2] % نقاب->ne[2] == 0 && چوٹی_k->ne[3] % نقاب->ne[3] == 0);

    اگر (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == چوٹی_k->ne[1] && score->ne[2] == چوٹی_k->ne[2] && score->ne[3] == چوٹی_k->ne[3]);
    }

    اگر (بلاکیں) {
        GGML_ASSERT(بلاکیں->type == GGML_TYPE_I32);
        GGML_ASSERT(بلاکیں->ne[2] == 1 && بلاکیں->ne[3] == 1);
        GGML_ASSERT(!score || بلاکیں->ne[1] == score->ne[0]);
    }

    اگر (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == چوٹی_k->ne[1] && extra->ne[2] == چوٹی_k->ne[2] && extra->ne[3] == چوٹی_k->ne[3]);
    }

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, نقاب->type, نقاب->ne[0], چوٹی_k->ne[1], چوٹی_k->ne[2], چوٹی_k->ne[3]);

    نتیجہ->op     = GGML_OP_چوٹی_K_نقاب;
    نتیجہ->src[0] = نقاب;
    نتیجہ->src[1] = چوٹی_k;
    نتیجہ->src[2] = score;
    نتیجہ->src[3] = بلاکیں;
    نتیجہ->src[4] = extra;

    واپس نتیجہ;
}

////////////////////////////////////////////////////////////////////////////////

ساخت ggml_hash_سیٹ ggml_hash_سیٹ_new(size_t سائز) {
    سائز = ggml_hash_سائز(سائز);
    ساخت ggml_hash_سیٹ نتیجہ;
    نتیجہ.سائز = سائز;
    نتیجہ.کلیدیں = GGML_MALLOC(حجم(ساخت ggml_tensor *) * سائز);
    نتیجہ.مستعمل = GGML_CALLOC(ggml_bitset_سائز(سائز), حجم(ggml_bitset_t));
    واپس نتیجہ;
}

خالی ggml_hash_سیٹ_reset(ساخت ggml_hash_سیٹ * hash_سیٹ) {
    memset(hash_سیٹ->مستعمل, 0, حجم(ggml_bitset_t) * ggml_bitset_سائز(hash_سیٹ->سائز));
}

خالی ggml_hash_سیٹ_free(ساخت ggml_hash_سیٹ * hash_سیٹ) {
    GGML_FREE(hash_سیٹ->مستعمل);
    GGML_FREE(hash_سیٹ->کلیدیں);
}

size_t ggml_hash_سائز(size_t min_sz) {
    // next primes after powers of two
    ساکن ثابت size_t مفرد_اعداد[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ساکن ثابت size_t n_مفرد_اعداد = حجم(مفرد_اعداد)/حجم(مفرد_اعداد[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_مفرد_اعداد;
    جب (l < r) {
        size_t m = (l + r)/2;
        اگر (مفرد_اعداد[m] < min_sz) {
            l = m + 1;
        } ورنہ {
            r = m;
        }
    }
    size_t sz = l < n_مفرد_اعداد ? مفرد_اعداد[l] : min_sz | 1;
    واپس sz;
}

ساخت hash_map {
    ساخت ggml_hash_سیٹ سیٹ;
    ساخت ggml_tensor ** vals;
};

ساکن ساخت hash_map * ggml_new_hash_map(size_t سائز) {
    ساخت hash_map * نتیجہ = GGML_MALLOC(حجم(ساخت hash_map));
    نتیجہ->سیٹ = ggml_hash_سیٹ_new(سائز);
    نتیجہ->vals = GGML_CALLOC(نتیجہ->سیٹ.سائز, حجم(ساخت ggml_tensor *));
    واپس نتیجہ;
}

ساکن خالی ggml_hash_map_free(ساخت hash_map * map) {
    ggml_hash_سیٹ_free(&map->سیٹ);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

ساکن خالی ggml_جوڑو_or_سیٹ(
        ساخت ggml_context * ctx,
        ساخت ggml_cgraph  * cgraph,
        size_t                isrc,
        ساخت ggml_tensor  * tensor) {
    ساخت ggml_tensor * src = cgraph->دیکھا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_جوڑو_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } ورنہ {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساکن خالی ggml_acc_or_سیٹ(
        ساخت ggml_context * ctx,
        ساخت ggml_cgraph  * cgraph,
        size_t                isrc,
        ساخت ggml_tensor  * tensor,
        ثابت  size_t         nb1,
        ثابت  size_t         nb2,
        ثابت  size_t         nb3,
        ثابت  size_t         انحراف) {
    ساخت ggml_tensor * src = cgraph->دیکھا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, انحراف, cgraph->grad_accs[isrc]);
    } ورنہ {
        ساخت ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, انحراف, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->دیکھا_hash_سیٹ.کلیدیں[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساکن خالی ggml_add1_or_سیٹ(
        ساخت ggml_context * ctx,
        ساخت ggml_cgraph  * cgraph,
        size_t                isrc,
        ساخت ggml_tensor  * tensor) {
    ساخت ggml_tensor * src = cgraph->دیکھا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ورنہ {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساکن خالی ggml_sub_or_سیٹ(
        ساخت ggml_context * ctx,
        ساخت ggml_cgraph  * cgraph,
        size_t                isrc,
        ساخت ggml_tensor  * tensor) {
    ساخت ggml_tensor * src = cgraph->دیکھا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ورنہ {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساکن خالی ggml_compute_backward(
        ساخت ggml_context * ctx, ساخت ggml_cgraph * cgraph, صحیح i, ثابت bool * grads_needed) {
    ساخت ggml_tensor * tensor = cgraph->nodes[i];
    ساخت ggml_tensor * grad   = ggml_گراف_get_grad(cgraph, tensor);

    اگر (!grad) {
        واپس;
    }

    ساخت ggml_tensor * src0 = tensor->src[0];
    ساخت ggml_tensor * src1 = tensor->src[1];
    ساخت ggml_tensor * src2 = tensor->src[2];
    ساخت ggml_hash_سیٹ * hash_سیٹ = &cgraph->دیکھا_hash_سیٹ;
    ثابت size_t isrc0 = src0 ? ggml_hash_ڈھونڈو(hash_سیٹ, src0) : (size_t) -1;
    ثابت size_t isrc1 = src1 ? ggml_hash_ڈھونڈو(hash_سیٹ, src1) : (size_t) -1;
    ثابت size_t isrc2 = src2 ? ggml_hash_ڈھونڈو(hash_سیٹ, src2) : (size_t) -1;
    ثابت bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_بھرا && ggml_bitset_get(hash_سیٹ->مستعمل, isrc0) && grads_needed[isrc0];
    ثابت bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_بھرا && ggml_bitset_get(hash_سیٹ->مستعمل, isrc1) && grads_needed[isrc1];
    ثابت bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_بھرا && ggml_bitset_get(hash_سیٹ->مستعمل, isrc2) && grads_needed[isrc2];

    انتخاب (tensor->op) {
        حالت GGML_OP_DUP: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
        } توڑو;
        حالت GGML_OP_جوڑو: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ساخت ggml_tensor * tmp = grad;
                اگر (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_عقب(ctx, tmp, src1);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, tmp);
            }
        } توڑو;
        حالت GGML_OP_ADD1: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_اوسط(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } توڑو;
        حالت GGML_OP_ACC: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ثابت size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ثابت size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ثابت size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ثابت size_t انحراف = ((int32_t *) tensor->op_params)[3];

                ساخت ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, انحراف);

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } توڑو;
        حالت GGML_OP_SUB: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc1, grad);
            }
        } توڑو;
        حالت GGML_OP_MUL: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            اگر (src1_needs_grads) {
                ساخت ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                اگر (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_عقب(ctx, tmp, src1);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, tmp);
            }
        } توڑو;
        حالت GGML_OP_DIV: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            اگر (src1_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } توڑو;
        حالت GGML_OP_SQR: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } توڑو;
        حالت GGML_OP_SQRT: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } توڑو;
        حالت GGML_OP_LOG: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } توڑو;
        حالت GGML_OP_SIN: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } توڑو;
        حالت GGML_OP_COS: {
            اگر (src0_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } توڑو;
        حالت GGML_OP_مجموع: {
            اگر (src0_needs_grads) {
                ggml_add1_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
        } توڑو;
        حالت GGML_OP_مجموع_صفیں: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } توڑو;
        حالت GGML_OP_اوسط: {
            اگر (src0_needs_grads) {
                ggml_add1_or_سیٹ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } توڑو;
        حالت GGML_OP_REPEAT: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat_عقب(ctx, grad, src0));
            }
        } توڑو;
        حالت GGML_OP_REPEAT_عقب: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } توڑو;
        حالت GGML_OP_RMS_NORM: {
            اگر (src0_needs_grads) {
                اعشاری eps;
                حافظہ_نقل(&eps, tensor->op_params, حجم(اعشاری));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_rms_norm_عقب(ctx, grad, src0, eps));
            }
        } توڑو;
        حالت GGML_OP_MUL_MAT: {
            // https://cs231n.github.io/optimization-2/#staged
            // # forward pass
            // s0 = np.random.randn(5, 10)
            // s1 = np.random.randn(10, 3)
            // t = s0.dot(s1)

            // # now suppose we had the gradient on t from above in the circuit
            // dt = np.random.randn(*t.shape) # same shape as t
            // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
            // ds1 = t.T.dot(dt)

            // tensor.shape [m,p,qq,rr]
            // src0.shape   [n,m,q1,r1]
            // src1.shape   [n,p,qq,rr]

            اگر (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ساخت ggml_tensor * tmp =
                    ggml_برآمد_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                اگر (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ثابت int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ثابت size_t nb2 = tmp->nb[2] * nr2;
                    ثابت size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_عقب(ctx, tmp, src0);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, tmp);
            }
            اگر (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_برآمد_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } توڑو;
        حالت GGML_OP_SCALE: {
            اگر (src0_needs_grads) {
                اعشاری s;
                حافظہ_نقل(&s, tensor->op_params, حجم(اعشاری));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } توڑو;
        حالت GGML_OP_سیٹ: {
            ثابت size_t nb1    = ((ثابت int32_t *) tensor->op_params)[0];
            ثابت size_t nb2    = ((ثابت int32_t *) tensor->op_params)[1];
            ثابت size_t nb3    = ((ثابت int32_t *) tensor->op_params)[2];
            ثابت size_t انحراف = ((ثابت int32_t *) tensor->op_params)[3];

            ساخت ggml_tensor * tensor_grad_view = تہی;

            اگر (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, انحراف);
            }

            اگر (src0_needs_grads) {
                ساخت ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, انحراف, false));
            }

            اگر (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } توڑو;
        حالت GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            اگر (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            اگر (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } توڑو;
        حالت GGML_OP_CONT: {
            // same as cpy
            اگر (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } توڑو;
        حالت GGML_OP_RESHAPE: {
            اگر (src0_needs_grads) {
                ساخت ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } توڑو;
        حالت GGML_OP_VIEW: {
            اگر (src0_needs_grads) {
                size_t انحراف;

                حافظہ_نقل(&انحراف, tensor->op_params, حجم(انحراف));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                اگر (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_سائز(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_سائز(src0);
                    GGML_ASSERT(انحراف % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    انحراف = (انحراف / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_سیٹ(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, انحراف);
            }
        } توڑو;
        حالت GGML_OP_ترتیب_بدلو: {
            اگر (src0_needs_grads) {
                ثابت int32_t * axes = (ثابت int32_t *) tensor->op_params;
                ثابت صحیح axis0 = axes[0] & 0x3;
                ثابت صحیح axis1 = axes[1] & 0x3;
                ثابت صحیح axis2 = axes[2] & 0x3;
                ثابت صحیح axis3 = axes[3] & 0x3;
                صحیح axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_ترتیب_بدلو(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } توڑو;
        حالت GGML_OP_TRANSPOSE: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } توڑو;
        حالت GGML_OP_GET_صفیں: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_get_صفیں_عقب(ctx, grad, src1, src0));
            }
            اگر (src1_needs_grads) {
                // noop
            }
        } توڑو;
        حالت GGML_OP_DIAG_نقاب_INF: {
            اگر (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ثابت صحیح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_diag_نقاب_zero_impl(ctx, grad, n_past, false));
            }
        } توڑو;
        حالت GGML_OP_DIAG_نقاب_ZERO: {
            اگر (src0_needs_grads) {
                ثابت صحیح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_diag_نقاب_zero_impl(ctx, grad, n_past, false));
            }
        } توڑو;
        حالت GGML_OP_SOFT_MAX: {
            اگر (src0_needs_grads) {
                اعشاری scale    = 1.0f;
                اعشاری max_bias = 0.0f;

                حافظہ_نقل(&scale,    (ثابت اعشاری *) tensor->op_params + 0, حجم(اعشاری));
                حافظہ_نقل(&max_bias, (ثابت اعشاری *) tensor->op_params + 1, حجم(اعشاری));

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_soft_max_ext_عقب(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } توڑو;
        حالت GGML_OP_ROPE: {
            اگر (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ثابت صحیح n_dims     = ((ثابت int32_t *) tensor->op_params)[1];
                ثابت صحیح منوال       = ((ثابت int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ثابت صحیح n_ctx_orig = ((ثابت int32_t *) tensor->op_params)[4];
                اعشاری freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow;
                صحیح sections[4] = {0, 0, 0, 0};

                حافظہ_نقل(&freq_بنیاد,   (ثابت اعشاری *) tensor->op_params +  5, حجم(اعشاری));
                حافظہ_نقل(&freq_scale,  (ثابت اعشاری *) tensor->op_params +  6, حجم(اعشاری));
                حافظہ_نقل(&ext_ضربی_عدد,  (ثابت اعشاری *) tensor->op_params +  7, حجم(اعشاری));
                حافظہ_نقل(&attn_ضربی_عدد, (ثابت اعشاری *) tensor->op_params +  8, حجم(اعشاری));
                حافظہ_نقل(&beta_fast,   (ثابت اعشاری *) tensor->op_params +  9, حجم(اعشاری));
                حافظہ_نقل(&beta_slow,   (ثابت اعشاری *) tensor->op_params + 10, حجم(اعشاری));
                حافظہ_نقل(&sections,                    tensor->op_params + 11, حجم(sections));

                ساخت ggml_tensor * rope_عقب = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_عقب(ctx, grad, src1, src2, n_dims,
                        منوال, n_ctx_orig, freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow) :
                    ggml_rope_multi_عقب(ctx, grad, src1, src2, n_dims, sections,
                        منوال, n_ctx_orig, freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, rope_عقب);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } توڑو;
        حالت GGML_OP_IM2_ستون: {
            اگر (src1_needs_grads) {
                ثابت int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ثابت int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ثابت int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ثابت int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ثابت int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ثابت int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ثابت bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_im2col_عقب(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } توڑو;
        حالت GGML_OP_POOL_2D: {
            اگر (src0_needs_grads) {
                ثابت شمار ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ثابت      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ثابت      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ثابت      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ثابت      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ثابت      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ثابت      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_pool_2d_عقب(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } توڑو;
        حالت GGML_OP_WIN_PART:
        حالت GGML_OP_WIN_UNPART:
        حالت GGML_OP_UNARY: {
            انتخاب (ggml_get_unary_op(tensor)) {
                حالت GGML_UNARY_OP_ABS: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } توڑو;
                حالت GGML_UNARY_OP_SGN: {
                    // noop
                } توڑو;
                حالت GGML_UNARY_OP_NEG: {
                    اگر (src0_needs_grads) {
                        ggml_sub_or_سیٹ(ctx, cgraph, isrc0, grad);
                    }
                } توڑو;
                حالت GGML_UNARY_OP_قدم: {
                    // noop
                } توڑو;
                حالت GGML_UNARY_OP_RELU: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_قدم(ctx, src0), grad));
                    }
                } توڑو;
                حالت GGML_UNARY_OP_SILU: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_silu_عقب(ctx, grad, src0));
                    }
                } توڑو;
                حالت GGML_UNARY_OP_EXP: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } توڑو;
                حالت GGML_UNARY_OP_EXPM1: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } توڑو;
                حالت GGML_UNARY_OP_SOFTPLUS: {
                    اگر (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } توڑو;
                طےشدہ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS: {
            اگر (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_cross_entropy_loss_عقب(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } توڑو;
        حالت GGML_OP_GLU: {
            انتخاب (ggml_get_glu_op(tensor)) {
                حالت GGML_GLU_OP_SWIGLU: {
                    اگر (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_silu_عقب(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    اگر (src1_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } توڑو;
                طےشدہ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } توڑو;
        حالت GGML_OP_NONE: {
            // noop
        } توڑو;
        حالت GGML_OP_گنتی:
        طےشدہ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ساکن size_t ggml_دورہ_parents_گراف(ساخت ggml_cgraph * cgraph, ساخت ggml_tensor * گرہ, bool compute) {
    اگر (گرہ->op != GGML_OP_NONE && compute) {
        گرہ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ثابت size_t گرہ_hash_pos = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ);
    GGML_ASSERT(گرہ_hash_pos != GGML_HASHSET_بھرا);

    اگر (ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, گرہ_hash_pos)) {
        // already visited

        اگر (compute) {
            // update the compute flag regardless
            برائے (صحیح i = 0; i < GGML_MAX_SRC; ++i) {
                ساخت ggml_tensor * src = گرہ->src[i];
                اگر (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_دورہ_parents_گراف(cgraph, src, true);
                }
            }
        }

        واپس گرہ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->دیکھا_hash_سیٹ.کلیدیں[گرہ_hash_pos] = گرہ;
    ggml_bitset_سیٹ(cgraph->دیکھا_hash_سیٹ.مستعمل, گرہ_hash_pos);
    cgraph->use_counts[گرہ_hash_pos] = 0;

    برائے (صحیح i = 0; i < GGML_MAX_SRC; ++i) {
        ثابت صحیح k =
            (cgraph->ترتیب == GGML_CGRAPH_EVAL_ترتیب_بایاں_مقصد_دایاں) ? i :
            (cgraph->ترتیب == GGML_CGRAPH_EVAL_ترتیب_دایاں_مقصد_بایاں) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ساخت ggml_tensor * src = گرہ->src[k];
        اگر (src) {
            ثابت size_t src_hash_pos = ggml_دورہ_parents_گراف(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    اگر (گرہ->op == GGML_OP_NONE && !(گرہ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->سائز);

        اگر (لڑی_لمبائی(گرہ->name) == 0) {
            ggml_format_name(گرہ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = گرہ;
        cgraph->n_leafs++;
    } ورنہ {
        GGML_ASSERT(cgraph->n_nodes < cgraph->سائز);

        اگر (لڑی_لمبائی(گرہ->name) == 0) {
            ggml_format_name(گرہ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = گرہ;
        cgraph->n_nodes++;
    }

    واپس گرہ_hash_pos;
}

ساکن خالی ggml_build_forward_impl(ساخت ggml_cgraph * cgraph, ساخت ggml_tensor * tensor, bool expand, bool compute) {
    اگر (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_گراف_clear(cgraph);
    }

    ثابت صحیح n_old = cgraph->n_nodes;

    ggml_دورہ_parents_گراف(cgraph, tensor, compute);

    ثابت صحیح n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    اگر (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ساخت ggml_tensor * ggml_build_forward_select(
        ساخت ggml_cgraph  * cgraph,
        ساخت ggml_tensor ** tensors,
        صحیح                   n_tensors,
        صحیح                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    برائے (صحیح i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    واپس tensors[idx];
}

خالی ggml_build_forward_expand(ساخت ggml_cgraph * cgraph, ساخت ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

خالی ggml_build_forward_ترتیب(ساخت ggml_cgraph * cgraph, ساخت ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

خالی ggml_build_backward_expand(
        ساخت ggml_context *  ctx,
        ساخت ggml_cgraph  *  cgraph,
        ساخت ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ثابت صحیح n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->دیکھا_hash_سیٹ.سائز*حجم(ساخت ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->دیکھا_hash_سیٹ.سائز*حجم(ساخت ggml_tensor *));
    bool * grads_needed = صفر_حافظہ_دو(cgraph->دیکھا_hash_سیٹ.سائز, حجم(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        برائے (صحیح i = 0; i < n_nodes_f; ++i) {
            ساخت ggml_tensor * گرہ = cgraph->nodes[i];
            any_params = any_params || (گرہ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (گرہ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    برائے (صحیح i = 0; i < n_nodes_f; ++i) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[i];

        اگر (گرہ->type == GGML_TYPE_I32) {
            جاری;
        }

        bool گرہ_needs_grad = (گرہ->flags & GGML_TENSOR_FLAG_PARAM) || (گرہ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        انتخاب (گرہ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            حالت GGML_OP_IM2_ستون:      // only used for its shape
            حالت GGML_OP_IM2_ستون_عقب: // same as IM2COL
                ignore_src[0] = true;
                توڑو;
            حالت GGML_OP_UNARY: {
                ثابت شمار ggml_unary_op uop = ggml_get_unary_op(گرہ);
                // SGN and STEP unary ops are piecewise constant
                اگر (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_قدم) {
                    ignore_src[0] = true;
                }
            } توڑو;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            حالت GGML_OP_CPY:           // gradients in CPY target are irrelevant
            حالت GGML_OP_GET_صفیں:      // row indices not differentiable
            حالت GGML_OP_GET_صفیں_عقب: // same as for GET_ROWS
            حالت GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                توڑو;

            طےشدہ:
                توڑو;
        }
        برائے (صحیح j = 0; j < GGML_MAX_SRC; ++j) {
            اگر (!گرہ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ->src[j])]) {
                جاری;
            }
            GGML_ASSERT(گرہ->src[j]->type == GGML_TYPE_F32 || گرہ->src[j]->type == GGML_TYPE_F16);
            گرہ_needs_grad = true;
            توڑو;
        }
        اگر (!گرہ_needs_grad) {
            جاری;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!گرہ->view_src || گرہ->op == GGML_OP_CPY || گرہ->op == GGML_OP_سیٹ_صفیں || گرہ->op == GGML_OP_VIEW ||
            گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_ترتیب_بدلو || گرہ->op == GGML_OP_TRANSPOSE);

        ثابت size_t ihash = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ);
        GGML_ASSERT(ihash != GGML_HASHSET_بھرا);
        GGML_ASSERT(ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, ihash));
        اگر (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } ورنہ اگر (گرہ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    برائے (صحیح i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    آزاد(grads_needed);
}

ساکن خالی * incr_ptr_aligned(خالی ** p, size_t سائز, size_t align) {
    خالی * ptr = *p;
    ptr = (خالی *) GGML_PAD((uintptr_t) ptr, align);
    *p = (خالی *) ((حرف *) ptr + سائز);
    واپس ptr;
}

ساکن size_t ggml_گراف_nbytes(size_t سائز, bool grads) {
    size_t hash_سائز = ggml_hash_سائز(سائز * 2);
    خالی * p = (حرف *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, حجم(ساخت ggml_cgraph), 1);
    incr_ptr_aligned(&p, سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_سائز * حجم(int32_t), حجم(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)); // hash keys
    اگر (grads) {
        incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_سائز(hash_سائز) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    واپس nbytes - 1024;
}

size_t ggml_گراف_overhead_custom(size_t سائز, bool grads) {
    واپس GGML_OBJECT_سائز + GGML_PAD(ggml_گراف_nbytes(سائز, grads), GGML_MEM_ALIGN);
}

size_t ggml_گراف_overhead(خالی) {
    واپس ggml_گراف_overhead_custom(GGML_DEFAULT_گراف_سائز, false);
}

ساخت ggml_cgraph * ggml_new_گراف_custom(ساخت ggml_context * ctx, size_t سائز, bool grads) {
    ثابت size_t obj_سائز = ggml_گراف_nbytes(سائز, grads);
    ساخت ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_گراف, obj_سائز);
    ساخت ggml_cgraph * cgraph = (ساخت ggml_cgraph *) ((حرف *) ctx->mem_بفر + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_سائز = ggml_hash_سائز(سائز * 2);

    خالی * p = cgraph + 1;

    ساخت ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, سائز      * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *));
    ساخت ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, سائز      * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_سائز * حجم(int32_t), حجم(int32_t));
    ساخت ggml_tensor ** hash_کلیدیں_ptr  =         incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *));
    ساخت ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)) : تہی;
    ساخت ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_سائز * حجم(ساخت ggml_tensor *), حجم(ساخت ggml_tensor *)) : تہی;

    ggml_bitset_t * hash_مستعمل = incr_ptr_aligned(&p, ggml_bitset_سائز(hash_سائز) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    أكد(obj_سائز == (size_t)((حرف *)p - (حرف *)cgraph));

    *cgraph = (ساخت ggml_cgraph) {
        /*.size         =*/ سائز,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_سائز, hash_مستعمل, hash_کلیدیں_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ترتیب_بایاں_مقصد_دایاں,
        /*.uid          =*/ 0,
    };

    ggml_hash_سیٹ_reset(&cgraph->دیکھا_hash_سیٹ);
    اگر (grads) {
        memset(cgraph->grads,     0, hash_سائز*حجم(ساخت ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_سائز*حجم(ساخت ggml_tensor *));
    }

    واپس cgraph;
}

ساخت ggml_cgraph * ggml_new_گراف(ساخت ggml_context * ctx) {
    واپس ggml_new_گراف_custom(ctx, GGML_DEFAULT_گراف_سائز, false);
}

ساخت ggml_cgraph ggml_گراف_view(ساخت ggml_cgraph * cgraph0, صحیح i0, صحیح i1) {
    ساخت ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ تہی, // gradients would need visited_hash_set
        /*.grad_accs        =*/ تہی,
        /*.leafs            =*/ تہی,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->دیکھا_hash_سیٹ,
        /*.order            =*/ cgraph0->ترتیب,
        /*.uid              =*/ 0
    };

    واپس cgraph;
}

خالی ggml_گراف_cpy(ساخت ggml_cgraph * src, ساخت ggml_cgraph * dst) {
    GGML_ASSERT(dst->سائز >= src->n_leafs);
    GGML_ASSERT(dst->سائز >= src->n_nodes);
    GGML_ASSERT(dst->دیکھا_hash_سیٹ.سائز >= src->دیکھا_hash_سیٹ.سائز);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->ترتیب   = src->ترتیب;

    برائے (صحیح i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    برائے (صحیح i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    برائے (size_t i = 0; i < src->دیکھا_hash_سیٹ.سائز; ++i) {
        // copy all hashset keys (tensors) that are in use
        اگر (ggml_bitset_get(src->دیکھا_hash_سیٹ.مستعمل, i)) {
            size_t new_hash_pos = ggml_hash_داخل_کرو(&dst->دیکھا_hash_سیٹ, src->دیکھا_hash_سیٹ.کلیدیں[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    اگر (dst->grads) {
        memset(dst->grads,     0, dst->دیکھا_hash_سیٹ.سائز*حجم(ساخت ggml_tensor *));
        memset(dst->grad_accs, 0, dst->دیکھا_hash_سیٹ.سائز*حجم(ساخت ggml_tensor *));
    }
    اگر (src->grads) {
        GGML_ASSERT(dst->grads     != تہی);
        GGML_ASSERT(dst->grad_accs != تہی);
        برائے (صحیح i = 0; i < src->n_nodes; ++i) {
            ثابت size_t igrad_src = ggml_hash_ڈھونڈو(&src->دیکھا_hash_سیٹ, src->nodes[i]);
            ثابت size_t igrad_dst = ggml_hash_ڈھونڈو(&dst->دیکھا_hash_سیٹ, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_بھرا);
            GGML_ASSERT(ggml_bitset_get(src->دیکھا_hash_سیٹ.مستعمل, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_بھرا);
            GGML_ASSERT(ggml_bitset_get(dst->دیکھا_hash_سیٹ.مستعمل, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ساخت ggml_cgraph * ggml_گراف_dup(ساخت ggml_context * ctx, ساخت ggml_cgraph * cgraph, bool force_grads) {
    ساخت ggml_cgraph * نتیجہ = ggml_new_گراف_custom(ctx, cgraph->سائز, cgraph->grads || force_grads);
    ggml_گراف_cpy(cgraph, نتیجہ);
    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_سیٹ_zero(ساخت ggml_tensor * tensor) {
    اگر (ggml_is_خالی_خانے(tensor)) {
        واپس tensor;
    }
    اگر (tensor->بفر) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } ورنہ {
        GGML_ASSERT(tensor->ڈیٹا);
        memset(tensor->ڈیٹا, 0, ggml_nbytes(tensor));
    }
    واپس tensor;
}

خالی ggml_گراف_reset(ساخت ggml_cgraph * cgraph) {
    اگر (!cgraph) {
        واپس;
    }
    GGML_ASSERT(cgraph->grads != تہی);

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ     = cgraph->nodes[i];
        ساخت ggml_tensor * grad_acc = ggml_گراف_get_grad_acc(cgraph, گرہ);

        اگر (گرہ->op == GGML_OP_OPT_قدم_ADAMW) {
            // clear momenta
            ggml_سیٹ_zero(گرہ->src[2]);
            ggml_سیٹ_zero(گرہ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        اگر (grad_acc) {
            اگر (گرہ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ثابت اعشاری onef = 1.0f;
                اگر (grad_acc->بفر) {
                    ggml_backend_tensor_سیٹ(grad_acc, &onef, 0, حجم(اعشاری));
                } ورنہ {
                    GGML_ASSERT(grad_acc->ڈیٹا);
                    *((اعشاری *) grad_acc->ڈیٹا) = onef;
                }
            } ورنہ {
                ggml_سیٹ_zero(grad_acc);
            }
        }
    }
}

خالی ggml_گراف_clear(ساخت ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_سیٹ_reset(&cgraph->دیکھا_hash_سیٹ);
}

صحیح ggml_گراف_سائز(ساخت ggml_cgraph * cgraph) {
    واپس cgraph->سائز;
}

ساخت ggml_tensor * ggml_گراف_گرہ(ساخت ggml_cgraph * cgraph, صحیح i) {
    اگر (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        واپس cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    واپس cgraph->nodes[i];
}

ساخت ggml_tensor ** ggml_گراف_nodes(ساخت ggml_cgraph * cgraph) {
    واپس cgraph->nodes;
}

صحیح ggml_گراف_n_nodes(ساخت ggml_cgraph * cgraph) {
    واپس cgraph->n_nodes;
}

خالی ggml_گراف_جوڑو_گرہ(ساخت ggml_cgraph * cgraph, ساخت ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->سائز > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ساخت ggml_tensor * ggml_گراف_get_tensor(ثابت ساخت ggml_cgraph * cgraph, ثابت حرف * name) {
    برائے (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ساخت ggml_tensor * برگ = cgraph->leafs[i];

        اگر (strcmp(برگ->name, name) == 0) {
            واپس برگ;
        }
    }

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[i];

        اگر (strcmp(گرہ->name, name) == 0) {
            واپس گرہ;
        }
    }

    واپس تہی;
}

ساخت ggml_tensor * ggml_گراف_get_grad(ثابت ساخت ggml_cgraph * cgraph, ثابت ساخت ggml_tensor * گرہ) {
    ثابت size_t igrad = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ);
    واپس igrad != GGML_HASHSET_بھرا && ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, igrad) && cgraph->grads ? cgraph->grads[igrad] : تہی;
}

ساخت ggml_tensor * ggml_گراف_get_grad_acc(ثابت ساخت ggml_cgraph * cgraph, ثابت ساخت ggml_tensor * گرہ) {
    ثابت size_t igrad = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ);
    واپس igrad != GGML_HASHSET_بھرا && ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : تہی;
}

خالی ggml_گراف_print(ثابت ساخت ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                گرہ->ne[0], گرہ->ne[1], گرہ->ne[2],
                ggml_op_name(گرہ->op), (گرہ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_گراف_get_grad(cgraph, گرہ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    برائے (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ساخت ggml_tensor * گرہ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                گرہ->ne[0], گرہ->ne[1],
                ggml_op_name(گرہ->op),
                ggml_get_name(گرہ));
    }

    GGML_LOG_INFO("========================================\n");
}

ساکن صحیح ggml_گرہ_list_ڈھونڈو_tensor(ثابت ساخت ggml_cgraph * cgraph,
                                      ثابت صحیح *                idxs,
                                      صحیح                        گنتی,
                                      ثابت ساخت ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    برائے (صحیح i = 0; i < گنتی; ++i) {
        ثابت صحیح گرہ_idx = idxs[i];

        اگر (گرہ_idx >= cgraph->n_nodes) {
            واپس -1;
        }
        اگر (cgraph->nodes[گرہ_idx] == tensor) {
            واپس i;
        }
    }
    واپس -1;
}

ساکن bool ggml_is_constant(ثابت ساخت ggml_tensor * tensor) {
    واپس tensor->بفر != تہی && ggml_backend_بفر_get_usage(tensor->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ثابت ساخت ggml_cgraph * cgraph,
                                ثابت صحیح *                گرہ_idxs,
                                صحیح                        گنتی,
                                ثابت شمار ggml_op *       ops,
                                ثابت صحیح *                outputs,
                                صحیح                        عدد_outputs) {
    GGML_ASSERT(outputs && عدد_outputs > 0);

    برائے (صحیح i = 0; i < گنتی; ++i) {
        اگر (گرہ_idxs[i] >= cgraph->n_nodes) {
            واپس false;
        }

        ثابت ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_idxs[i]];

        اگر (گرہ->op != ops[i]) {
            واپس false;
        }

        اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            واپس false;
        }

        اگر (ggml_گرہ_list_ڈھونڈو_tensor(cgraph, outputs, عدد_outputs, گرہ) != -1) {
            جاری;
        }

        اگر (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            واپس false;
        }

        صحیح subgraph_استعمال_کرتا = 0;
        برائے (صحیح j = i + 1; j < گنتی; ++j) {
            ثابت ساخت ggml_tensor * other_گرہ = cgraph->nodes[گرہ_idxs[j]];
            برائے (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                اگر (other_گرہ->src[src_idx] == گرہ) {
                    subgraph_استعمال_کرتا++;
                }
            }
        }

        اگر (subgraph_استعمال_کرتا != ggml_گرہ_get_use_گنتی(cgraph, گرہ_idxs[i])) {
            واپس false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ساخت ggml_tensor * view_src = گرہ->view_src;
        جب (view_src) {
            اگر (ggml_گرہ_list_ڈھونڈو_tensor(cgraph, گرہ_idxs, گنتی, view_src) == -1 && !ggml_is_constant(view_src)) {
                واپس false;
            }
            view_src = view_src->view_src;
        }
    }

    واپس true;
}

// check if node is part of the graph
ساکن bool ggml_گراف_ڈھونڈو(ثابت ساخت ggml_cgraph * cgraph, ثابت ساخت ggml_tensor * گرہ) {
    اگر (cgraph == تہی) {
        واپس true;
    }

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        اگر (cgraph->nodes[i] == گرہ) {
            واپس true;
        }
    }

    واپس false;
}

ساکن ساخت ggml_tensor * ggml_گراف_get_والد(ثابت ساخت ggml_cgraph * cgraph, ثابت ساخت ggml_tensor * گرہ) {
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * والد = cgraph->nodes[i];
        ساخت ggml_tensor * grad = ggml_گراف_get_grad(cgraph, والد);

        اگر (grad == گرہ) {
            واپس والد;
        }
    }

    واپس تہی;
}

ساکن خالی ggml_گراف_dump_dot_گرہ_کنارہ(FILE * fp, ثابت ساخت ggml_cgraph * gb, ساخت ggml_tensor * گرہ, ساخت ggml_tensor * والد, ثابت حرف * label)  {
    ساخت ggml_tensor * gparent = ggml_گراف_get_والد(gb, گرہ);
    ساخت ggml_tensor * gparent0 = ggml_گراف_get_والد(gb, والد);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (خالی *) gparent0 : (خالی *) والد,
            gparent ? (خالی *) gparent : (خالی *) گرہ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ساکن خالی ggml_گراف_dump_dot_برگ_کنارہ(FILE * fp, ساخت ggml_tensor * گرہ, ساخت ggml_tensor * والد, ثابت حرف * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (خالی *) والد,
            (خالی *) گرہ,
            label);
}

خالی ggml_گراف_dump_dot(ثابت ساخت ggml_cgraph * gb, ثابت ساخت ggml_cgraph * cgraph, ثابت حرف * filename) {
    حرف color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    برائے (صحیح i = 0; i < gb->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = gb->nodes[i];
        ساخت ggml_tensor * grad = ggml_گراف_get_grad(gb, گرہ);

        اگر (ggml_گراف_get_والد(gb, گرہ) != تہی) {
            جاری;
        }

        اگر (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, حجم(color), "yellow");
        } ورنہ اگر (grad) {
            اگر (ggml_گراف_ڈھونڈو(cgraph, گرہ)) {
                snprintf(color, حجم(color), "green");
            } ورنہ {
                snprintf(color, حجم(color), "lightblue");
            }
        } ورنہ {
            snprintf(color, حجم(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (خالی *) گرہ, color);

        اگر (لڑی_لمبائی(گرہ->name) > 0) {
            fprintf(fp, "%s (%s)|", گرہ->name, ggml_type_name(گرہ->type));
        } ورنہ {
            fprintf(fp, "(%s)|", ggml_type_name(گرہ->type));
        }

        اگر (ggml_is_میٹرکس(گرہ)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, گرہ->ne[0], گرہ->ne[1], ggml_op_علامت(گرہ->op));
        } ورنہ {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, گرہ->ne[0], گرہ->ne[1], گرہ->ne[2], ggml_op_علامت(گرہ->op));
        }

        اگر (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_علامت(grad->op));
        } ورنہ {
            fprintf(fp, "\"; ]\n");
        }
    }

    برائے (صحیح i = 0; i < gb->n_leafs; i++) {
        ساخت ggml_tensor * گرہ = gb->leafs[i];

        snprintf(color, حجم(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (خالی *) گرہ, color);

        اگر (لڑی_لمبائی(گرہ->name) > 0) {
            fprintf(fp, "%s (%s)|", گرہ->name, ggml_type_name(گرہ->type));
        } ورنہ {
            fprintf(fp, "(%s)|", ggml_type_name(گرہ->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, گرہ->ne[0], گرہ->ne[1]);
        اگر (ggml_nelements(گرہ) < 5 && گرہ->ڈیٹا != تہی) {
            fprintf(fp, " | (");
            برائے (صحیح j = 0; j < ggml_nelements(گرہ); j++) {
                // FIXME: use ggml-backend to obtain the tensor data
                //if (node->type == GGML_TYPE_I8 || node->type == GGML_TYPE_I16 || node->type == GGML_TYPE_I32) {
                //    fprintf(fp, "%d", ggml_get_i32_1d(node, j));
                //}
                //else if (node->type == GGML_TYPE_F32 ||
                //         node->type == GGML_TYPE_F16 ||
                //         node->type == GGML_TYPE_BF16) {
                //    fprintf(fp, "%.1e", (double)ggml_get_f32_1d(node, j));
                //}
                //else
                {
                    fprintf(fp, "#");
                }
                اگر (j < ggml_nelements(گرہ) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    برائے (صحیح i = 0; i < gb->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = gb->nodes[i];

        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گرہ->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_گراف_dump_dot_گرہ_کنارہ(fp, gb, گرہ, گرہ->src[j], label);
            }
        }
    }

    برائے (صحیح i = 0; i < gb->n_leafs; i++) {
        ساخت ggml_tensor * گرہ = gb->leafs[i];

        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گرہ->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_گراف_dump_dot_برگ_کنارہ(fp, گرہ, گرہ->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

خالی ggml_سیٹ_داخلہ(ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_داخلہ;
}

خالی ggml_سیٹ_output(ساخت ggml_tensor * tensor) {
    برائے (ساخت ggml_tensor * cur = tensor; cur != تہی; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

خالی ggml_سیٹ_param(ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

خالی ggml_سیٹ_loss(ساخت ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

خالی ggml_quantize_init(شمار ggml_type type) {
    ggml_critical_section_آغاز();

    انتخاب (type) {
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); توڑو;
        حالت GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); توڑو;
        حالت GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); توڑو;
        طےشدہ: // nothing
            توڑو;
    }

    ggml_critical_section_انتہا();
}

خالی ggml_quantize_free(خالی) {
    ggml_critical_section_آغاز();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_انتہا();
}

bool ggml_quantize_requires_imatrix(شمار ggml_type type) {
    واپس
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        شمار ggml_type   type,
           ثابت اعشاری * src,
                  خالی * dst,
               int64_t   آغاز,
               int64_t   nrows,
               int64_t   n_فی_صف,
           ثابت اعشاری * imatrix) {
    ثابت int64_t n = nrows * n_فی_صف;

    اگر (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != تہی);
    }

    GGML_ASSERT(آغاز % type_traits[type].blck_سائز == 0);
    GGML_ASSERT(آغاز % n_فی_صف == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ثابت size_t آغاز_صف = آغاز / n_فی_صف;
    ثابت size_t صف_سائز  = ggml_صف_سائز(type, n_فی_صف);

    size_t نتیجہ = 0;

    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:    نتیجہ = quantize_q1_0   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q2_0:    نتیجہ = quantize_q2_0   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q4_0:    نتیجہ = quantize_q4_0   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q4_1:    نتیجہ = quantize_q4_1   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q5_0:    نتیجہ = quantize_q5_0   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q5_1:    نتیجہ = quantize_q5_1   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q8_0:    نتیجہ = quantize_q8_0   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_MXFP4:   نتیجہ = quantize_mxfp4  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_NVFP4:   نتیجہ = quantize_nvfp4  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q2_K:    نتیجہ = quantize_q2_K   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q3_K:    نتیجہ = quantize_q3_K   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q4_K:    نتیجہ = quantize_q4_K   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q5_K:    نتیجہ = quantize_q5_K   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_Q6_K:    نتیجہ = quantize_q6_K   (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_TQ1_0:   نتیجہ = quantize_tq1_0  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_TQ2_0:   نتیجہ = quantize_tq2_0  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ2_XXS: نتیجہ = quantize_iq2_xxs(src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ2_XS:  نتیجہ = quantize_iq2_xs (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ3_XXS: نتیجہ = quantize_iq3_xxs(src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ3_S:   نتیجہ = quantize_iq3_s  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ2_S:   نتیجہ = quantize_iq2_s  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ1_S:   نتیجہ = quantize_iq1_s  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ1_M:   نتیجہ = quantize_iq1_m  (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ4_NL:  نتیجہ = quantize_iq4_nl (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_IQ4_XS:  نتیجہ = quantize_iq4_xs (src + آغاز, (حرف *) dst + آغاز_صف * صف_سائز, nrows, n_فی_صف, imatrix); توڑو;
        حالت GGML_TYPE_F16:
            {
                size_t elemsize = حجم(ggml_fp16_t);
                ggml_fp32_مقصد_fp16_صف(src + آغاز, (ggml_fp16_t *)dst + آغاز, n);
                نتیجہ = n * elemsize;
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                size_t elemsize = حجم(ggml_bf16_t);
                ggml_fp32_مقصد_bf16_صف_ref(src + آغاز, (ggml_bf16_t *)dst + آغاز, n);
                نتیجہ = n * elemsize;
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                size_t elemsize = حجم(اعشاری);
                نتیجہ = n * elemsize;
                حافظہ_نقل((uint8_t *)dst + آغاز * elemsize, src + آغاز, نتیجہ);
            } توڑو;
        طےشدہ:
            أكد(false);
    }

    GGML_ASSERT(نتیجہ == nrows * صف_سائز);

    واپس نتیجہ;
}

////////////////////////////////////////////////////////////////////////////////

خالی ggml_log_get(ggml_log_callback * log_callback, خالی ** user_ڈیٹا) {
    *log_callback = g_logger_state.log_callback;
    *user_ڈیٹا    = g_logger_state.log_callback_user_ڈیٹا;
}

خالی ggml_log_سیٹ(ggml_log_callback log_callback, خالی * user_ڈیٹا) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_ڈیٹا = user_ڈیٹا;
}

خالی ggml_threadpool_params_init(ساخت ggml_threadpool_params * p, صحیح n_دھاگے) {
    p->n_دھاگے  = n_دھاگے;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_دھاگے); // all-zero means use the default affinity (usually inherited)
}

ساخت ggml_threadpool_params ggml_threadpool_params_default(صحیح n_دھاگے) {
    ساخت ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_دھاگے);
    واپس p;
}

bool ggml_threadpool_params_مطابقت(ثابت ساخت ggml_threadpool_params * p0, ثابت ساخت ggml_threadpool_params * p1) {
    اگر (p0->n_دھاگے  != p1->n_دھاگے  ) واپس false;
    اگر (p0->prio       != p1->prio       ) واپس false;
    اگر (p0->poll       != p1->poll       ) واپس false;
    اگر (p0->strict_cpu != p1->strict_cpu ) واپس false;
    واپس memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_دھاگے) == 0;
}
