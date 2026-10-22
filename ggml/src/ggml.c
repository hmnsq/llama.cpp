#تعریف_کرو _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#تعریف_کرو USE_MATH_مقرر_کردا // For M_PI on MSVC

#ضمن "ggml-version.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-threading.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml.h"

// FIXME: required here for quantization functions
#ضمن "ggml-quants.h"

#جے_تعریف_شدہ GGML_USE_CPU_HBM
#ضمن <hbwmalloc.h>
#جے_انت

#إذا defined(_MSC_VER) || defined(__MINGW32__)
#ضمن <malloc.h> // using malloc.h with MSC/MINGW
#وإلاإذا !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ضمن <alloca.h>
#جے_انت

#ضمن <assert.h>
#ضمن <errno.h>
#ضمن <time.h>
#ضمن <math.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#ضمن <stdint.h>
#ضمن <inttypes.h>
#ضمن <stdio.h>
#ضمن <float.h>
#ضمن <limits.h>
#ضمن <stdarg.h>
#ضمن <signal.h>
#إذا defined(__gnu_linux__)
#ضمن <syscall.h>
#جے_انت

#إذا defined(__APPLE__)
#ضمن <unistd.h>
#ضمن <mach/mach.h>
#ضمن <TargetConditionals.h>
#جے_انت

#إذا defined(_WIN32)
#تعریف_کرو WIN32_LEAN_AND_اوسط
#جے_بنا_تعریف NOMINMAX
    #تعریف_کرو NOMINMAX
#جے_انت
#ضمن <windows.h>
#جے_انت

#تعریف_کرو بن_ورتیا GGML_بن_ورتیا

uint64_t ggml_گراف_اگلا_uid(فراغ) {
#جے_تعریف_شدہ _MSC_VER
#إذا defined(_WIN32)
    ساكن متطاير LONG counter = 1;
    أرجع (uint64_t) InterlockedIncrement(&counter) - 1;
#وإلا
    ساكن متطاير لما لما counter = 1;
    أرجع (uint64_t) _InterlockedIncrement64(&counter) - 1;
#جے_انت
#وإلا
    ساكن uint64_t counter = 1;
    أرجع __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#جے_انت
}

// Needed for ggml_fp32_to_bf16_row()
#إذا defined(__AVX512BF16__)
#إذا defined(_MSC_VER)
#تعریف_کرو m512i(p) p
#وإلا
#ضمن <immintrin.h>
#تعریف_کرو m512i(p) (__m512i)(p)
#جے_انت // defined(_MSC_VER)
#جے_انت // defined(__AVX512BF16__)

#إذا defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !نشانہ_OS_TV && !نشانہ_OS_WATCH)

#ضمن <unistd.h>
#ضمن <sys/types.h>
#ضمن <sys/stat.h>
#ضمن <sys/wait.h>
#إذا defined(__linux__)
#ضمن <sys/prctl.h>
#جے_انت

#إذا defined(__ANDROID__)
#ضمن <unwind.h>
#ضمن <dlfcn.h>
#ضمن <stdio.h>

بنية backtrace_state {
    فراغ ** current;
    فراغ ** سرا;
};

ساكن Unwind_Reason_کوڈ unwind_callback(بنية _Unwind_Context* context, فراغ* arg) {
    بنية backtrace_state * state = (بنية backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    إذا (pc) {
        إذا (state->current == state->سرا) {
            أرجع URC_سرا_دا_ڈھیر;
        } وإلا {
            *state->current++ = (فراغ*)pc;
        }
    }
    أرجع _URC_NO_REASON;
}

ساكن فراغ ggml_print_backtrace_symbols(فراغ) {
    ثابت صحيح max = 100;
    فراغ* بفر[max];

    بنية backtrace_state state = {بفر, بفر + max};
    _Unwind_Backtrace(unwind_callback, &state);

    صحيح گنتی = state.current - بفر;

    لكل (صحيح idx = 0; idx < گنتی; ++idx) {
        ثابت فراغ * addr = بفر[idx];
        ثابت حرف * علامت = "";

        Dl_info info;
        إذا (dladdr(addr, &info) && info.dli_sname) {
            علامت = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, علامت);
    }
}
#وإلاإذا defined(__linux__) && defined(__GLIBC__)
#ضمن <execinfo.h>
ساكن فراغ ggml_print_backtrace_symbols(فراغ) {
    فراغ * trace[100];
    صحيح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وإلاإذا defined(__APPLE__)
#ضمن <execinfo.h>
ساكن فراغ ggml_print_backtrace_symbols(فراغ) {
    فراغ * trace[100];
    صحيح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وإلا
ساكن فراغ ggml_print_backtrace_symbols(فراغ) {
    // platform not supported
}
#جے_انت

فراغ ggml_print_backtrace(فراغ) {
    ثابت حرف * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    إذا (GGML_NO_BACKTRACE) {
        أرجع;
    }
#إذا defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ثابت حرف * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    إذا (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        أرجع;
    }
#جے_انت
#إذا defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t ناپ = 0;
    حرف * سطر = عدم;
    ssize_t لمائی = 0;
    طالما ((لمائی = getline(&سطر, &ناپ, f)) > 0) {
        إذا (!لڑی_مقابلہ(سطر, "TracerPid:", حجم("TracerPid:") - 1) &&
            (لمائی != حجم("TracerPid:\t0\n") - 1 || سطر[لمائی - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            حرر(سطر);
            fclose(f);
            أرجع;
        }
    }
    حرر(سطر);
    fclose(f);
    صحيح جندرا[2] = { -1, -1 };
    (فراغ) !pipe(جندرا); // Don't start gdb until after PR_SET_PTRACER
#جے_انت
    ثابت صحيح ماپا_pid = getpid();
    ثابت صحيح بچہ_pid = fork();
    إذا (بچہ_pid < 0) { // error
#إذا defined(__linux__)
        close(جندرا[1]);
        close(جندرا[0]);
#جے_انت
        أرجع;
    } وإلا إذا (بچہ_pid == 0) { //بچہ
        حرف attach[32];
        snprintf(attach, حجم(attach), "attach %d", ماپا_pid);
#إذا defined(__linux__)
        close(جندرا[1]);
        (فراغ) !اقرأ(جندرا[0], جندرا, 1);
        close(جندرا[0]);
#جے_انت
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (حرف *) عدم);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[حجم("attach ") - 1],
            (حرف *) عدم);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } وإلا { //ماپا
#إذا defined(__linux__)
        prctl(PR_سیٹ_PTRACER, بچہ_pid);
        close(جندرا[1]);
        close(جندرا[0]);
#جے_انت
        waitpid(بچہ_pid, عدم, 0);
    }
}
#وإلا
فراغ ggml_print_backtrace(فراغ) {
    // platform not supported
}
#جے_انت

ساكن ggml_abort_callback_t g_abort_callback = عدم;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_سیٹ_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    أرجع ret_val;
}

فراغ ggml_abort(ثابت حرف * file, صحيح سطر, ثابت حرف * fmt, ...) {
    fflush(stdout);

    حرف سنیہا[2048];
    صحيح وتھ = snprintf(سنیہا, حجم(سنیہا), "%s:%d: ", file, سطر);

    va_list دلیلاں_2;
    va_start(دلیلاں_2, fmt);
    vsnprintf(سنیہا + وتھ, حجم(سنیہا) - وتھ, fmt, دلیلاں_2);
    va_end(دلیلاں_2);

    إذا (g_abort_callback) {
        g_abort_callback(سنیہا);
    } وإلا {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", سنیہا);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

بنية ggml_logger_state {
    ggml_log_callback log_callback;
    فراغ * log_callback_user_انکڑے;
};
ساكن بنية ggml_logger_state g_logger_state = {ggml_log_callback_default, عدم};

ساكن فراغ ggml_log_اندرونی_v(تعداد ggml_log_level level, ثابت حرف * format, va_list دلیلاں_2) {
    إذا (format == عدم) {
        أرجع;
    }
    va_list دلیلاں_copy;
    va_copy(دلیلاں_copy, دلیلاں_2);
    حرف بفر[128];
    صحيح len = vsnprintf(بفر, 128, format, دلیلاں_2);
    إذا (len < 128) {
        g_logger_state.log_callback(level, بفر, g_logger_state.log_callback_user_انکڑے);
    } وإلا {
        حرف * buffer2 = (حرف *) صفر_حافظہ_دیو(len + 1, حجم(حرف));
        vsnprintf(buffer2, len + 1, format, دلیلاں_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_انکڑے);
        حرر(buffer2);
    }
    va_end(دلیلاں_copy);
}

فراغ ggml_log_اندرونی(تعداد ggml_log_level level, ثابت حرف * format, ...) {
    va_list دلیلاں_2;
    va_start(دلیلاں_2, format);
    ggml_log_اندرونی_v(level, format, دلیلاں_2);
    va_end(دلیلاں_2);
}

فراغ ggml_log_callback_default(تعداد ggml_log_level level, ثابت حرف * متن_2, فراغ * user_انکڑے) {
    (فراغ) level;
    (فراغ) user_انکڑے;
    fputs(متن_2, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#جے_تعریف_شدہ GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#جے_انت


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#إذا defined(__has_feature)
#إذا __has_feature(thread_sanitizer)
#تعریف_کرو GGML_TSAN_ENABLED 1
#جے_انت
#وإلا
#إذا defined(__SANITIZE_THREAD__)
#تعریف_کرو GGML_TSAN_ENABLED 1
#جے_انت
#جے_انت

فراغ * ggml_aligned_malloc(size_t ناپ) {
#إذا defined(__s390x__)
    ثابت صحيح alignment = 256;
#وإلا
    ثابت صحيح alignment = 64;
#جے_انت

#إذا defined(_MSC_VER) || defined(__MINGW32__)
    أرجع _aligned_malloc(ناپ, alignment);
#وإلا
    إذا (ناپ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        أرجع عدم;
    }
    فراغ * aligned_حافظہ = عدم;
  #جے_تعریف_شدہ GGML_USE_CPU_HBM
    صحيح نتیجہ = hbw_posix_memalign(&aligned_حافظہ, alignment, ناپ);
  #وإلاإذا نشانہ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_بن_ورتیا(alignment);
    kern_return_t alloc_حالت = vm_allocate((vm_map_t) mach_task_self(), (vm_پتہ_t *) &aligned_حافظہ, ناپ, VM_FLAGS_ANYWHERE);
    صحيح نتیجہ = EFAULT;
    بدل (alloc_حالت) {
        حالة KERN_SUCCESS:
            نتیجہ = 0;
            اكسر;
        حالة KERN_INVALID_پتہ:
            نتیجہ = EINVAL;
            اكسر;
        حالة KERN_NO_SPACE:
            نتیجہ = ENOMEM;
            اكسر;
        افتراضي:
            نتیجہ = EFAULT;
            اكسر;
    }
  #وإلا
    صحيح نتیجہ = posix_memalign(&aligned_حافظہ, alignment, ناپ);
  #جے_انت
    إذا (نتیجہ != 0) {
        // Handle allocation failure
        ثابت حرف *error_desc = "unknown allocation error";
        بدل (نتیجہ) {
            حالة EINVAL:
                error_desc = "invalid alignment value";
                اكسر;
            حالة ENOMEM:
                error_desc = "insufficient memory";
                اكسر;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, ناپ/(1024.0*1024.0));
        أرجع عدم;
    }
    أرجع aligned_حافظہ;
#جے_انت
}

فراغ ggml_aligned_free(فراغ * ptr, size_t ناپ) {
    GGML_بن_ورتیا(ناپ);
#إذا defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#وإلاإذا GGML_USE_CPU_HBM
    إذا (ptr != عدم) {
        hbw_free(ptr);
    }
#وإلاإذا نشانہ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    إذا (ptr != عدم) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_پتہ_t)ptr, ناپ);
    }
#وإلا
    حرر(ptr);
#جے_انت
}


مضمن ساكن فراغ * ggml_malloc(size_t ناپ) {
    إذا (ناپ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        أرجع عدم;
    }
    فراغ * نتیجہ = حافظہ_دیو(ناپ);
    إذا (نتیجہ == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ناپ/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    أرجع نتیجہ;
}

// calloc
مضمن ساكن فراغ * ggml_calloc(size_t عدد, size_t ناپ) {
    إذا (عدد == 0 || ناپ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        أرجع عدم;
    }
    فراغ * نتیجہ = صفر_حافظہ_دیو(عدد, ناپ);
    إذا (نتیجہ == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ناپ/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    أرجع نتیجہ;
}

#تعریف_کرو GGML_MALLOC(ناپ)      ggml_malloc(ناپ)
#تعریف_کرو GGML_CALLOC(عدد, ناپ) ggml_calloc(عدد, ناپ)

#تعریف_کرو GGML_FREE(ptr) حرر(ptr)

ثابت حرف * ggml_حالت_منزل_string(تعداد ggml_حالت حالت) {
    بدل (حالت) {
        حالة GGML_حالت_ALLOC_FAILED: أرجع "GGML status: error (failed to allocate memory)";
        حالة GGML_حالت_FAILED:       أرجع "GGML status: error (operation failed)";
        حالة GGML_حالت_SUCCESS:      أرجع "GGML status: success";
        حالة GGML_حالت_ABORTED:      أرجع "GGML status: warning (operation aborted)";
    }

    أرجع "GGML status: unknown";
}

عائم ggml_fp16_منزل_fp32(ggml_fp16_t x) {
#تعریف_کرو ggml_fp16_منزل_fp32 do_not_use_ggml_fp16_منزل_fp32_وچ_ggml
    أرجع GGML_FP16_منزل_FP32(x);
}

ggml_fp16_t ggml_fp32_منزل_fp16(عائم x) {
#تعریف_کرو ggml_fp32_منزل_fp16 do_not_use_ggml_fp32_منزل_fp16_وچ_ggml
    أرجع GGML_FP32_منزل_FP16(x);
}

عائم ggml_bf16_منزل_fp32(ggml_bf16_t x) {
#تعریف_کرو ggml_bf16_منزل_fp32 do_not_use_ggml_bf16_منزل_fp32_وچ_ggml
    أرجع GGML_BF16_منزل_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_منزل_bf16(عائم x) {
#تعریف_کرو ggml_fp32_منزل_bf16 do_not_use_ggml_fp32_منزل_bf16_وچ_ggml
    أرجع GGML_FP32_منزل_BF16(x);
}

فراغ ggml_fp16_منزل_fp32_صف(ثابت ggml_fp16_t * x, عائم * y, int64_t n) {
    لكل (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_منزل_FP32(x[i]);
    }
}

فراغ ggml_fp32_منزل_fp16_صف(ثابت عائم * x, ggml_fp16_t * y, int64_t n) {
    صحيح i = 0;
    لكل (; i < n; ++i) {
        y[i] = GGML_FP32_منزل_FP16(x[i]);
    }
}

فراغ ggml_bf16_منزل_fp32_صف(ثابت ggml_bf16_t * x, عائم * y, int64_t n) {
    صحيح i = 0;
    لكل (; i < n; ++i) {
        y[i] = GGML_BF16_منزل_FP32(x[i]);
    }
}

فراغ ggml_fp32_منزل_bf16_صف_ref(ثابت عائم * x, ggml_bf16_t * y, int64_t n) {
    لكل (صحيح i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_منزل_bf16(x[i]);
    }
}

فراغ ggml_fp32_منزل_bf16_صف(ثابت عائم * x, ggml_bf16_t * y, int64_t n) {
  صحيح i = 0;
#إذا defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  لكل (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#جے_انت
    لكل (; i < n; i++) {
        y[i] = GGML_FP32_منزل_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    أرجع memcmp(guid_a, guid_b, حجم(ggml_guid)) == 0;
}

ثابت حرف * ggml_نسخہ(فراغ) {
    أرجع GGML_نسخہ;
}

ثابت حرف * ggml_commit(فراغ) {
    أرجع GGML_COMMIT;
}

//
// timing
//

#إذا defined(_MSC_VER) || defined(__MINGW32__)
ساكن int64_t timer_freq, timer_شروع;
ساكن BOOL CALLBACK ggml_ویلا_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    بن_ورتیا(once);
    بن_ورتیا(param);
    بن_ورتیا(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_شروع = t.QuadPart;

    أرجع TRUE;
}
فراغ ggml_ویلا_init(فراغ) {
    ساكن INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_ویلا_init_once, عدم, عدم);
}
int64_t ggml_ویلا_ms(فراغ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    أرجع ((t.QuadPart-timer_شروع) * 1000) / timer_freq;
}
int64_t ggml_ویلا_us(فراغ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    أرجع ((t.QuadPart-timer_شروع) * 1000000) / timer_freq;
}
#وإلا
فراغ ggml_ویلا_init(فراغ) {}
int64_t ggml_ویلا_ms(فراغ) {
    بنية timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    أرجع (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_ویلا_us(فراغ) {
    بنية timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    أرجع (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#جے_انت

int64_t ggml_چکراں(فراغ) {
    أرجع clock();
}

int64_t ggml_چکراں_فی_ms(فراغ) {
    أرجع CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#جے_تعریف_شدہ _WIN32
ساكن wchar_t * ggml_mbstowcs(ثابت حرف * mbs) {
    صحيح wlen = Multi_Byte_منزل_Wide_Char(CP_UTF8, 0, mbs, -1, عدم, 0);
    إذا (!wlen) {
        errno = EINVAL;
        أرجع عدم;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * حجم(wchar_t));
    wlen = Multi_Byte_منزل_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    إذا (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        أرجع عدم;
    }

    أرجع wbuf;
}
#جے_انت

FILE * ggml_fopen(ثابت حرف * fname, ثابت حرف * منوال) {
#جے_تعریف_شدہ _WIN32
    FILE * file = عدم;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    إذا (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(منوال);
        إذا (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    أرجع file;
#وإلا
    أرجع fopen(fname, منوال);
#جے_انت

}

ساكن ثابت بنية ggml_type_traits type_traits[GGML_TYPE_گنتی] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(مزدوج),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(عائم),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(ggml_fp16_t),
        .is_quantized             = false,
        .منزل_float                 = (ggml_منزل_float_t) ggml_fp16_منزل_fp32_صف,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_منزل_fp16_صف,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_ناپ                = QK1_0,
        .type_ناپ                = حجم(بلاک_q1_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_ناپ                = QK2_0,
        .type_ناپ                = حجم(بلاک_q2_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_ناپ                = QK4_0,
        .type_ناپ                = حجم(بلاک_q4_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_ناپ                = QK4_1,
        .type_ناپ                = حجم(بلاک_q4_1),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_ناپ                = QK5_0,
        .type_ناپ                = حجم(بلاک_q5_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_ناپ                = QK5_1,
        .type_ناپ                = حجم(بلاک_q5_1),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_ناپ                = QK8_0,
        .type_ناپ                = حجم(بلاک_q8_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_ناپ                = QK8_1,
        .type_ناپ                = حجم(بلاک_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_ناپ                = QK_MXFP4,
        .type_ناپ                = حجم(بلاک_mxfp4),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_ناپ                = QK_NVFP4,
        .type_ناپ                = حجم(بلاک_nvfp4),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q2_K),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q3_K),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q4_K),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q5_K),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q6_K),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq2_xxs),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq2_xxs,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq2_xs),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq2_xs,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq3_xxs),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq3_s),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq2_s),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq1_s),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq1_s,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq1_m),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq1_m,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_ناپ                = QK4_NL,
        .type_ناپ                = حجم(بلاک_iq4_nl),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_iq4_xs),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_صف_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_ناپ                = 1,
        .type_ناپ                = حجم(ggml_bf16_t),
        .is_quantized             = false,
        .منزل_float                 = (ggml_منزل_float_t) ggml_bf16_منزل_fp32_صف,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_منزل_bf16_صف_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_tq1_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_ناپ                = QK_K,
        .type_ناپ                = حجم(بلاک_tq2_0),
        .is_quantized             = true,
        .منزل_float                 = (ggml_منزل_float_t) dequantize_صف_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_صف_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ناپ                = 0,
        .type_ناپ                = 0,
        .is_quantized             = false,
    },
};

ثابت بنية ggml_type_traits * ggml_get_type_traits(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع &type_traits[type];
}

//
// ggml object
//

بنية ggml_object {
    size_t offs;
    size_t ناپ;

    بنية ggml_object * اگلا;

    تعداد ggml_object_type type;

    حرف padding[4];
};

ساكن ثابت size_t GGML_OBJECT_ناپ = حجم(بنية ggml_object);

//
// ggml context
//

بنية ggml_context {
    size_t mem_ناپ;
    فراغ * mem_بفر;
    bool   mem_بفر_owned;
    bool   no_alloc;

    صحيح    n_objects;

    بنية ggml_object * objects_begin;
    بنية ggml_object * objects_سرا;
};

//
// data types
//

ساكن ثابت حرف * GGML_OP_NAME[GGML_OP_گنتی] = {
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

ساكن ثابت حرف * GGML_OP_علامت[GGML_OP_گنتی] = {
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

ساكن ثابت حرف * GGML_UNARY_OP_NAME[GGML_UNARY_OP_گنتی] = {
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

ساكن ثابت حرف * GGML_GLU_OP_NAME[GGML_GLU_OP_گنتی] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_گنتی == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(حجم(بنية ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(حجم(بنية ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

فراغ ggml_print_object(ثابت بنية ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->ناپ, (ثابت فراغ *) obj->اگلا);
}

فراغ ggml_print_objects(ثابت بنية ggml_context * ctx) {
    بنية ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ثابت فراغ *) ctx);

    طالما (obj != عدم) {
        ggml_print_object(obj);
        obj = obj->اگلا;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        إذا (tensor->ne[i] <= 0) {
            أرجع 0;
        }
    }

    size_t nbytes;
    ثابت size_t blck_ناپ = ggml_blck_ناپ(tensor->type);
    إذا (blck_ناپ == 1) {
        nbytes = ggml_type_ناپ(tensor->type);
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    وإلا {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_ناپ;
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    أرجع nbytes;
}

size_t ggml_nbytes_pad(ثابت بنية ggml_tensor * tensor) {
    أرجع GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_ناپ(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع type_traits[type].blck_ناپ;
}

size_t ggml_type_ناپ(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع type_traits[type].type_ناپ;
}

size_t ggml_صف_ناپ(تعداد ggml_type type, int64_t ne) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أكد(ne % ggml_blck_ناپ(type) == 0);
    أرجع ggml_type_ناپ(type)*ne/ggml_blck_ناپ(type);
}

مزدوج ggml_type_sizef(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع ((مزدوج)(type_traits[type].type_ناپ))/type_traits[type].blck_ناپ;
}

ثابت حرف * ggml_type_name(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع type_traits[type].type_name;
}

bool ggml_is_quantized(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_گنتی);
    أرجع type_traits[type].is_quantized;
}

ثابت حرف * ggml_op_name(تعداد ggml_op op) {
    أرجع GGML_OP_NAME[op];
}

ثابت حرف * ggml_op_علامت(تعداد ggml_op op) {
    أرجع GGML_OP_علامت[op];
}

ثابت حرف * ggml_unary_op_name(تعداد ggml_unary_op op) {
    أرجع GGML_UNARY_OP_NAME[op];
}

ثابت حرف * ggml_glu_op_name(تعداد ggml_glu_op op) {
    أرجع GGML_GLU_OP_NAME[op];
}

ثابت حرف * ggml_op_desc(ثابت بنية ggml_tensor * t) {
    إذا (t->op == GGML_OP_UNARY) {
        تعداد ggml_unary_op uop = ggml_get_unary_op(t);
        أرجع ggml_unary_op_name(uop);
    }
    إذا (t->op == GGML_OP_GLU) {
        تعداد ggml_glu_op gop = ggml_get_glu_op(t);
        أرجع ggml_glu_op_name(gop);
    }
    أرجع ggml_op_name(t->op);
}

size_t ggml_element_ناپ(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_type_ناپ(tensor->type);
}

bool ggml_is_scalar(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_سمتیہ(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_میٹرکس(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->ne[3] == 1;
}

صحيح ggml_n_dims(ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        إذا (tensor->ne[i] > 1) {
            أرجع i + 1;
        }
    }
    أرجع 1;
}

تعداد ggml_type ggml_ftype_منزل_ggml_type(تعداد ggml_ftype ftype) {
    تعداد ggml_type wtype = GGML_TYPE_گنتی;

    بدل (ftype) {
        حالة GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   اكسر;
        حالة GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   اكسر;
        حالة GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  اكسر;
        حالة GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; اكسر;
        حالة GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; اكسر;
        حالة GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  اكسر;
        حالة GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    اكسر;
        حالة GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    اكسر;
        حالة GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_گنتی; اكسر;
        حالة GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_گنتی; اكسر;
    }

    GGML_ASSERT(wtype != GGML_TYPE_گنتی);

    أرجع wtype;
}

size_t ggml_tensor_overhead(فراغ) {
    أرجع GGML_OBJECT_ناپ + GGML_TENSOR_ناپ;
}

bool ggml_is_transposed(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->nb[0] > tensor->nb[1];
}

ساكن bool ggml_is_contiguous_m_n(ثابت بنية ggml_tensor * tensor, صحيح m, صحيح n) {
    size_t اگلا_nb = ggml_type_ناپ(tensor->type);
    إذا (tensor->ne[0] != ggml_blck_ناپ(tensor->type) && tensor->nb[0] != اگلا_nb) {
        أرجع false;
    }
    اگلا_nb *= tensor->ne[0]/ggml_blck_ناپ(tensor->type);
    لكل (صحيح i = 1; i < n; i++) {
        إذا (i > m) {
            إذا (tensor->ne[i] != 1 && tensor->nb[i] != اگلا_nb) {
                أرجع false;
            }
            اگلا_nb *= tensor->ne[i];
        } وإلا {
            // this dimension does not need to be contiguous
            اگلا_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    أرجع true;
}

bool ggml_is_contiguous(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_منزل_1(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_منزل_2(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_منزل_3(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_ناپ(tensor->type)/ggml_blck_ناپ(tensor->type);
}

bool ggml_is_permuted(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ثابت بنية ggml_tensor * tensor) {
    أرجع
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_ناپ(tensor->type);
}

bool ggml_is_contiguous_صفاں(ثابت بنية ggml_tensor * tensor) {
    أرجع
        tensor->ne[0] == ggml_blck_ناپ(tensor->type) ||
        tensor->nb[0] == ggml_type_ناپ(tensor->type);
}

ساكن مضمن bool ggml_is_padded_1d(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع
        tensor->nb[0] == ggml_type_ناپ(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_خالی_تھاواں(ثابت بنية ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        إذا (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            أرجع true;
        }
    }
    أرجع false;
}

bool ggml_are_same_shape(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ثابت بنية ggml_tensor * t) {
    أرجع ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع ggml_is_خالی_تھاواں(t0) ? ggml_is_خالی_تھاواں(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ساكن مضمن bool ggml_can_repeat_صفاں(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#تعریف_کرو GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

بنية ggml_context * ggml_init(بنية ggml_init_params params) {
    ساكن bool is_پہلا_call = true;

    ggml_critical_section_شروع();

    إذا (is_پہلا_call) {
        // initialize time system (required on Windows)
        ggml_ویلا_init();

        is_پہلا_call = false;
    }

    ggml_critical_section_سرا();

    بنية ggml_context * ctx = GGML_MALLOC(حجم(بنية ggml_context));

    // allow to call ggml_init with 0 size
    إذا (params.mem_ناپ == 0) {
        params.mem_ناپ = GGML_MEM_ALIGN;
    }

    ثابت size_t mem_ناپ = params.mem_بفر ? params.mem_ناپ : GGML_PAD(params.mem_ناپ, GGML_MEM_ALIGN);

    *ctx = (بنية ggml_context) {
        /*.mem_size           =*/ mem_ناپ,
        /*.mem_buffer         =*/ params.mem_بفر ? params.mem_بفر : ggml_aligned_malloc(mem_ناپ),
        /*.mem_buffer_owned   =*/ params.mem_بفر ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ عدم,
        /*.objects_end        =*/ عدم,
    };

    GGML_ASSERT(ctx->mem_بفر != عدم);

    GGML_ASSERT_ALIGNED(ctx->mem_بفر);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    أرجع ctx;
}

فراغ ggml_reset(بنية ggml_context * ctx) {
    إذا (ctx == عدم) {
        أرجع;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = عدم;
    ctx->objects_سرا   = عدم;
}

فراغ ggml_free(بنية ggml_context * ctx) {
    إذا (ctx == عدم) {
        أرجع;
    }

    إذا (ctx->mem_بفر_owned) {
        ggml_aligned_free(ctx->mem_بفر, ctx->mem_ناپ);
    }

    GGML_FREE(ctx);
}

size_t ggml_ورتیا_mem(ثابت بنية ggml_context * ctx) {
    أرجع ctx->objects_سرا == عدم ? 0 : ctx->objects_سرا->offs + ctx->objects_سرا->ناپ;
}

bool ggml_get_no_alloc(بنية ggml_context * ctx) {
    أرجع ctx->no_alloc;
}

فراغ ggml_سیٹ_no_alloc(بنية ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

فراغ * ggml_get_mem_بفر(ثابت بنية ggml_context * ctx) {
    أرجع ctx->mem_بفر;
}

size_t ggml_get_mem_ناپ(ثابت بنية ggml_context * ctx) {
    أرجع ctx->mem_ناپ;
}

size_t ggml_get_max_tensor_ناپ(ثابت بنية ggml_context * ctx) {
    size_t max_ناپ = 0;

    لكل (بنية ggml_tensor * tensor = ggml_get_پہلا_tensor(ctx); tensor != عدم; tensor = ggml_get_اگلا_tensor(ctx, tensor)) {
        size_t بائٹاں = ggml_nbytes(tensor);
        max_ناپ = MAX(max_ناپ, بائٹاں);
    }

    أرجع max_ناپ;
}

////////////////////////////////////////////////////////////////////////////////

ساكن بنية ggml_object * ggml_new_object(بنية ggml_context * ctx, تعداد ggml_object_type type, size_t ناپ) {
    // always insert objects at the end of the context's memory pool
    بنية ggml_object * obj_cur = ctx->objects_سرا;

    ثابت size_t cur_offs = obj_cur == عدم ? 0 : obj_cur->offs;
    ثابت size_t cur_ناپ = obj_cur == عدم ? 0 : obj_cur->ناپ;
    ثابت size_t cur_سرا  = cur_offs + cur_ناپ;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(ناپ <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t ناپ_needed = GGML_PAD(ناپ, GGML_MEM_ALIGN);

    حرف * ثابت mem_بفر = ctx->mem_بفر;
    بنية ggml_object * ثابت obj_new = (بنية ggml_object *)(mem_بفر + cur_سرا);

    // integer overflow checks
    إذا (cur_سرا > SIZE_MAX - ناپ_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_سرا, ناپ_needed);
        أرجع عدم;
    }
    إذا (cur_سرا + ناپ_needed > SIZE_MAX - GGML_OBJECT_ناپ) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_سرا, ناپ_needed, (size_t) GGML_OBJECT_ناپ);
        أرجع عدم;
    }

    إذا (cur_سرا + ناپ_needed + GGML_OBJECT_ناپ > ctx->mem_ناپ) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_سرا + ناپ_needed + GGML_OBJECT_ناپ, ctx->mem_ناپ);
#جے_بنا_تعریف NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#جے_انت
        أرجع عدم;
    }

    *obj_new = (بنية ggml_object) {
        .offs = cur_سرا + GGML_OBJECT_ناپ,
        .ناپ = ناپ_needed,
        .اگلا = عدم,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_بفر + obj_new->offs);

    إذا (obj_cur != عدم) {
        obj_cur->اگلا = obj_new;
    } وإلا {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_سرا = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    أرجع obj_new;
}

ساكن بنية ggml_tensor * ggml_new_tensor_impl(
        بنية ggml_context * ctx,
        تعداد   ggml_type      type,
        صحيح                   n_dims,
        ثابت int64_t       * ne,
        بنية ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_گنتی);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    إذا (view_src != عدم && view_src->view_src != عدم) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    لكل (صحيح i = 1; i < n_dims; i++) {
        إذا (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t انکڑے_ناپ = ggml_صف_ناپ(type, ne[0]);
    لكل (صحيح i = 1; i < n_dims; i++) {
        انکڑے_ناپ *= ne[i];
    }

    GGML_ASSERT(view_src == عدم || انکڑے_ناپ == 0 || انکڑے_ناپ + view_offs <= ggml_nbytes(view_src));

    فراغ * انکڑے = view_src != عدم ? view_src->انکڑے : عدم;
    إذا (انکڑے != عدم) {
        انکڑے = (حرف *) انکڑے + view_offs;
    }

    size_t obj_alloc_ناپ = 0;

    إذا (view_src == عدم && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_ناپ = انکڑے_ناپ;
    }

    GGML_ASSERT(GGML_TENSOR_ناپ <= SIZE_MAX - obj_alloc_ناپ);

    بنية ggml_object * ثابت obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_ناپ + obj_alloc_ناپ);
    GGML_ASSERT(obj_new);

    بنية ggml_tensor * ثابت نتیجہ = (بنية ggml_tensor *)((حرف *)ctx->mem_بفر + obj_new->offs);

    *نتیجہ = (بنية ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ عدم,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { عدم },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_ناپ > 0 ? (فراغ *)(نتیجہ + 1) : انکڑے,
        /*.name         =*/ { 0 },
        /*.extra        =*/ عدم,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    لكل (صحيح i = 0; i < n_dims; i++) {
        نتیجہ->ne[i] = ne[i];
    }

    نتیجہ->nb[0] = ggml_type_ناپ(type);
    نتیجہ->nb[1] = نتیجہ->nb[0]*(نتیجہ->ne[0]/ggml_blck_ناپ(type));
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = نتیجہ->nb[i - 1]*نتیجہ->ne[i - 1];
    }

    ctx->n_objects++;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_new_tensor(
        بنية ggml_context * ctx,
        تعداد   ggml_type      type,
        صحيح                   n_dims,
        ثابت int64_t       * ne) {
    أرجع ggml_new_tensor_impl(ctx, type, n_dims, ne, عدم, 0);
}

بنية ggml_tensor * ggml_new_tensor_1d(
        بنية ggml_context * ctx,
        تعداد   ggml_type      type,
        int64_t ne0) {
    أرجع ggml_new_tensor(ctx, type, 1, &ne0);
}

بنية ggml_tensor * ggml_new_tensor_2d(
        بنية ggml_context * ctx,
        تعداد   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ثابت int64_t ne[2] = { ne0, ne1 };
    أرجع ggml_new_tensor(ctx, type, 2, ne);
}

بنية ggml_tensor * ggml_new_tensor_3d(
        بنية ggml_context * ctx,
        تعداد   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    أرجع ggml_new_tensor(ctx, type, 3, ne);
}

بنية ggml_tensor * ggml_new_tensor_4d(
        بنية ggml_context * ctx,
        تعداد   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    أرجع ggml_new_tensor(ctx, type, 4, ne);
}

فراغ * ggml_new_بفر(بنية ggml_context * ctx, size_t nbytes) {
    بنية ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_بفر, nbytes);

    أرجع (uint8_t *)ctx->mem_بفر + obj->offs;
}

بنية ggml_tensor * ggml_dup_tensor(بنية ggml_context * ctx, ثابت بنية ggml_tensor * src) {
    أرجع ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

فراغ ggml_unravel_اشاریہ(ثابت بنية ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ثابت int64_t ne2 = tensor->ne[2];
    ثابت int64_t ne1 = tensor->ne[1];
    ثابت int64_t ne0 = tensor->ne[0];

    ثابت int64_t i3_ = (i/(ne2*ne1*ne0));
    ثابت int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ثابت int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ثابت int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    إذا (i0) {
        * i0 = i0_;
    }
    إذا (i1) {
        * i1 = i1_;
    }
    إذا (i2) {
        * i2 = i2_;
    }
    إذا (i3) {
        * i3 = i3_;
    }
}

فراغ * ggml_get_انکڑے(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->انکڑے;
}

عائم * ggml_get_انکڑے_f32(ثابت بنية ggml_tensor * tensor) {
    أكد(tensor->type == GGML_TYPE_F32);
    أرجع (عائم *)(tensor->انکڑے);
}

تعداد ggml_unary_op ggml_get_unary_op(ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    أرجع (تعداد ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

تعداد ggml_glu_op ggml_get_glu_op(ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    أرجع (تعداد ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ثابت حرف * ggml_get_name(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->name;
}

بنية ggml_tensor * ggml_سیٹ_name(بنية ggml_tensor * tensor, ثابت حرف * name) {
    size_t i;
    لكل (i = 0; i < حجم(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    أرجع tensor;
}

بنية ggml_tensor * ggml_format_name(بنية ggml_tensor * tensor, ثابت حرف * fmt, ...) {
    va_list دلیلاں_2;
    va_start(دلیلاں_2, fmt);
    vsnprintf(tensor->name, حجم(tensor->name), fmt, دلیلاں_2);
    va_end(دلیلاں_2);
    أرجع tensor;
}

بنية ggml_tensor * ggml_view_tensor(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * src) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(نتیجہ, "%s (view)", src->name);

    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = src->nb[i];
    }

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_get_پہلا_tensor(ثابت بنية ggml_context * ctx) {
    بنية ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            أرجع (بنية ggml_tensor *)(mem_بفر + obj->offs);
        }

        obj = obj->اگلا;
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_get_اگلا_tensor(ثابت بنية ggml_context * ctx, بنية ggml_tensor * tensor) {
    بنية ggml_object * obj = (بنية ggml_object *) ((حرف *)tensor - GGML_OBJECT_ناپ);
    obj = obj->اگلا;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            أرجع (بنية ggml_tensor *)(mem_بفر + obj->offs);
        }

        obj = obj->اگلا;
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_get_tensor(بنية ggml_context * ctx, ثابت حرف * name) {
    بنية ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_بفر = ctx->mem_بفر;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            بنية ggml_tensor * cur = (بنية ggml_tensor *)(mem_بفر + obj->offs);
            إذا (strcmp(cur->name, name) == 0) {
                أرجع cur;
            }
        }

        obj = obj->اگلا;
    }

    أرجع عدم;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ساكن بنية ggml_tensor * ggml_dup_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_DUP;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_dup(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_dup_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_dup_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_dup_impl(ctx, a, true);
}

// ggml_add

ساكن بنية ggml_tensor * ggml_جوڑو_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_جوڑو;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_جوڑو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_جوڑو_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_جوڑو_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_جوڑو_impl(ctx, a, b, true);
}

// ggml_add_cast

ساكن بنية ggml_tensor * ggml_جوڑو_cast_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_صفاں(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    نتیجہ->op     = GGML_OP_جوڑو;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_جوڑو_cast(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد   ggml_type      type) {
    أرجع ggml_جوڑو_cast_impl(ctx, a, b, type);
}

بنية ggml_tensor * ggml_جوڑو_id(
            بنية ggml_context * ctx,
            بنية ggml_tensor  * a,
            بنية ggml_tensor  * b,
            بنية ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_جوڑو_ID;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = ids;

    أرجع نتیجہ;
}

// ggml_add1

ساكن بنية ggml_tensor * ggml_add1_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_ADD1;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_add1(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_add1_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_add1_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ساكن بنية ggml_tensor * ggml_acc_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(وتھ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, وتھ, inplace ? 1 : 0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_ACC;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_acc(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ) {
    أرجع ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, وتھ, false);
}

بنية ggml_tensor * ggml_acc_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ) {
    أرجع ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, وتھ, true);
}

// ggml_sub

ساكن بنية ggml_tensor * ggml_sub_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SUB;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_sub(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_sub_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_sub_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ساكن بنية ggml_tensor * ggml_mul_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_MUL;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_mul(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_mul_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_mul_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ساكن بنية ggml_tensor * ggml_div_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_DIV;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_div(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_div_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_div_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ساكن بنية ggml_tensor * ggml_sqr_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SQR;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_sqr(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sqr_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_sqr_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ساكن بنية ggml_tensor * ggml_sqrt_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SQRT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_sqrt(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sqrt_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_sqrt_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ساكن بنية ggml_tensor * ggml_log_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_LOG;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_log(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_log_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_log_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_log_impl(ctx, a, true);
}

بنية ggml_tensor * ggml_expm1(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

بنية ggml_tensor * ggml_expm1_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

بنية ggml_tensor * ggml_softplus(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

بنية ggml_tensor * ggml_softplus_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ساكن بنية ggml_tensor * ggml_sin_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SIN;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_sin(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sin_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_sin_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ساكن بنية ggml_tensor * ggml_cos_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_COS;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_cos(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_cos_impl(ctx, a, false);
}

بنية ggml_tensor * ggml_cos_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_cos_impl(ctx, a, true);
}

// ggml_sum

بنية ggml_tensor * ggml_جوڑ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, a->type, 1);

    نتیجہ->op     = GGML_OP_جوڑ;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_sum_rows

بنية ggml_tensor * ggml_جوڑ_صفاں(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    نتیجہ->op     = GGML_OP_جوڑ_صفاں;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_cumsum

بنية ggml_tensor * ggml_cumsum(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_CUMSUM;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_mean

بنية ggml_tensor * ggml_اوسط(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_اوسط;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_argmax

بنية ggml_tensor * ggml_argmax(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_میٹرکس(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    نتیجہ->op     = GGML_OP_ARGMAX;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_count_equal

بنية ggml_tensor * ggml_گنتی_equal(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    نتیجہ->op     = GGML_OP_گنتی_EQUAL;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_repeat

بنية ggml_tensor * ggml_repeat(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    نتیجہ->op     = GGML_OP_REPEAT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_repeat_4d(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ثابت bool can_repeat = ggml_is_خالی_تھاواں(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    نتیجہ->op     = GGML_OP_REPEAT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_repeat_back

بنية ggml_tensor * ggml_repeat_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    نتیجہ->op     = GGML_OP_REPEAT_پٹھ;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_concat

بنية ggml_tensor * ggml_concat(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    بنية ggml_tensor  * b,
    صحيح                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    لكل (صحيح d = 0; d < GGML_MAX_DIMS; ++d) {
        إذا (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            استمر;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, dim);

    نتیجہ->op     = GGML_OP_CONCAT;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_abs

بنية ggml_tensor * ggml_abs(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

بنية ggml_tensor * ggml_abs_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

بنية ggml_tensor * ggml_sgn(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

بنية ggml_tensor * ggml_sgn_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

بنية ggml_tensor * ggml_neg(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

بنية ggml_tensor * ggml_neg_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

بنية ggml_tensor * ggml_قدم(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_قدم);
}

بنية ggml_tensor * ggml_قدم_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_قدم);
}

// ggml_tanh

بنية ggml_tensor * ggml_tanh(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

بنية ggml_tensor * ggml_tanh_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

بنية ggml_tensor * ggml_elu(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

بنية ggml_tensor * ggml_elu_inplace(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

بنية ggml_tensor * ggml_relu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

بنية ggml_tensor * ggml_relu_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

بنية ggml_tensor * ggml_leaky_relu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 negative_ڈھلوان,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &negative_ڈھلوان, حجم(negative_ڈھلوان));

    نتیجہ->op     = GGML_OP_LEAKY_RELU;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_sigmoid

بنية ggml_tensor * ggml_sigmoid(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

بنية ggml_tensor * ggml_sigmoid_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

بنية ggml_tensor * ggml_gelu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

بنية ggml_tensor * ggml_gelu_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

بنية ggml_tensor * ggml_gelu_erf(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

بنية ggml_tensor * ggml_gelu_erf_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

بنية ggml_tensor * ggml_gelu_تیز(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_GELU_تیز);
}

بنية ggml_tensor * ggml_gelu_تیز_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_تیز);
}

// ggml_silu

بنية ggml_tensor * ggml_silu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

بنية ggml_tensor * ggml_silu_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

بنية ggml_tensor * ggml_xielu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم alpha_n,
        عائم alpha_p,
        عائم beta,
        عائم eps) {
    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_سیٹ_op_params_f32(نتیجہ, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_سیٹ_op_params_f32(نتیجہ, 3, beta);
    ggml_سیٹ_op_params_f32(نتیجہ, 4, eps);

    نتیجہ->op     = GGML_OP_UNARY;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_silu_back

بنية ggml_tensor * ggml_silu_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SILU_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml hardswish

بنية ggml_tensor * ggml_hardswish(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

بنية ggml_tensor * ggml_hardsigmoid(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

بنية ggml_tensor * ggml_exp(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

بنية ggml_tensor * ggml_exp_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ساكن بنية ggml_tensor * ggml_glu_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    إذا (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, عدم, 0);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) op);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, (int32_t) swapped);

    نتیجہ->op     = GGML_OP_GLU;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_floor

بنية ggml_tensor * ggml_floor(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

بنية ggml_tensor * ggml_floor_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

بنية ggml_tensor * ggml_ceil(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

بنية ggml_tensor * ggml_ceil_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

بنية ggml_tensor * ggml_گیڑا(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_گیڑا);
}

بنية ggml_tensor * ggml_گیڑا_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_گیڑا);
}

//ggml_trunc

بنية ggml_tensor * ggml_trunc(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

بنية ggml_tensor * ggml_trunc_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

بنية ggml_tensor * ggml_glu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_glu_op      op,
        bool                  swapped) {
    أرجع ggml_glu_impl(ctx, a, عدم, op, swapped);
}

بنية ggml_tensor * ggml_glu_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد ggml_glu_op      op) {
    أرجع ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

بنية ggml_tensor * ggml_reglu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_REGLU, false);
}

بنية ggml_tensor * ggml_reglu_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_REGLU, true);
}

بنية ggml_tensor * ggml_reglu_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

بنية ggml_tensor * ggml_geglu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU, false);
}

بنية ggml_tensor * ggml_geglu_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU, true);
}

بنية ggml_tensor * ggml_geglu_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

بنية ggml_tensor * ggml_swiglu(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_SWIGLU, false);
}

بنية ggml_tensor * ggml_swiglu_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_SWIGLU, true);
}

بنية ggml_tensor * ggml_swiglu_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

بنية ggml_tensor * ggml_geglu_erf(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_ERF, false);
}

بنية ggml_tensor * ggml_geglu_erf_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_ERF, true);
}

بنية ggml_tensor * ggml_geglu_erf_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

بنية ggml_tensor * ggml_geglu_تیز(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_تیز, false);
}

بنية ggml_tensor * ggml_geglu_تیز_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_تیز, true);
}

بنية ggml_tensor * ggml_geglu_تیز_ونڈو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_تیز, false);
}

بنية ggml_tensor * ggml_swiglu_oai(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 alpha,
        عائم                 حد_بندی) {
    بنية ggml_tensor * نتیجہ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_سیٹ_op_params_f32(نتیجہ, 2, alpha);
    ggml_سیٹ_op_params_f32(نتیجہ, 3, حد_بندی);

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_swiglu_clamp(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 حد_بندی) {
    بنية ggml_tensor * نتیجہ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_سیٹ_op_params_f32(نتیجہ, 3, حد_بندی);

    أرجع نتیجہ;
}

// ggml_norm

ساكن بنية ggml_tensor * ggml_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_NORM;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_norm(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_norm_impl(ctx, a, eps, false);
}

بنية ggml_tensor * ggml_norm_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ساكن بنية ggml_tensor * ggml_rms_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_RMS_NORM;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_rms_norm(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_rms_norm_impl(ctx, a, eps, false);
}

بنية ggml_tensor * ggml_rms_norm_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

بنية ggml_tensor * ggml_rms_norm_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 eps) {
    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params(نتیجہ, &eps, حجم(eps));

    نتیجہ->op     = GGML_OP_RMS_NORM_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_group_norm

ساكن بنية ggml_tensor * ggml_گروہ_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, n_groups);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, eps);

    نتیجہ->op     = GGML_OP_گروہ_NORM;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_گروہ_norm(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps) {
    أرجع ggml_گروہ_norm_impl(ctx, a, n_groups, eps, false);
}

بنية ggml_tensor * ggml_گروہ_norm_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps) {
    أرجع ggml_گروہ_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ساكن بنية ggml_tensor * ggml_l2_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, eps);

    نتیجہ->op     = GGML_OP_L2_NORM;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_l2_norm(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_l2_norm_impl(ctx, a, eps, false);
}

بنية ggml_tensor * ggml_l2_norm_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps) {
    أرجع ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_سیٹ_acc(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    بدل (a->op) {
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 0, prec_i32);
            }
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 3, prec_i32);
            }
            اكسر;
        افتراضي:
            أرجع false;
    };

    أرجع true;
}

bool ggml_prec_سیٹ_src(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec,
        صحيح                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    بدل (a->op) {
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                إذا (idx != 1) {
                    أرجع false;
                }

                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_سیٹ_op_params_i32(a, 2 + idx, prec_i32);
            }
            اكسر;
        افتراضي:
            أرجع false;
    };

    أرجع true;
}

// ggml_mul_mat

ساكن مضمن bool ggml_can_mul_mat(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

بنية ggml_tensor * ggml_mul_mat(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ثابت int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_MUL_MAT;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

فراغ ggml_mul_mat_سیٹ_prec(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_سیٹ_op_params_i32(a, 0, prec_i32);
}

فراغ ggml_mul_mat_سیٹ_hint(
        بنية ggml_tensor * a,
        تعداد ggml_op_hint    hint) {
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
بنية ggml_tensor * ggml_mul_mat_id(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * as,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ثابت int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_MUL_MAT_ID;
    نتیجہ->src[0] = as;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = ids;

    أرجع نتیجہ;
}

// ggml_out_prod

ساكن مضمن bool ggml_can_باہرلا_prod(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

بنية ggml_tensor * ggml_باہرلا_prod(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_باہرلا_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ثابت int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_باہرلا_PROD;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_scale

ساكن بنية ggml_tensor * ggml_scale_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s,
        عائم                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[2] = { s, b };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_SCALE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_scale(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s) {
    أرجع ggml_scale_impl(ctx, a, s, 0.0, false);
}

بنية ggml_tensor * ggml_scale_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s) {
    أرجع ggml_scale_impl(ctx, a, s, 0.0, true);
}

بنية ggml_tensor * ggml_scale_bias(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s,
        عائم                 b) {
    أرجع ggml_scale_impl(ctx, a, s, b, false);
}

بنية ggml_tensor * ggml_scale_bias_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s,
        عائم                 b) {
    أرجع ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ساكن بنية ggml_tensor * ggml_سیٹ_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(وتھ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, وتھ, inplace ? 1 : 0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_سیٹ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_سیٹ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, nb1, nb2, nb3, وتھ, false);
}

بنية ggml_tensor * ggml_سیٹ_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, nb1, nb2, nb3, وتھ, true);
}

بنية ggml_tensor * ggml_سیٹ_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], وتھ, false);
}

بنية ggml_tensor * ggml_سیٹ_1d_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], وتھ, true);
}

بنية ggml_tensor * ggml_سیٹ_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], وتھ, false);
}

بنية ggml_tensor * ggml_سیٹ_2d_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                وتھ) {
    أرجع ggml_سیٹ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], وتھ, true);
}

// ggml_cpy

ساكن بنية ggml_tensor * ggml_cpy_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, b);
    إذا (لڑی_لمبائی(b->name) > 0) {
        ggml_format_name(نتیجہ, "%s (copy of %s)", b->name, a->name);
    } وإلا {
        ggml_format_name(نتیجہ, "%s (copy)", a->name);
    }

    نتیجہ->op     = GGML_OP_CPY;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_cpy(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a,
        بنية ggml_tensor * b) {
    أرجع ggml_cpy_impl(ctx, a, b);
}

بنية ggml_tensor * ggml_cast(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد   ggml_type      type) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(نتیجہ, "%s (copy)", a->name);

    نتیجہ->op     = GGML_OP_CPY;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = نتیجہ; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    أرجع نتیجہ;
}

// ggml_cont

ساكن بنية ggml_tensor * ggml_cont_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);
    ggml_format_name(نتیجہ, "%s (cont)", a->name);

    نتیجہ->op     = GGML_OP_CONT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_cont(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a) {
    أرجع ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API بنية ggml_tensor * ggml_cont_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0) {
    أرجع ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API بنية ggml_tensor * ggml_cont_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    أرجع ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API بنية ggml_tensor * ggml_cont_3d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    أرجع ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

بنية ggml_tensor * ggml_cont_4d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(نتیجہ, "%s (cont)", a->name);

    نتیجہ->op     = GGML_OP_CONT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_reshape

بنية ggml_tensor * ggml_reshape(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a,
        بنية ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_reshape_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ثابت int64_t ne[1] = { ne0 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_reshape_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ثابت int64_t ne[2] = { ne0, ne1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_reshape_3d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_reshape_4d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(نتیجہ, "%s (reshaped)", a->name);

    نتیجہ->op     = GGML_OP_RESHAPE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

ساكن بنية ggml_tensor * ggml_view_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_dims,
        ثابت int64_t       * ne,
        size_t                وتھ) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, وتھ);
    ggml_format_name(نتیجہ, "%s (view)", a->name);

    ggml_سیٹ_op_params(نتیجہ, &وتھ, حجم(وتھ));

    نتیجہ->op     = GGML_OP_VIEW;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_view_1d

بنية ggml_tensor * ggml_view_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        size_t                وتھ) {
    بنية ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 1, &ne0, وتھ);

    أرجع نتیجہ;
}

// ggml_view_2d

بنية ggml_tensor * ggml_view_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                وتھ) {
    ثابت int64_t ne[2] = { ne0, ne1 };

    بنية ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 2, ne, وتھ);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = نتیجہ->nb[1]*ne1;
    نتیجہ->nb[3] = نتیجہ->nb[2];

    أرجع نتیجہ;
}

// ggml_view_3d

بنية ggml_tensor * ggml_view_3d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                وتھ) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };

    بنية ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 3, ne, وتھ);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = nb2;
    نتیجہ->nb[3] = نتیجہ->nb[2]*ne2;

    أرجع نتیجہ;
}

// ggml_view_4d

بنية ggml_tensor * ggml_view_4d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                وتھ) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    بنية ggml_tensor * نتیجہ = ggml_view_impl(ctx, a, 4, ne, وتھ);

    نتیجہ->nb[1] = nb1;
    نتیجہ->nb[2] = nb2;
    نتیجہ->nb[3] = nb3;

    أرجع نتیجہ;
}

// ggml_permute

بنية ggml_tensor * ggml_ترتیب_بدلو(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   axis0,
        صحيح                   axis1,
        صحيح                   axis2,
        صحيح                   axis3) {
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

    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);
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

    أرجع نتیجہ;
}

// ggml_transpose

بنية ggml_tensor * ggml_transpose(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);
    ggml_format_name(نتیجہ, "%s (transposed)", a->name);

    نتیجہ->ne[0] = a->ne[1];
    نتیجہ->ne[1] = a->ne[0];

    نتیجہ->nb[0] = a->nb[1];
    نتیجہ->nb[1] = a->nb[0];

    نتیجہ->op     = GGML_OP_TRANSPOSE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_get_rows

بنية ggml_tensor * ggml_get_صفاں(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    تعداد ggml_type type = GGML_TYPE_F32;
    إذا (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    نتیجہ->op     = GGML_OP_GET_صفاں;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_get_rows_back

بنية ggml_tensor * ggml_get_صفاں_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_میٹرکس(a) && ggml_is_سمتیہ(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_میٹرکس(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    نتیجہ->op     = GGML_OP_GET_صفاں_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_set_rows

بنية ggml_tensor * ggml_سیٹ_صفاں(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_صفاں(a));
    GGML_ASSERT(ggml_is_contiguous_صفاں(b));

    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_سیٹ_صفاں;
    نتیجہ->src[0] = b;
    نتیجہ->src[1] = c;
    نتیجہ->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    أرجع نتیجہ;
}

// ggml_diag

بنية ggml_tensor * ggml_diag(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ثابت int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, 4, ne);

    نتیجہ->op     = GGML_OP_DIAG;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_diag_mask_inf

ساكن بنية ggml_tensor * ggml_diag_نقاب_inf_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_DIAG_نقاب_INF;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_diag_نقاب_inf(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_نقاب_inf_impl(ctx, a, n_past, false);
}

بنية ggml_tensor * ggml_diag_نقاب_inf_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_نقاب_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ساكن بنية ggml_tensor * ggml_diag_نقاب_zero_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_DIAG_نقاب_ZERO;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_diag_نقاب_zero(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_نقاب_zero_impl(ctx, a, n_past, false);
}

بنية ggml_tensor * ggml_diag_نقاب_zero_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_نقاب_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ساكن بنية ggml_tensor * ggml_clamp_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 min,
        عائم                 max,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[] = { min, max };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CLAMP;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_clamp(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 min,
    عائم                 max) {
    أرجع ggml_clamp_impl(ctx, a, min, max, false);
}

بنية ggml_tensor * ggml_clamp_inplace(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 min,
    عائم                 max) {
    أرجع ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ساكن بنية ggml_tensor * ggml_soft_max_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * نقاب,
        عائم                 scale,
        عائم                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    إذا (نقاب) {
        GGML_ASSERT(نقاب->type == GGML_TYPE_F16 || نقاب->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(نقاب));
        GGML_ASSERT(نقاب->ne[0] == a->ne[0]);
        GGML_ASSERT(نقاب->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%نقاب->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%نقاب->ne[3] == 0);
    }

    إذا (max_bias > 0.0f) {
        GGML_ASSERT(نقاب);
    }

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[] = { scale, max_bias };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_SOFT_MAX;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = نقاب;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_soft_max(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_soft_max_impl(ctx, a, عدم, 1.0f, 0.0f, false);
}

بنية ggml_tensor * ggml_soft_max_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_soft_max_impl(ctx, a, عدم, 1.0f, 0.0f, true);
}

بنية ggml_tensor * ggml_soft_max_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * نقاب,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_impl(ctx, a, نقاب, scale, max_bias, false);
}

بنية ggml_tensor * ggml_soft_max_ext_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * نقاب,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_impl(ctx, a, نقاب, scale, max_bias, true);
}

فراغ ggml_soft_max_جوڑو_sinks(
        بنية ggml_tensor * a,
        بنية ggml_tensor * sinks) {
    إذا (!sinks) {
        a->src[2] = عدم;
        أرجع;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == عدم);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ساكن بنية ggml_tensor * ggml_soft_max_ext_پٹھ_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias,
        bool                  inplace) {
    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_SOFT_MAX_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    حافظہ_نقل((عائم *) نتیجہ->op_params + 0, &scale,    حجم(عائم));
    حافظہ_نقل((عائم *) نتیجہ->op_params + 1, &max_bias, حجم(عائم));

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_soft_max_ext_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_ext_پٹھ_impl(ctx, a, b, scale, max_bias, false);
}

بنية ggml_tensor * ggml_soft_max_ext_پٹھ_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_ext_پٹھ_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ساكن بنية ggml_tensor * ggml_rope_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((منوال & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_سمتیہ(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_ورتیا = منوال & GGML_ROPE_TYPE_MROPE;
    إذا (mrope_ورتیا) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } وإلا {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    إذا (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, منوال, /*n_ctx*/ 0, n_ctx_orig };
    حافظہ_نقل(params +  5, &freq_ادھار,    حجم(عائم));
    حافظہ_نقل(params +  6, &freq_scale,   حجم(عائم));
    حافظہ_نقل(params +  7, &ext_ضارب,   حجم(عائم));
    حافظہ_نقل(params +  8, &attn_ضارب,  حجم(عائم));
    حافظہ_نقل(params +  9, &beta_fast,    حجم(عائم));
    حافظہ_نقل(params + 10, &beta_slow,    حجم(عائم));
    إذا (mrope_ورتیا && sections) {
        حافظہ_نقل(params + 11, sections,  حجم(int32_t) * GGML_MROPE_SECTIONS);
    } وإلا {
        memset(params + 11, 0,         حجم(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_ROPE;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_rope(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   منوال) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, منوال, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

بنية ggml_tensor * ggml_rope_multi(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_multi_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, true
    );
}

بنية ggml_tensor * ggml_rope_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   منوال) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, منوال, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

بنية ggml_tensor * ggml_rope_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, عدم, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_ext_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, عدم, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, true
    );
}

بنية ggml_tensor * ggml_rope_custom(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_custom_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, منوال, n_ctx_orig, freq_ادھار, freq_scale,
        ext_ضارب, attn_ضارب, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ساكن عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم ادھار) {
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم)M_PI)) / (2 * logf(ادھار));
}

فراغ ggml_rope_yarn_corr_dims(
    صحيح n_dims, صحيح n_ctx_orig, عائم freq_ادھار, عائم beta_fast, عائم beta_slow, عائم dims[2]
) {
    // start and end correction dims
    عائم شروع = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ادھار));
    عائم سرا   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ادھار));
    dims[0] = MAX(0, شروع);
    dims[1] = MIN(n_dims - 1, سرا);
}

// ggml_rope_back

بنية ggml_tensor * ggml_rope_ext_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    بنية ggml_tensor * نتیجہ = ggml_rope_ext(
        ctx, a, b, c, n_dims, منوال, n_ctx_orig, freq_ادھار, freq_scale, ext_ضارب, attn_ضارب, beta_fast, beta_slow);
    نتیجہ->op = GGML_OP_ROPE_پٹھ;
    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_rope_multi_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[4],
        صحيح                   منوال,
        صحيح                   n_ctx_orig,
        عائم                 freq_ادھار,
        عائم                 freq_scale,
        عائم                 ext_ضارب,
        عائم                 attn_ضارب,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    بنية ggml_tensor * نتیجہ = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, منوال, n_ctx_orig, freq_ادھار, freq_scale, ext_ضارب, attn_ضارب, beta_fast, beta_slow);
    نتیجہ->op = GGML_OP_ROPE_پٹھ;
    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_rope_سیٹ_وتھ(
        بنية ggml_tensor  * a,
        صحيح                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_پٹھ);
    GGML_ASSERT(n_offs >= 0);

    ثابت int32_t منوال = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(منوال != GGML_ROPE_TYPE_VISION);

    ggml_سیٹ_op_params_i32(a, 15, n_offs);
    أرجع a;
}

ساكن int64_t ggml_calc_conv_output_ناپ(int64_t ins, int64_t ks, صحيح s, صحيح p, صحيح d) {
    أرجع (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
بنية ggml_tensor * ggml_im2col(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   s1,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   d0,
        صحيح                   d1,
        bool                  is_2D,
        تعداد ggml_type        dst_type) {
    إذا (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } وإلا {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ثابت int64_t OH = is_2D ? ggml_calc_conv_output_ناپ(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ثابت int64_t OW =         ggml_calc_conv_output_ناپ(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ثابت int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_خانہ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_im2col_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        int64_t             * ne,
        صحيح                   s0,
        صحيح                   s1,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   d0,
        صحيح                   d1,
        bool                  is_2D) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_خانہ_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_conv_1d

بنية ggml_tensor * ggml_conv_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   p0,
        صحيح                   d0) {
    بنية ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    بنية ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    نتیجہ = ggml_reshape_3d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 2, 1, 3));                // [N, OC, OL]

    أرجع نتیجہ;
}

// ggml_conv_1d_ph

بنية ggml_tensor* ggml_conv_1d_ph(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s,
        صحيح                   d) {
    أرجع ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

بنية ggml_tensor * ggml_conv_1d_dw(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   p0,
        صحيح                   d0) {
    بنية ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    بنية ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    بنية ggml_tensor * نتیجہ = ggml_mul_mat(ctx, im2col, a);

    نتیجہ = ggml_reshape_3d(ctx, نتیجہ, نتیجہ->ne[0], نتیجہ->ne[2], 1);

    أرجع نتیجہ;
}

// ggml_conv_1d_dw_ph

بنية ggml_tensor * ggml_conv_1d_dw_ph(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   d0) {
    أرجع ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

بنية ggml_tensor * ggml_col2im_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   s0,
        صحيح                   oc,
        صحيح                   p0) {
    GGML_ASSERT(ggml_is_میٹرکس(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ثابت int64_t K_OC = a->ne[0];
    ثابت int64_t T_وچ = a->ne[1];
    ثابت int64_t K = K_OC / oc;
    ثابت int64_t T_باہرلا = (T_وچ - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_باہرلا > 0);

    ثابت int64_t ne[4] = { T_باہرلا, oc, 1, 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_COL2IM_1D;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_conv_transpose_1d

ساكن int64_t ggml_calc_conv_transpose_1d_output_ناپ(int64_t ins, int64_t ks, صحيح s, صحيح p, صحيح d) {
    أرجع (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API بنية ggml_tensor * ggml_conv_transpose_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   p0,
        صحيح                   d0) {
    GGML_ASSERT(ggml_is_میٹرکس(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_ناپ(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CONV_TRANSPOSE_1D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
بنية ggml_tensor * ggml_conv_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   s1,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   d0,
        صحيح                   d1) {
    بنية ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    بنية ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 1, 3, 2)); // [N, OC, OH, OW]


    أرجع نتیجہ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
بنية ggml_tensor * ggml_im2col_3d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        int64_t               IC,
        صحيح                   s0, // stride width
        صحيح                   s1, // stride height
        صحيح                   s2, // stride depth
        صحيح                   p0, // padding width
        صحيح                   p1, // padding height
        صحيح                   p2, // padding depth
        صحيح                   d0, // dilation width
        صحيح                   d1, // dilation height
        صحيح                   d2, // dilation depth
        تعداد ggml_type        dst_type) {
    ثابت int64_t N = b->ne[3] / IC;
    ثابت int64_t ID = b->ne[2];
    ثابت int64_t IH = b->ne[1];
    ثابت int64_t IW = b->ne[0];

    ثابت int64_t OC = a->ne[3] / IC;
    بن_ورتیا(OC);
    ثابت int64_t KD = a->ne[2];
    ثابت int64_t KH = a->ne[1];
    ثابت int64_t KW = a->ne[0];
    ثابت int64_t OD = ggml_calc_conv_output_ناپ(ID, KD, s2, p2, d2);
    ثابت int64_t OH = ggml_calc_conv_output_ناپ(IH, KH, s1, p1, d1);
    ثابت int64_t OW = ggml_calc_conv_output_ناپ(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ثابت int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_IM2_خانہ_3_D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
بنية ggml_tensor * ggml_conv_3d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        int64_t               IC,
        صحيح                   s0, // stride width
        صحيح                   s1, // stride height
        صحيح                   s2, // stride depth
        صحيح                   p0, // padding width
        صحيح                   p1, // padding height
        صحيح                   p2, // padding depth
        صحيح                   d0, // dilation width
        صحيح                   d1, // dilation height
        صحيح                   d2  // dilation depth
        ) {
    بنية ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    بنية ggml_tensor * نتیجہ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    نتیجہ = ggml_cont(ctx, ggml_ترتیب_بدلو(ctx, نتیجہ, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    أرجع نتیجہ;
}

// ggml_conv_2d_sk_p0

بنية ggml_tensor * ggml_conv_2d_sk_p0(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

بنية ggml_tensor * ggml_conv_2d_s1_ph(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

بنية ggml_tensor * ggml_conv_2d_dw(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   s1,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   d0,
        صحيح                   d1) {
    بنية ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    بنية ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    بنية ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    بنية ggml_tensor * نتیجہ = ggml_mul_mat(ctx, new_a, new_b);
    نتیجہ = ggml_reshape_4d(ctx, نتیجہ, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    أرجع نتیجہ;
}

// ggml_conv_2d_dw_direct

بنية ggml_tensor * ggml_conv_2d_dw_سدھا(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   stride0,
        صحيح                   stride1,
        صحيح                   pad0,
        صحيح                   pad1,
        صحيح                   dilation0,
        صحيح                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ناپ(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_ناپ(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, b->type, 4, ne);

    إذا (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ثابت int64_t type_ناپ = ggml_type_ناپ(نتیجہ->type);
        GGML_ASSERT(ggml_blck_ناپ(نتیجہ->type) == 1);
        نتیجہ->nb[0] = نتیجہ->ne[2] * type_ناپ;
        نتیجہ->nb[1] = نتیجہ->ne[0] * نتیجہ->nb[0];
        نتیجہ->nb[2] = type_ناپ;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_CONV_2D_DW;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    أرجع نتیجہ;
}

// ggml_conv_2d_direct

بنية ggml_tensor * ggml_conv_2d_سدھا(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        بنية ggml_tensor  * b,   // input data [W, H, C, N]
        صحيح                   s0,  // stride dimension 0
        صحيح                   s1,  // stride dimension 1
        صحيح                   p0,  // padding dimension 0
        صحيح                   p1,  // padding dimension 1
        صحيح                   d0,  // dilation dimension 0
        صحيح                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ناپ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ناپ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, s0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, s1);
    ggml_سیٹ_op_params_i32(نتیجہ, 2, p0);
    ggml_سیٹ_op_params_i32(نتیجہ, 3, p1);
    ggml_سیٹ_op_params_i32(نتیجہ, 4, d0);
    ggml_سیٹ_op_params_i32(نتیجہ, 5, d1);

    نتیجہ->op = GGML_OP_CONV_2D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_conv_3d_direct

بنية ggml_tensor * ggml_conv_3d_سدھا(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   s1,
        صحيح                   s2,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   p2,
        صحيح                   d0,
        صحيح                   d1,
        صحيح                   d2,
        صحيح                   c,
        صحيح                   n,
        صحيح                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ناپ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ناپ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_ناپ(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    أرجع نتیجہ;
}

// ggml_conv_transpose_2d_p0

ساكن int64_t ggml_calc_conv_transpose_output_ناپ(int64_t ins, int64_t ks, صحيح s, صحيح p) {
    أرجع (ins - 1) * s - 2 * p + ks;
}

بنية ggml_tensor * ggml_conv_transpose_2d_p0(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_output_ناپ(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_ناپ(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    بنية ggml_tensor* نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, stride);

    نتیجہ->op     = GGML_OP_CONV_TRANSPOSE_2D;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_pool_*

ساكن int64_t ggml_calc_pool_output_ناپ(int64_t ins, صحيح ks, صحيح s, عائم p) {
    أرجع (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

بنية ggml_tensor * ggml_pool_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_op_pool     op,
        صحيح                   k0,
        صحيح                   s0,
        صحيح                   p0) {
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_ناپ(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_POOL_1D;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_pool_2d

بنية ggml_tensor * ggml_pool_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_op_pool     op,
        صحيح                   k0,
        صحيح                   k1,
        صحيح                   s0,
        صحيح                   s1,
        عائم                 p0,
        عائم                 p1) {
    بنية ggml_tensor * نتیجہ;
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_ناپ(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_ناپ(a->ne[1], k1, s1, p1),
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

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_pool_2d_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * af,
        تعداد ggml_op_pool     op,
        صحيح                   k0,
        صحيح                   k1,
        صحيح                   s0,
        صحيح                   s1,
        عائم                 p0,
        عائم                 p1) {
    بنية ggml_tensor * نتیجہ;
    نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_POOL_2_D_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = af;

    أرجع نتیجہ;
}

// ggml_upscale / ggml_interpolate

ساكن بنية ggml_tensor * ggml_interpolate_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              منوال) {
    GGML_ASSERT((منوال & 0xFF) < GGML_SCALE_منوال_گنتی);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(منوال & GGML_SCALE_FLAG_ANTIALIAS) || (منوال & 0xFF) == GGML_SCALE_منوال_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t)منوال);

    نتیجہ->op     = GGML_OP_UPSCALE;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_upscale(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   scale_ضارب,
        تعداد ggml_scale_منوال  منوال) {
    GGML_ASSERT(scale_ضارب > 1);
    أرجع ggml_interpolate_impl(ctx, a, a->ne[0] * scale_ضارب, a->ne[1] * scale_ضارب, a->ne[2], a->ne[3], منوال);
}

بنية ggml_tensor * ggml_upscale_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   ne0,
        صحيح                   ne1,
        صحيح                   ne2,
        صحيح                   ne3,
        تعداد ggml_scale_منوال  منوال) {
    أرجع ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, منوال);
}

بنية ggml_tensor * ggml_interpolate(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              منوال) {
    أرجع ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, منوال);
}

// ggml_pad

بنية ggml_tensor * ggml_pad(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   p2,
        صحيح                   p3) {
    أرجع ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

بنية ggml_tensor * ggml_pad_circular(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   p0,
        صحيح                   p1,
        صحيح                   p2,
        صحيح                   p3) {
    أرجع ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

بنية ggml_tensor * ggml_pad_ext(
            بنية ggml_context * ctx,
            بنية ggml_tensor  * a,
            صحيح                  lp0,
            صحيح                  rp0,
            صحيح                  lp1,
            صحيح                  rp1,
            صحيح                  lp2,
            صحيح                  rp2,
            صحيح                  lp3,
            صحيح                  rp3
            ) {
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type,
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

    أرجع نتیجہ;
}

// ggml_pad_ext_circular

بنية ggml_tensor * ggml_pad_ext_circular(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                  lp0,
        صحيح                  rp0,
        صحيح                  lp1,
        صحيح                  rp1,
        صحيح                  lp2,
        صحيح                  rp2,
        صحيح                  lp3,
        صحيح                  rp3
        ) {
    بنية ggml_tensor * نتیجہ = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_سیٹ_op_params_i32(نتیجہ, 8, 1); // circular
    أرجع نتیجہ;
}

// ggml_pad_reflect_1d

بنية ggml_tensor * ggml_pad_reflect_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   p0,
        صحيح                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_PAD_REFLECT_1D;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_roll

بنية ggml_tensor * ggml_roll(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   shift0,
        صحيح                   shift1,
        صحيح                   shift2,
        صحيح                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_ناپ(a->type));
    GGML_ASSERT(مطلق_قدر(shift0) < a->ne[0]);
    GGML_ASSERT(مطلق_قدر(shift1) < a->ne[1]);
    GGML_ASSERT(مطلق_قدر(shift2) < a->ne[2]);
    GGML_ASSERT(مطلق_قدر(shift3) < a->ne[3]);

    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, shift0);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, shift1);
    ggml_سیٹ_op_params_i32(نتیجہ, 2, shift2);
    ggml_سیٹ_op_params_i32(نتیجہ, 3, shift3);

    نتیجہ->op     = GGML_OP_ROLL;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_timestep_embedding

بنية ggml_tensor * ggml_timestep_embedding(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * timesteps,
        صحيح                   dim,
        صحيح                   max_period) {

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, dim);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, max_period);

    نتیجہ->op     = GGML_OP_TIMESTEP_EMBEDDING;
    نتیجہ->src[0] = timesteps;

    أرجع نتیجہ;
}

// ggml_tri

بنية ggml_tensor * ggml_tri(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    تعداد ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, type);

    نتیجہ->op = GGML_OP_TRI;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_fill

ساكن بنية ggml_tensor * ggml_fill_impl(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, c);

    نتیجہ->op = GGML_OP_FILL;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_fill(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 c) {
    أرجع ggml_fill_impl(ctx, a, c, false);
}

بنية ggml_tensor * ggml_fill_inplace(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 c) {
    أرجع ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

بنية ggml_tensor * ggml_argsort(
        بنية ggml_context  * ctx,
        بنية ggml_tensor   * a,
        تعداد ggml_ترتیب_دیو_ترتیب   ترتیب) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) ترتیب);

    نتیجہ->op     = GGML_OP_ARGSORT;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_argsort_top_k

بنية ggml_tensor * ggml_argsort_چوٹی_k(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    بنية ggml_tensor * نتیجہ = ggml_argsort(ctx, a, GGML_ترتیب_دیو_ترتیب_DESC);

    نتیجہ = ggml_view_4d(ctx, نتیجہ,
                k, نتیجہ->ne[1], نتیجہ->ne[2], نتیجہ->ne[3],
                   نتیجہ->nb[1], نتیجہ->nb[2], نتیجہ->nb[3],
                0);

    أرجع نتیجہ;
}

// ggml_top_k

بنية ggml_tensor * ggml_چوٹی_k(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    نتیجہ->op     = GGML_OP_چوٹی_K;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_arange

بنية ggml_tensor * ggml_arange(
        بنية ggml_context * ctx,
        عائم                 شروع,
        عائم                 stop,
        عائم                 قدم) {
    GGML_ASSERT(stop > شروع);

    ثابت int64_t قدماں = (int64_t) ceilf((stop - شروع) / قدم);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, قدماں);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, شروع);
    ggml_سیٹ_op_params_f32(نتیجہ, 1, stop);
    ggml_سیٹ_op_params_f32(نتیجہ, 2, قدم);

    نتیجہ->op = GGML_OP_ARANGE;

    أرجع نتیجہ;
}

// ggml_flash_attn_ext

بنية ggml_tensor * ggml_flash_attn_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * نقاب,
        عائم                 scale,
        عائم                 max_bias,
        عائم                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    إذا (نقاب) {
        GGML_ASSERT(نقاب->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(نقاب));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % نقاب->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % نقاب->ne[3] == 0);
    }

    إذا (max_bias > 0.0f) {
        GGML_ASSERT(نقاب);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    عائم params[] = { scale, max_bias, logit_softcap };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_FLASH_ATTN_EXT;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = نقاب;

    أرجع نتیجہ;
}


فراغ ggml_flash_attn_ext_سیٹ_prec(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_سیٹ_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

تعداد ggml_prec ggml_flash_attn_ext_get_prec(
        ثابت بنية ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    أرجع (تعداد ggml_prec) prec_i32;
}

فراغ ggml_flash_attn_ext_سیٹ_n_kv_max(
        بنية ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_سیٹ_op_params_i32(a, 4, n_kv_max);
}

فراغ ggml_flash_attn_ext_جوڑو_sinks(
        بنية ggml_tensor * a,
        بنية ggml_tensor * sinks) {
    إذا (!sinks) {
        a->src[4] = عدم;
        أرجع;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == عدم);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

بنية ggml_tensor * ggml_flash_attn_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * d,
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

    تعداد ggml_type نتیجہ_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_ناپ(نتیجہ_type) == 1);
    ثابت size_t tsize = ggml_type_ناپ(نتیجہ_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ثابت size_t سرا    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ثابت size_t nelements = (سرا + tsize - 1)/tsize;

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_سیٹ_op_params(نتیجہ, &masked_i, حجم(masked_i));

    نتیجہ->op     = GGML_OP_FLASH_ATTN_پٹھ;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = d;

    أرجع نتیجہ;
}

// ggml_ssm_conv

بنية ggml_tensor * ggml_ssm_conv(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * sx,
        بنية ggml_tensor  * c) {
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

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    نتیجہ->op     = GGML_OP_SSM_CONV;
    نتیجہ->src[0] = sx;
    نتیجہ->src[1] = c;

    أرجع نتیجہ;
}

// ggml_ssm_scan

بنية ggml_tensor * ggml_ssm_جانچ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * s,
        بنية ggml_tensor  * x,
        بنية ggml_tensor  * dt,
        بنية ggml_tensor  * A,
        بنية ggml_tensor  * B,
        بنية ggml_tensor  * C,
        بنية ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_ناپ(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_ناپ(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_ناپ(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ثابت int64_t d_state      = s->ne[0];
        ثابت int64_t سر_dim     = x->ne[0];
        ثابت int64_t n_سر       = x->ne[1];
        ثابت int64_t n_seq_ٹوکن = x->ne[2];
        ثابت int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_سر);
        GGML_ASSERT(dt->ne[1] == n_seq_ٹوکن);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == سر_dim);
        GGML_ASSERT(s->ne[2] == n_سر);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_ٹوکن);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_سمتیہ(ids));
        GGML_ASSERT(A->ne[1] == n_سر);
        GGML_ASSERT(ggml_is_میٹرکس(A));

        إذا (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    نتیجہ->op   = GGML_OP_SSM_جانچ;
    نتیجہ->src[0] = s;
    نتیجہ->src[1] = x;
    نتیجہ->src[2] = dt;
    نتیجہ->src[3] = A;
    نتیجہ->src[4] = B;
    نتیجہ->src[5] = C;
    نتیجہ->src[6] = ids;

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) K);

    أرجع نتیجہ;
}

// ggml_win_part

بنية ggml_tensor * ggml_win_part(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ثابت صحيح px = (w - a->ne[1]%w)%w;
    ثابت صحيح py = (w - a->ne[2]%w)%w;

    ثابت صحيح npx = (px + a->ne[1])/w;
    ثابت صحيح npy = (py + a->ne[2])/w;
    ثابت صحيح np  = npx*npy;

    ثابت int64_t ne[4] = { a->ne[0], w, w, np, };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_WIN_PART;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_win_unpart

بنية ggml_tensor * ggml_win_unpart(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   w0,
        صحيح                   h0,
        صحيح                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ثابت int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_سیٹ_op_params(نتیجہ, params, حجم(params));

    نتیجہ->op     = GGML_OP_WIN_UNPART;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_get_rel_pos

بنية ggml_tensor * ggml_get_rel_pos(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   qh,
        صحيح                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ثابت int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    نتیجہ->op     = GGML_OP_GET_REL_POS;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

// ggml_add_rel_pos

ساكن بنية ggml_tensor * ggml_جوڑو_rel_pos_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * pw,
        بنية ggml_tensor  * ph,
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

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_سیٹ_op_params_i32(نتیجہ, 0, inplace ? 1 : 0);

    نتیجہ->op     = GGML_OP_جوڑو_REL_POS;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = pw;
    نتیجہ->src[2] = ph;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_جوڑو_rel_pos(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * pw,
        بنية ggml_tensor  * ph) {
    أرجع ggml_جوڑو_rel_pos_impl(ctx, a, pw, ph, false);
}

بنية ggml_tensor * ggml_جوڑو_rel_pos_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * pw,
        بنية ggml_tensor  * ph) {
    أرجع ggml_جوڑو_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

بنية ggml_tensor * ggml_rwkv_wkv6(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * r,
        بنية ggml_tensor  * tf,
        بنية ggml_tensor  * td,
        بنية ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_ٹوکن = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ٹوکن);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_ٹوکن);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_ٹوکن);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_ٹوکن + S * n_seqs, 1, 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_RWKV_WKV6;
    نتیجہ->src[0] = k;
    نتیجہ->src[1] = v;
    نتیجہ->src[2] = r;
    نتیجہ->src[3] = tf;
    نتیجہ->src[4] = td;
    نتیجہ->src[5] = state;

    أرجع نتیجہ;
}

// ggml_gated_linear_attn

بنية ggml_tensor * ggml_gated_خطی_attn(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * g,
        بنية ggml_tensor  * state,
        عائم scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_ٹوکن = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ٹوکن);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_ٹوکن);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_ٹوکن);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_ٹوکن + S * n_seqs, 1, 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, scale);

    نتیجہ->op     = GGML_OP_GATED_خطی_ATTN;
    نتیجہ->src[0] = k;
    نتیجہ->src[1] = v;
    نتیجہ->src[2] = q;
    نتیجہ->src[3] = g;
    نتیجہ->src[4] = state;

    أرجع نتیجہ;
}

// ggml_rwkv_wkv7

بنية ggml_tensor * ggml_rwkv_wkv7(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * r,
        بنية ggml_tensor  * w,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ثابت int64_t S = k->ne[0];
    ثابت int64_t H = k->ne[1];
    ثابت int64_t n_ٹوکن = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_ٹوکن);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_ٹوکن);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ٹوکن);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_ٹوکن);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_ٹوکن);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_ٹوکن + S * n_seqs, 1, 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op     = GGML_OP_RWKV_WKV7;
    نتیجہ->src[0] = r;
    نتیجہ->src[1] = w;
    نتیجہ->src[2] = k;
    نتیجہ->src[3] = v;
    نتیجہ->src[4] = a;
    نتیجہ->src[5] = b;
    نتیجہ->src[6] = state;

    أرجع نتیجہ;
}

// ggml_unary

ساكن بنية ggml_tensor * ggml_unary_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_صفاں(a));

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) op);

    نتیجہ->op     = GGML_OP_UNARY;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_unary(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_unary_op    op) {
    أرجع ggml_unary_impl(ctx, a, op, false);
}

بنية ggml_tensor * ggml_unary_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_unary_op    op) {
    أرجع ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ساكن بنية ggml_tensor * ggml_map_custom1_impl(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM1;
    نتیجہ->src[0] = a;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_map_custom1(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

بنية ggml_tensor * ggml_map_custom1_inplace(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ساكن بنية ggml_tensor * ggml_map_custom2_impl(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM2;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_map_custom2(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

بنية ggml_tensor * ggml_map_custom2_inplace(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ساكن بنية ggml_tensor * ggml_map_custom3_impl(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        بنية ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    بنية ggml_tensor * نتیجہ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op     = GGML_OP_MAP_CUSTOM3;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_map_custom3(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        بنية ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

بنية ggml_tensor * ggml_map_custom3_inplace(
        بنية ggml_context      * ctx,
        بنية ggml_tensor       * a,
        بنية ggml_tensor       * b,
        بنية ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحيح                        n_tasks,
        فراغ                     * userdata) {
    أرجع ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

بنية ggml_tensor * ggml_custom_4d(
        بنية ggml_context * ctx,
        تعداد ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        بنية ggml_tensor ** دلیلاں_2,
        صحيح                   n_دلیلاں,
        ggml_custom_op_t      fun,
        صحيح                   n_tasks,
        فراغ                * userdata) {

    GGML_ASSERT(n_دلیلاں < GGML_MAX_SRC);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    بنية ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op = GGML_OP_CUSTOM;
    لكل (صحيح i = 0; i < n_دلیلاں; i++) {
        نتیجہ->src[i] = دلیلاں_2[i];
    }

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_custom_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor ** دلیلاں_2,
        صحيح                   n_دلیلاں,
        ggml_custom_op_t      fun,
        صحيح                   n_tasks,
        فراغ                * userdata) {

    GGML_ASSERT(n_دلیلاں < GGML_MAX_SRC - 1);

    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    بنية ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_سیٹ_op_params(نتیجہ, &params, حجم(params));

    نتیجہ->op = GGML_OP_CUSTOM;
    نتیجہ->src[0] = a;
    لكل (صحيح i = 0; i < n_دلیلاں; i++) {
        نتیجہ->src[i + 1] = دلیلاں_2[i];
    }

    أرجع نتیجہ;
}
// ggml_cross_entropy_loss

بنية ggml_tensor * ggml_cross_entropy_loss(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, a->type, 1);

    نتیجہ->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_cross_entropy_loss_back

بنية ggml_tensor * ggml_cross_entropy_loss_پٹھ(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    بنية ggml_tensor * نتیجہ = ggml_dup_tensor(ctx, b);

    نتیجہ->op     = GGML_OP_CROSS_ENTROPY_LOSS_پٹھ;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;
    نتیجہ->src[2] = c;

    أرجع نتیجہ;
}

// opt_step_adamw

بنية ggml_tensor * ggml_opt_قدم_adamw(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * grad,
        بنية ggml_tensor  * m,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_OPT_قدم_ADAMW;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = grad;
    نتیجہ->src[2] = m;
    نتیجہ->src[3] = v;
    نتیجہ->src[4] = adamw_params;

    أرجع نتیجہ;
}

// opt_step_sgd

بنية ggml_tensor * ggml_opt_قدم_sgd(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * grad,
        بنية ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    بنية ggml_tensor * نتیجہ = ggml_view_tensor(ctx, a);

    نتیجہ->op     = GGML_OP_OPT_قدم_SGD;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = grad;
    نتیجہ->src[2] = params;

    أرجع نتیجہ;
}

// solve_tri

بنية ggml_tensor * ggml_solve_tri(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  بایاں,
        bool                  ہیٹھلا,
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

    GGML_ASSERT(ہیٹھلا && بایاں && !uni); // TODO: support other variants

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    نتیجہ->op     = GGML_OP_SOLVE_TRI;
    نتیجہ->src[0] = a;
    نتیجہ->src[1] = b;

    أرجع نتیجہ;
}

// ggml_gated_delta_net

بنية ggml_tensor * ggml_gated_delta_net(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * g,
        بنية ggml_tensor  * beta,
        بنية ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_صفاں(q));
    GGML_ASSERT(ggml_is_contiguous_صفاں(k));
    GGML_ASSERT(ggml_is_contiguous_صفاں(v));
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
    ثابت int64_t n_ٹوکن = v->ne[2];
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
    ثابت int64_t state_صفاں = K * S_v * n_seqs;
    ثابت int64_t ne[4] = { S_v * H, n_ٹوکن * n_seqs + state_صفاں, 1, 1 };
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_سیٹ_op_params_i32(نتیجہ, 0, (int32_t) K);

    نتیجہ->op     = GGML_OP_GATED_DELTA_NET;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = v;
    نتیجہ->src[3] = g;
    نتیجہ->src[4] = beta;
    نتیجہ->src[5] = state;

    أرجع نتیجہ;
}

// ggml_lightning_indexer

بنية ggml_tensor * ggml_lightning_indexer(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * weights,
        بنية ggml_tensor  * نقاب) {

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
    بنية ggml_tensor * نتیجہ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    نتیجہ->op   = GGML_OP_LIGHTNING_INDEXER;
    نتیجہ->src[0] = q;
    نتیجہ->src[1] = k;
    نتیجہ->src[2] = weights;
    نتیجہ->src[3] = نقاب;

    أرجع نتیجہ;
}

// ggml_dsv4_hc_comb

بنية ggml_tensor * ggml_dsv4_hc_comb(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * mixes,
        بنية ggml_tensor  * scale,
        بنية ggml_tensor  * ادھار,
        عائم                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ادھار->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ثابت int64_t hc_mix_dim = mixes->ne[0];
    ثابت int64_t n_ٹوکن   = mixes->ne[1];

    int64_t hc = 0;
    لكل (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        إذا ((2 + i)*i == hc_mix_dim) {
            hc = i;
            اكسر;
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
    GGML_ASSERT(ادھار->ne[0] == hc_mix_dim);
    GGML_ASSERT(ادھار->ne[1] == 1);
    GGML_ASSERT(ادھار->ne[2] == 1);
    GGML_ASSERT(ادھار->ne[3] == 1);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_ٹوکن);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, eps);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, n_iter);

    نتیجہ->op     = GGML_OP_DSV4_HC_COMB;
    نتیجہ->src[0] = mixes;
    نتیجہ->src[1] = scale;
    نتیجہ->src[2] = ادھار;

    أرجع نتیجہ;
}

// ggml_dsv4_hc_pre

ساكن بنية ggml_tensor * ggml_dsv4_hc_pre_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * x,
        بنية ggml_tensor  * weights,
        عائم                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_ٹوکن = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    إذا (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ٹوکن);
    } وإلا {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ٹوکن);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ٹوکن);

    ggml_سیٹ_op_params_f32(نتیجہ, 0, scale);
    ggml_سیٹ_op_params_i32(نتیجہ, 1, gated ? 1 : 0);

    نتیجہ->op     = GGML_OP_DSV4_HC_PRE;
    نتیجہ->src[0] = x;
    نتیجہ->src[1] = weights;

    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_dsv4_hc_pre(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * x,
        بنية ggml_tensor  * weights) {
    أرجع ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

بنية ggml_tensor * ggml_dsv4_hc_pre_gated(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * x,
        بنية ggml_tensor  * gate,
        عائم                 scale) {
    أرجع ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

بنية ggml_tensor * ggml_dsv4_hc_post(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * x,
        بنية ggml_tensor  * residual,
        بنية ggml_tensor  * post,
        بنية ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t n_ٹوکن = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ٹوکن);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ٹوکن);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    إذا (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ٹوکن);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_ٹوکن);

    نتیجہ->op     = GGML_OP_DSV4_HC_POST;
    نتیجہ->src[0] = x;
    نتیجہ->src[1] = residual;
    نتیجہ->src[2] = post;
    نتیجہ->src[3] = comb;

    أرجع نتیجہ;
}

// ggml_top_k_mask

بنية ggml_tensor * ggml_چوٹی_k_نقاب(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * نقاب,
        بنية ggml_tensor  * چوٹی_k,
        بنية ggml_tensor  * score,
        بنية ggml_tensor  * بلاکاں,
        بنية ggml_tensor  * extra) {
    GGML_ASSERT(نقاب->type == GGML_TYPE_F16 || نقاب->type == GGML_TYPE_F32);
    GGML_ASSERT(نقاب->nb[0] == ggml_type_ناپ(نقاب->type));
    GGML_ASSERT(چوٹی_k->type == GGML_TYPE_I32);
    GGML_ASSERT(چوٹی_k->ne[1] % نقاب->ne[1] == 0 && چوٹی_k->ne[2] % نقاب->ne[2] == 0 && چوٹی_k->ne[3] % نقاب->ne[3] == 0);

    إذا (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == چوٹی_k->ne[1] && score->ne[2] == چوٹی_k->ne[2] && score->ne[3] == چوٹی_k->ne[3]);
    }

    إذا (بلاکاں) {
        GGML_ASSERT(بلاکاں->type == GGML_TYPE_I32);
        GGML_ASSERT(بلاکاں->ne[2] == 1 && بلاکاں->ne[3] == 1);
        GGML_ASSERT(!score || بلاکاں->ne[1] == score->ne[0]);
    }

    إذا (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == چوٹی_k->ne[1] && extra->ne[2] == چوٹی_k->ne[2] && extra->ne[3] == چوٹی_k->ne[3]);
    }

    بنية ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, نقاب->type, نقاب->ne[0], چوٹی_k->ne[1], چوٹی_k->ne[2], چوٹی_k->ne[3]);

    نتیجہ->op     = GGML_OP_چوٹی_K_نقاب;
    نتیجہ->src[0] = نقاب;
    نتیجہ->src[1] = چوٹی_k;
    نتیجہ->src[2] = score;
    نتیجہ->src[3] = بلاکاں;
    نتیجہ->src[4] = extra;

    أرجع نتیجہ;
}

////////////////////////////////////////////////////////////////////////////////

بنية ggml_hash_سیٹ ggml_hash_سیٹ_new(size_t ناپ) {
    ناپ = ggml_hash_ناپ(ناپ);
    بنية ggml_hash_سیٹ نتیجہ;
    نتیجہ.ناپ = ناپ;
    نتیجہ.کلیدیں = GGML_MALLOC(حجم(بنية ggml_tensor *) * ناپ);
    نتیجہ.ورتیا = GGML_CALLOC(ggml_bitset_ناپ(ناپ), حجم(ggml_bitset_t));
    أرجع نتیجہ;
}

فراغ ggml_hash_سیٹ_reset(بنية ggml_hash_سیٹ * hash_سیٹ) {
    memset(hash_سیٹ->ورتیا, 0, حجم(ggml_bitset_t) * ggml_bitset_ناپ(hash_سیٹ->ناپ));
}

فراغ ggml_hash_سیٹ_free(بنية ggml_hash_سیٹ * hash_سیٹ) {
    GGML_FREE(hash_سیٹ->ورتیا);
    GGML_FREE(hash_سیٹ->کلیدیں);
}

size_t ggml_hash_ناپ(size_t min_sz) {
    // next primes after powers of two
    ساكن ثابت size_t مفرد_عدد[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ساكن ثابت size_t n_مفرد_عدد = حجم(مفرد_عدد)/حجم(مفرد_عدد[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_مفرد_عدد;
    طالما (l < r) {
        size_t m = (l + r)/2;
        إذا (مفرد_عدد[m] < min_sz) {
            l = m + 1;
        } وإلا {
            r = m;
        }
    }
    size_t sz = l < n_مفرد_عدد ? مفرد_عدد[l] : min_sz | 1;
    أرجع sz;
}

بنية hash_map {
    بنية ggml_hash_سیٹ سیٹ;
    بنية ggml_tensor ** vals;
};

ساكن بنية hash_map * ggml_new_hash_map(size_t ناپ) {
    بنية hash_map * نتیجہ = GGML_MALLOC(حجم(بنية hash_map));
    نتیجہ->سیٹ = ggml_hash_سیٹ_new(ناپ);
    نتیجہ->vals = GGML_CALLOC(نتیجہ->سیٹ.ناپ, حجم(بنية ggml_tensor *));
    أرجع نتیجہ;
}

ساكن فراغ ggml_hash_map_free(بنية hash_map * map) {
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

ساكن فراغ ggml_جوڑو_or_سیٹ(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->ویکھیا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_جوڑو_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } وإلا {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_acc_or_سیٹ(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor,
        ثابت  size_t         nb1,
        ثابت  size_t         nb2,
        ثابت  size_t         nb3,
        ثابت  size_t         وتھ) {
    بنية ggml_tensor * src = cgraph->ویکھیا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, وتھ, cgraph->grad_accs[isrc]);
    } وإلا {
        بنية ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, وتھ, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->ویکھیا_hash_سیٹ.کلیدیں[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_add1_or_سیٹ(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->ویکھیا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } وإلا {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_sub_or_سیٹ(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->ویکھیا_hash_سیٹ.کلیدیں[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } وإلا {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_compute_backward(
        بنية ggml_context * ctx, بنية ggml_cgraph * cgraph, صحيح i, ثابت bool * grads_needed) {
    بنية ggml_tensor * tensor = cgraph->nodes[i];
    بنية ggml_tensor * grad   = ggml_گراف_get_grad(cgraph, tensor);

    إذا (!grad) {
        أرجع;
    }

    بنية ggml_tensor * src0 = tensor->src[0];
    بنية ggml_tensor * src1 = tensor->src[1];
    بنية ggml_tensor * src2 = tensor->src[2];
    بنية ggml_hash_سیٹ * hash_سیٹ = &cgraph->ویکھیا_hash_سیٹ;
    ثابت size_t isrc0 = src0 ? ggml_hash_لبھو(hash_سیٹ, src0) : (size_t) -1;
    ثابت size_t isrc1 = src1 ? ggml_hash_لبھو(hash_سیٹ, src1) : (size_t) -1;
    ثابت size_t isrc2 = src2 ? ggml_hash_لبھو(hash_سیٹ, src2) : (size_t) -1;
    ثابت bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_بھریا && ggml_bitset_get(hash_سیٹ->ورتیا, isrc0) && grads_needed[isrc0];
    ثابت bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_بھریا && ggml_bitset_get(hash_سیٹ->ورتیا, isrc1) && grads_needed[isrc1];
    ثابت bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_بھریا && ggml_bitset_get(hash_سیٹ->ورتیا, isrc2) && grads_needed[isrc2];

    بدل (tensor->op) {
        حالة GGML_OP_DUP: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
        } اكسر;
        حالة GGML_OP_جوڑو: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                بنية ggml_tensor * tmp = grad;
                إذا (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_پٹھ(ctx, tmp, src1);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, tmp);
            }
        } اكسر;
        حالة GGML_OP_ADD1: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_اوسط(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } اكسر;
        حالة GGML_OP_ACC: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ثابت size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ثابت size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ثابت size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ثابت size_t وتھ = ((int32_t *) tensor->op_params)[3];

                بنية ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, وتھ);

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } اكسر;
        حالة GGML_OP_SUB: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc1, grad);
            }
        } اكسر;
        حالة GGML_OP_MUL: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            إذا (src1_needs_grads) {
                بنية ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                إذا (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_پٹھ(ctx, tmp, src1);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, tmp);
            }
        } اكسر;
        حالة GGML_OP_DIV: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            إذا (src1_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } اكسر;
        حالة GGML_OP_SQR: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } اكسر;
        حالة GGML_OP_SQRT: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } اكسر;
        حالة GGML_OP_LOG: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_SIN: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } اكسر;
        حالة GGML_OP_COS: {
            إذا (src0_needs_grads) {
                ggml_sub_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } اكسر;
        حالة GGML_OP_جوڑ: {
            إذا (src0_needs_grads) {
                ggml_add1_or_سیٹ(ctx, cgraph, isrc0, grad);
            }
        } اكسر;
        حالة GGML_OP_جوڑ_صفاں: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_اوسط: {
            إذا (src0_needs_grads) {
                ggml_add1_or_سیٹ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } اكسر;
        حالة GGML_OP_REPEAT: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat_پٹھ(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_REPEAT_پٹھ: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_RMS_NORM: {
            إذا (src0_needs_grads) {
                عائم eps;
                حافظہ_نقل(&eps, tensor->op_params, حجم(عائم));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_rms_norm_پٹھ(ctx, grad, src0, eps));
            }
        } اكسر;
        حالة GGML_OP_MUL_MAT: {
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

            إذا (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                بنية ggml_tensor * tmp =
                    ggml_باہرلا_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                إذا (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ثابت int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ثابت size_t nb2 = tmp->nb[2] * nr2;
                    ثابت size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_پٹھ(ctx, tmp, src0);
                }
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, tmp);
            }
            إذا (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_باہرلا_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } اكسر;
        حالة GGML_OP_SCALE: {
            إذا (src0_needs_grads) {
                عائم s;
                حافظہ_نقل(&s, tensor->op_params, حجم(عائم));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } اكسر;
        حالة GGML_OP_سیٹ: {
            ثابت size_t nb1    = ((ثابت int32_t *) tensor->op_params)[0];
            ثابت size_t nb2    = ((ثابت int32_t *) tensor->op_params)[1];
            ثابت size_t nb3    = ((ثابت int32_t *) tensor->op_params)[2];
            ثابت size_t وتھ = ((ثابت int32_t *) tensor->op_params)[3];

            بنية ggml_tensor * tensor_grad_view = عدم;

            إذا (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, وتھ);
            }

            إذا (src0_needs_grads) {
                بنية ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, وتھ, false));
            }

            إذا (src1_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } اكسر;
        حالة GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            إذا (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            إذا (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } اكسر;
        حالة GGML_OP_CONT: {
            // same as cpy
            إذا (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_RESHAPE: {
            إذا (src0_needs_grads) {
                بنية ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } اكسر;
        حالة GGML_OP_VIEW: {
            إذا (src0_needs_grads) {
                size_t وتھ;

                حافظہ_نقل(&وتھ, tensor->op_params, حجم(وتھ));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                إذا (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_ناپ(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_ناپ(src0);
                    GGML_ASSERT(وتھ % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    وتھ = (وتھ / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_سیٹ(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, وتھ);
            }
        } اكسر;
        حالة GGML_OP_ترتیب_بدلو: {
            إذا (src0_needs_grads) {
                ثابت int32_t * axes = (ثابت int32_t *) tensor->op_params;
                ثابت صحيح axis0 = axes[0] & 0x3;
                ثابت صحيح axis1 = axes[1] & 0x3;
                ثابت صحيح axis2 = axes[2] & 0x3;
                ثابت صحيح axis3 = axes[3] & 0x3;
                صحيح axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_ترتیب_بدلو(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } اكسر;
        حالة GGML_OP_TRANSPOSE: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } اكسر;
        حالة GGML_OP_GET_صفاں: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_get_صفاں_پٹھ(ctx, grad, src1, src0));
            }
            إذا (src1_needs_grads) {
                // noop
            }
        } اكسر;
        حالة GGML_OP_DIAG_نقاب_INF: {
            إذا (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ثابت صحيح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_diag_نقاب_zero_impl(ctx, grad, n_past, false));
            }
        } اكسر;
        حالة GGML_OP_DIAG_نقاب_ZERO: {
            إذا (src0_needs_grads) {
                ثابت صحيح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_diag_نقاب_zero_impl(ctx, grad, n_past, false));
            }
        } اكسر;
        حالة GGML_OP_SOFT_MAX: {
            إذا (src0_needs_grads) {
                عائم scale    = 1.0f;
                عائم max_bias = 0.0f;

                حافظہ_نقل(&scale,    (ثابت عائم *) tensor->op_params + 0, حجم(عائم));
                حافظہ_نقل(&max_bias, (ثابت عائم *) tensor->op_params + 1, حجم(عائم));

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_soft_max_ext_پٹھ(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } اكسر;
        حالة GGML_OP_ROPE: {
            إذا (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ثابت صحيح n_dims     = ((ثابت int32_t *) tensor->op_params)[1];
                ثابت صحيح منوال       = ((ثابت int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ثابت صحيح n_ctx_orig = ((ثابت int32_t *) tensor->op_params)[4];
                عائم freq_ادھار, freq_scale, ext_ضارب, attn_ضارب, beta_fast, beta_slow;
                صحيح sections[4] = {0, 0, 0, 0};

                حافظہ_نقل(&freq_ادھار,   (ثابت عائم *) tensor->op_params +  5, حجم(عائم));
                حافظہ_نقل(&freq_scale,  (ثابت عائم *) tensor->op_params +  6, حجم(عائم));
                حافظہ_نقل(&ext_ضارب,  (ثابت عائم *) tensor->op_params +  7, حجم(عائم));
                حافظہ_نقل(&attn_ضارب, (ثابت عائم *) tensor->op_params +  8, حجم(عائم));
                حافظہ_نقل(&beta_fast,   (ثابت عائم *) tensor->op_params +  9, حجم(عائم));
                حافظہ_نقل(&beta_slow,   (ثابت عائم *) tensor->op_params + 10, حجم(عائم));
                حافظہ_نقل(&sections,                    tensor->op_params + 11, حجم(sections));

                بنية ggml_tensor * rope_پٹھ = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_پٹھ(ctx, grad, src1, src2, n_dims,
                        منوال, n_ctx_orig, freq_ادھار, freq_scale, ext_ضارب, attn_ضارب, beta_fast, beta_slow) :
                    ggml_rope_multi_پٹھ(ctx, grad, src1, src2, n_dims, sections,
                        منوال, n_ctx_orig, freq_ادھار, freq_scale, ext_ضارب, attn_ضارب, beta_fast, beta_slow);
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, rope_پٹھ);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } اكسر;
        حالة GGML_OP_IM2_خانہ: {
            إذا (src1_needs_grads) {
                ثابت int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ثابت int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ثابت int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ثابت int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ثابت int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ثابت int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ثابت bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_im2col_پٹھ(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } اكسر;
        حالة GGML_OP_POOL_2D: {
            إذا (src0_needs_grads) {
                ثابت تعداد ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ثابت      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ثابت      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ثابت      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ثابت      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ثابت      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ثابت      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_pool_2d_پٹھ(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } اكسر;
        حالة GGML_OP_WIN_PART:
        حالة GGML_OP_WIN_UNPART:
        حالة GGML_OP_UNARY: {
            بدل (ggml_get_unary_op(tensor)) {
                حالة GGML_UNARY_OP_ABS: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SGN: {
                    // noop
                } اكسر;
                حالة GGML_UNARY_OP_NEG: {
                    إذا (src0_needs_grads) {
                        ggml_sub_or_سیٹ(ctx, cgraph, isrc0, grad);
                    }
                } اكسر;
                حالة GGML_UNARY_OP_قدم: {
                    // noop
                } اكسر;
                حالة GGML_UNARY_OP_RELU: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_قدم(ctx, src0), grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SILU: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_silu_پٹھ(ctx, grad, src0));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_EXP: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_EXPM1: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SOFTPLUS: {
                    إذا (src0_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } اكسر;
                افتراضي: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS: {
            إذا (src0_needs_grads) {
                ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_cross_entropy_loss_پٹھ(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } اكسر;
        حالة GGML_OP_GLU: {
            بدل (ggml_get_glu_op(tensor)) {
                حالة GGML_GLU_OP_SWIGLU: {
                    إذا (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc0, ggml_silu_پٹھ(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    إذا (src1_needs_grads) {
                        ggml_جوڑو_or_سیٹ(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } اكسر;
                افتراضي: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } اكسر;
        حالة GGML_OP_NONE: {
            // noop
        } اكسر;
        حالة GGML_OP_گنتی:
        افتراضي: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ساكن size_t ggml_ویکھو_parents_گراف(بنية ggml_cgraph * cgraph, بنية ggml_tensor * گرہ, bool compute) {
    إذا (گرہ->op != GGML_OP_NONE && compute) {
        گرہ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ثابت size_t گرہ_hash_pos = ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ);
    GGML_ASSERT(گرہ_hash_pos != GGML_HASHSET_بھریا);

    إذا (ggml_bitset_get(cgraph->ویکھیا_hash_سیٹ.ورتیا, گرہ_hash_pos)) {
        // already visited

        إذا (compute) {
            // update the compute flag regardless
            لكل (صحيح i = 0; i < GGML_MAX_SRC; ++i) {
                بنية ggml_tensor * src = گرہ->src[i];
                إذا (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_ویکھو_parents_گراف(cgraph, src, true);
                }
            }
        }

        أرجع گرہ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->ویکھیا_hash_سیٹ.کلیدیں[گرہ_hash_pos] = گرہ;
    ggml_bitset_سیٹ(cgraph->ویکھیا_hash_سیٹ.ورتیا, گرہ_hash_pos);
    cgraph->use_counts[گرہ_hash_pos] = 0;

    لكل (صحيح i = 0; i < GGML_MAX_SRC; ++i) {
        ثابت صحيح k =
            (cgraph->ترتیب == GGML_CGRAPH_EVAL_ترتیب_بایاں_منزل_دایاں) ? i :
            (cgraph->ترتیب == GGML_CGRAPH_EVAL_ترتیب_دایاں_منزل_بایاں) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        بنية ggml_tensor * src = گرہ->src[k];
        إذا (src) {
            ثابت size_t src_hash_pos = ggml_ویکھو_parents_گراف(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    إذا (گرہ->op == GGML_OP_NONE && !(گرہ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->ناپ);

        إذا (لڑی_لمبائی(گرہ->name) == 0) {
            ggml_format_name(گرہ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = گرہ;
        cgraph->n_leafs++;
    } وإلا {
        GGML_ASSERT(cgraph->n_nodes < cgraph->ناپ);

        إذا (لڑی_لمبائی(گرہ->name) == 0) {
            ggml_format_name(گرہ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = گرہ;
        cgraph->n_nodes++;
    }

    أرجع گرہ_hash_pos;
}

ساكن فراغ ggml_build_forward_impl(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor, bool expand, bool compute) {
    إذا (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_گراف_clear(cgraph);
    }

    ثابت صحيح n_old = cgraph->n_nodes;

    ggml_ویکھو_parents_گراف(cgraph, tensor, compute);

    ثابت صحيح n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    إذا (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

بنية ggml_tensor * ggml_build_forward_select(
        بنية ggml_cgraph  * cgraph,
        بنية ggml_tensor ** tensors,
        صحيح                   n_tensors,
        صحيح                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    لكل (صحيح i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    أرجع tensors[idx];
}

فراغ ggml_build_forward_expand(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

فراغ ggml_build_forward_ترتیب(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

فراغ ggml_build_backward_expand(
        بنية ggml_context *  ctx,
        بنية ggml_cgraph  *  cgraph,
        بنية ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ثابت صحيح n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->ویکھیا_hash_سیٹ.ناپ*حجم(بنية ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->ویکھیا_hash_سیٹ.ناپ*حجم(بنية ggml_tensor *));
    bool * grads_needed = صفر_حافظہ_دیو(cgraph->ویکھیا_hash_سیٹ.ناپ, حجم(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        لكل (صحيح i = 0; i < n_nodes_f; ++i) {
            بنية ggml_tensor * گرہ = cgraph->nodes[i];
            any_params = any_params || (گرہ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (گرہ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    لكل (صحيح i = 0; i < n_nodes_f; ++i) {
        بنية ggml_tensor * گرہ = cgraph->nodes[i];

        إذا (گرہ->type == GGML_TYPE_I32) {
            استمر;
        }

        bool گرہ_needs_grad = (گرہ->flags & GGML_TENSOR_FLAG_PARAM) || (گرہ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        بدل (گرہ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            حالة GGML_OP_IM2_خانہ:      // only used for its shape
            حالة GGML_OP_IM2_خانہ_پٹھ: // same as IM2COL
                ignore_src[0] = true;
                اكسر;
            حالة GGML_OP_UNARY: {
                ثابت تعداد ggml_unary_op uop = ggml_get_unary_op(گرہ);
                // SGN and STEP unary ops are piecewise constant
                إذا (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_قدم) {
                    ignore_src[0] = true;
                }
            } اكسر;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            حالة GGML_OP_CPY:           // gradients in CPY target are irrelevant
            حالة GGML_OP_GET_صفاں:      // row indices not differentiable
            حالة GGML_OP_GET_صفاں_پٹھ: // same as for GET_ROWS
            حالة GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                اكسر;

            افتراضي:
                اكسر;
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            إذا (!گرہ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ->src[j])]) {
                استمر;
            }
            GGML_ASSERT(گرہ->src[j]->type == GGML_TYPE_F32 || گرہ->src[j]->type == GGML_TYPE_F16);
            گرہ_needs_grad = true;
            اكسر;
        }
        إذا (!گرہ_needs_grad) {
            استمر;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!گرہ->view_src || گرہ->op == GGML_OP_CPY || گرہ->op == GGML_OP_سیٹ_صفاں || گرہ->op == GGML_OP_VIEW ||
            گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_ترتیب_بدلو || گرہ->op == GGML_OP_TRANSPOSE);

        ثابت size_t ihash = ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ);
        GGML_ASSERT(ihash != GGML_HASHSET_بھریا);
        GGML_ASSERT(ggml_bitset_get(cgraph->ویکھیا_hash_سیٹ.ورتیا, ihash));
        إذا (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } وإلا إذا (گرہ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, گرہ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    لكل (صحيح i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    حرر(grads_needed);
}

ساكن فراغ * incr_ptr_aligned(فراغ ** p, size_t ناپ, size_t align) {
    فراغ * ptr = *p;
    ptr = (فراغ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (فراغ *) ((حرف *) ptr + ناپ);
    أرجع ptr;
}

ساكن size_t ggml_گراف_nbytes(size_t ناپ, bool grads) {
    size_t hash_ناپ = ggml_hash_ناپ(ناپ * 2);
    فراغ * p = (حرف *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, حجم(بنية ggml_cgraph), 1);
    incr_ptr_aligned(&p, ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_ناپ * حجم(int32_t), حجم(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // hash keys
    إذا (grads) {
        incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_ناپ(hash_ناپ) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    أرجع nbytes - 1024;
}

size_t ggml_گراف_overhead_custom(size_t ناپ, bool grads) {
    أرجع GGML_OBJECT_ناپ + GGML_PAD(ggml_گراف_nbytes(ناپ, grads), GGML_MEM_ALIGN);
}

size_t ggml_گراف_overhead(فراغ) {
    أرجع ggml_گراف_overhead_custom(GGML_DEFAULT_گراف_ناپ, false);
}

بنية ggml_cgraph * ggml_new_گراف_custom(بنية ggml_context * ctx, size_t ناپ, bool grads) {
    ثابت size_t obj_ناپ = ggml_گراف_nbytes(ناپ, grads);
    بنية ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_گراف, obj_ناپ);
    بنية ggml_cgraph * cgraph = (بنية ggml_cgraph *) ((حرف *) ctx->mem_بفر + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_ناپ = ggml_hash_ناپ(ناپ * 2);

    فراغ * p = cgraph + 1;

    بنية ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, ناپ      * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    بنية ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, ناپ      * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_ناپ * حجم(int32_t), حجم(int32_t));
    بنية ggml_tensor ** hash_کلیدیں_ptr  =         incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    بنية ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)) : عدم;
    بنية ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_ناپ * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)) : عدم;

    ggml_bitset_t * hash_ورتیا = incr_ptr_aligned(&p, ggml_bitset_ناپ(hash_ناپ) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    أكد(obj_ناپ == (size_t)((حرف *)p - (حرف *)cgraph));

    *cgraph = (بنية ggml_cgraph) {
        /*.size         =*/ ناپ,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_ناپ, hash_ورتیا, hash_کلیدیں_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ترتیب_بایاں_منزل_دایاں,
        /*.uid          =*/ 0,
    };

    ggml_hash_سیٹ_reset(&cgraph->ویکھیا_hash_سیٹ);
    إذا (grads) {
        memset(cgraph->grads,     0, hash_ناپ*حجم(بنية ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_ناپ*حجم(بنية ggml_tensor *));
    }

    أرجع cgraph;
}

بنية ggml_cgraph * ggml_new_گراف(بنية ggml_context * ctx) {
    أرجع ggml_new_گراف_custom(ctx, GGML_DEFAULT_گراف_ناپ, false);
}

بنية ggml_cgraph ggml_گراف_view(بنية ggml_cgraph * cgraph0, صحيح i0, صحيح i1) {
    بنية ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ عدم, // gradients would need visited_hash_set
        /*.grad_accs        =*/ عدم,
        /*.leafs            =*/ عدم,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->ویکھیا_hash_سیٹ,
        /*.order            =*/ cgraph0->ترتیب,
        /*.uid              =*/ 0
    };

    أرجع cgraph;
}

فراغ ggml_گراف_cpy(بنية ggml_cgraph * src, بنية ggml_cgraph * dst) {
    GGML_ASSERT(dst->ناپ >= src->n_leafs);
    GGML_ASSERT(dst->ناپ >= src->n_nodes);
    GGML_ASSERT(dst->ویکھیا_hash_سیٹ.ناپ >= src->ویکھیا_hash_سیٹ.ناپ);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->ترتیب   = src->ترتیب;

    لكل (صحيح i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    لكل (صحيح i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    لكل (size_t i = 0; i < src->ویکھیا_hash_سیٹ.ناپ; ++i) {
        // copy all hashset keys (tensors) that are in use
        إذا (ggml_bitset_get(src->ویکھیا_hash_سیٹ.ورتیا, i)) {
            size_t new_hash_pos = ggml_hash_داخل_کرو(&dst->ویکھیا_hash_سیٹ, src->ویکھیا_hash_سیٹ.کلیدیں[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    إذا (dst->grads) {
        memset(dst->grads,     0, dst->ویکھیا_hash_سیٹ.ناپ*حجم(بنية ggml_tensor *));
        memset(dst->grad_accs, 0, dst->ویکھیا_hash_سیٹ.ناپ*حجم(بنية ggml_tensor *));
    }
    إذا (src->grads) {
        GGML_ASSERT(dst->grads     != عدم);
        GGML_ASSERT(dst->grad_accs != عدم);
        لكل (صحيح i = 0; i < src->n_nodes; ++i) {
            ثابت size_t igrad_src = ggml_hash_لبھو(&src->ویکھیا_hash_سیٹ, src->nodes[i]);
            ثابت size_t igrad_dst = ggml_hash_لبھو(&dst->ویکھیا_hash_سیٹ, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_بھریا);
            GGML_ASSERT(ggml_bitset_get(src->ویکھیا_hash_سیٹ.ورتیا, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_بھریا);
            GGML_ASSERT(ggml_bitset_get(dst->ویکھیا_hash_سیٹ.ورتیا, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

بنية ggml_cgraph * ggml_گراف_dup(بنية ggml_context * ctx, بنية ggml_cgraph * cgraph, bool force_grads) {
    بنية ggml_cgraph * نتیجہ = ggml_new_گراف_custom(ctx, cgraph->ناپ, cgraph->grads || force_grads);
    ggml_گراف_cpy(cgraph, نتیجہ);
    أرجع نتیجہ;
}

بنية ggml_tensor * ggml_سیٹ_zero(بنية ggml_tensor * tensor) {
    إذا (ggml_is_خالی_تھاواں(tensor)) {
        أرجع tensor;
    }
    إذا (tensor->بفر) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } وإلا {
        GGML_ASSERT(tensor->انکڑے);
        memset(tensor->انکڑے, 0, ggml_nbytes(tensor));
    }
    أرجع tensor;
}

فراغ ggml_گراف_reset(بنية ggml_cgraph * cgraph) {
    إذا (!cgraph) {
        أرجع;
    }
    GGML_ASSERT(cgraph->grads != عدم);

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * گرہ     = cgraph->nodes[i];
        بنية ggml_tensor * grad_acc = ggml_گراف_get_grad_acc(cgraph, گرہ);

        إذا (گرہ->op == GGML_OP_OPT_قدم_ADAMW) {
            // clear momenta
            ggml_سیٹ_zero(گرہ->src[2]);
            ggml_سیٹ_zero(گرہ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        إذا (grad_acc) {
            إذا (گرہ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ثابت عائم onef = 1.0f;
                إذا (grad_acc->بفر) {
                    ggml_backend_tensor_سیٹ(grad_acc, &onef, 0, حجم(عائم));
                } وإلا {
                    GGML_ASSERT(grad_acc->انکڑے);
                    *((عائم *) grad_acc->انکڑے) = onef;
                }
            } وإلا {
                ggml_سیٹ_zero(grad_acc);
            }
        }
    }
}

فراغ ggml_گراف_clear(بنية ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_سیٹ_reset(&cgraph->ویکھیا_hash_سیٹ);
}

صحيح ggml_گراف_ناپ(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->ناپ;
}

بنية ggml_tensor * ggml_گراف_گرہ(بنية ggml_cgraph * cgraph, صحيح i) {
    إذا (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        أرجع cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    أرجع cgraph->nodes[i];
}

بنية ggml_tensor ** ggml_گراف_nodes(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->nodes;
}

صحيح ggml_گراف_n_nodes(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->n_nodes;
}

فراغ ggml_گراف_جوڑو_گرہ(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->ناپ > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

بنية ggml_tensor * ggml_گراف_get_tensor(ثابت بنية ggml_cgraph * cgraph, ثابت حرف * name) {
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        بنية ggml_tensor * پتی = cgraph->leafs[i];

        إذا (strcmp(پتی->name, name) == 0) {
            أرجع پتی;
        }
    }

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = cgraph->nodes[i];

        إذا (strcmp(گرہ->name, name) == 0) {
            أرجع گرہ;
        }
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_گراف_get_grad(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * گرہ) {
    ثابت size_t igrad = ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ);
    أرجع igrad != GGML_HASHSET_بھریا && ggml_bitset_get(cgraph->ویکھیا_hash_سیٹ.ورتیا, igrad) && cgraph->grads ? cgraph->grads[igrad] : عدم;
}

بنية ggml_tensor * ggml_گراف_get_grad_acc(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * گرہ) {
    ثابت size_t igrad = ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ);
    أرجع igrad != GGML_HASHSET_بھریا && ggml_bitset_get(cgraph->ویکھیا_hash_سیٹ.ورتیا, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : عدم;
}

فراغ ggml_گراف_print(ثابت بنية ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                گرہ->ne[0], گرہ->ne[1], گرہ->ne[2],
                ggml_op_name(گرہ->op), (گرہ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_گراف_get_grad(cgraph, گرہ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        بنية ggml_tensor * گرہ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                گرہ->ne[0], گرہ->ne[1],
                ggml_op_name(گرہ->op),
                ggml_get_name(گرہ));
    }

    GGML_LOG_INFO("========================================\n");
}

ساكن صحيح ggml_گرہ_list_لبھو_tensor(ثابت بنية ggml_cgraph * cgraph,
                                      ثابت صحيح *                idxs,
                                      صحيح                        گنتی,
                                      ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    لكل (صحيح i = 0; i < گنتی; ++i) {
        ثابت صحيح گرہ_idx = idxs[i];

        إذا (گرہ_idx >= cgraph->n_nodes) {
            أرجع -1;
        }
        إذا (cgraph->nodes[گرہ_idx] == tensor) {
            أرجع i;
        }
    }
    أرجع -1;
}

ساكن bool ggml_is_constant(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->بفر != عدم && ggml_backend_بفر_get_usage(tensor->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ثابت بنية ggml_cgraph * cgraph,
                                ثابت صحيح *                گرہ_idxs,
                                صحيح                        گنتی,
                                ثابت تعداد ggml_op *       ops,
                                ثابت صحيح *                outputs,
                                صحيح                        عدد_outputs) {
    GGML_ASSERT(outputs && عدد_outputs > 0);

    لكل (صحيح i = 0; i < گنتی; ++i) {
        إذا (گرہ_idxs[i] >= cgraph->n_nodes) {
            أرجع false;
        }

        ثابت بنية ggml_tensor * گرہ = cgraph->nodes[گرہ_idxs[i]];

        إذا (گرہ->op != ops[i]) {
            أرجع false;
        }

        إذا ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            أرجع false;
        }

        إذا (ggml_گرہ_list_لبھو_tensor(cgraph, outputs, عدد_outputs, گرہ) != -1) {
            استمر;
        }

        إذا (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            أرجع false;
        }

        صحيح subgraph_ورتدا = 0;
        لكل (صحيح j = i + 1; j < گنتی; ++j) {
            ثابت بنية ggml_tensor * other_گرہ = cgraph->nodes[گرہ_idxs[j]];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                إذا (other_گرہ->src[src_idx] == گرہ) {
                    subgraph_ورتدا++;
                }
            }
        }

        إذا (subgraph_ورتدا != ggml_گرہ_get_use_گنتی(cgraph, گرہ_idxs[i])) {
            أرجع false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        بنية ggml_tensor * view_src = گرہ->view_src;
        طالما (view_src) {
            إذا (ggml_گرہ_list_لبھو_tensor(cgraph, گرہ_idxs, گنتی, view_src) == -1 && !ggml_is_constant(view_src)) {
                أرجع false;
            }
            view_src = view_src->view_src;
        }
    }

    أرجع true;
}

// check if node is part of the graph
ساكن bool ggml_گراف_لبھو(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * گرہ) {
    إذا (cgraph == عدم) {
        أرجع true;
    }

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i] == گرہ) {
            أرجع true;
        }
    }

    أرجع false;
}

ساكن بنية ggml_tensor * ggml_گراف_get_ماپا(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * گرہ) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * ماپا = cgraph->nodes[i];
        بنية ggml_tensor * grad = ggml_گراف_get_grad(cgraph, ماپا);

        إذا (grad == گرہ) {
            أرجع ماپا;
        }
    }

    أرجع عدم;
}

ساكن فراغ ggml_گراف_dump_dot_گرہ_کنارا(FILE * fp, ثابت بنية ggml_cgraph * gb, بنية ggml_tensor * گرہ, بنية ggml_tensor * ماپا, ثابت حرف * label)  {
    بنية ggml_tensor * gparent = ggml_گراف_get_ماپا(gb, گرہ);
    بنية ggml_tensor * gparent0 = ggml_گراف_get_ماپا(gb, ماپا);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (فراغ *) gparent0 : (فراغ *) ماپا,
            gparent ? (فراغ *) gparent : (فراغ *) گرہ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ساكن فراغ ggml_گراف_dump_dot_پتی_کنارا(FILE * fp, بنية ggml_tensor * گرہ, بنية ggml_tensor * ماپا, ثابت حرف * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (فراغ *) ماپا,
            (فراغ *) گرہ,
            label);
}

فراغ ggml_گراف_dump_dot(ثابت بنية ggml_cgraph * gb, ثابت بنية ggml_cgraph * cgraph, ثابت حرف * filename) {
    حرف color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    لكل (صحيح i = 0; i < gb->n_nodes; i++) {
        بنية ggml_tensor * گرہ = gb->nodes[i];
        بنية ggml_tensor * grad = ggml_گراف_get_grad(gb, گرہ);

        إذا (ggml_گراف_get_ماپا(gb, گرہ) != عدم) {
            استمر;
        }

        إذا (گرہ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, حجم(color), "yellow");
        } وإلا إذا (grad) {
            إذا (ggml_گراف_لبھو(cgraph, گرہ)) {
                snprintf(color, حجم(color), "green");
            } وإلا {
                snprintf(color, حجم(color), "lightblue");
            }
        } وإلا {
            snprintf(color, حجم(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (فراغ *) گرہ, color);

        إذا (لڑی_لمبائی(گرہ->name) > 0) {
            fprintf(fp, "%s (%s)|", گرہ->name, ggml_type_name(گرہ->type));
        } وإلا {
            fprintf(fp, "(%s)|", ggml_type_name(گرہ->type));
        }

        إذا (ggml_is_میٹرکس(گرہ)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, گرہ->ne[0], گرہ->ne[1], ggml_op_علامت(گرہ->op));
        } وإلا {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, گرہ->ne[0], گرہ->ne[1], گرہ->ne[2], ggml_op_علامت(گرہ->op));
        }

        إذا (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_علامت(grad->op));
        } وإلا {
            fprintf(fp, "\"; ]\n");
        }
    }

    لكل (صحيح i = 0; i < gb->n_leafs; i++) {
        بنية ggml_tensor * گرہ = gb->leafs[i];

        snprintf(color, حجم(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (فراغ *) گرہ, color);

        إذا (لڑی_لمبائی(گرہ->name) > 0) {
            fprintf(fp, "%s (%s)|", گرہ->name, ggml_type_name(گرہ->type));
        } وإلا {
            fprintf(fp, "(%s)|", ggml_type_name(گرہ->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, گرہ->ne[0], گرہ->ne[1]);
        إذا (ggml_nelements(گرہ) < 5 && گرہ->انکڑے != عدم) {
            fprintf(fp, " | (");
            لكل (صحيح j = 0; j < ggml_nelements(گرہ); j++) {
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
                إذا (j < ggml_nelements(گرہ) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    لكل (صحيح i = 0; i < gb->n_nodes; i++) {
        بنية ggml_tensor * گرہ = gb->nodes[i];

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (گرہ->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_گراف_dump_dot_گرہ_کنارا(fp, gb, گرہ, گرہ->src[j], label);
            }
        }
    }

    لكل (صحيح i = 0; i < gb->n_leafs; i++) {
        بنية ggml_tensor * گرہ = gb->leafs[i];

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (گرہ->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_گراف_dump_dot_پتی_کنارا(fp, گرہ, گرہ->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_سیٹ_داخلہ(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_داخلہ;
}

فراغ ggml_سیٹ_output(بنية ggml_tensor * tensor) {
    لكل (بنية ggml_tensor * cur = tensor; cur != عدم; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

فراغ ggml_سیٹ_param(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

فراغ ggml_سیٹ_loss(بنية ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_quantize_init(تعداد ggml_type type) {
    ggml_critical_section_شروع();

    بدل (type) {
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); اكسر;
        حالة GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); اكسر;
        حالة GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); اكسر;
        افتراضي: // nothing
            اكسر;
    }

    ggml_critical_section_سرا();
}

فراغ ggml_quantize_free(فراغ) {
    ggml_critical_section_شروع();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_سرا();
}

bool ggml_quantize_requires_imatrix(تعداد ggml_type type) {
    أرجع
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        تعداد ggml_type   type,
           ثابت عائم * src,
                  فراغ * dst,
               int64_t   شروع,
               int64_t   nrows,
               int64_t   n_فی_صف,
           ثابت عائم * imatrix) {
    ثابت int64_t n = nrows * n_فی_صف;

    إذا (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != عدم);
    }

    GGML_ASSERT(شروع % type_traits[type].blck_ناپ == 0);
    GGML_ASSERT(شروع % n_فی_صف == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ثابت size_t شروع_صف = شروع / n_فی_صف;
    ثابت size_t صف_ناپ  = ggml_صف_ناپ(type, n_فی_صف);

    size_t نتیجہ = 0;

    بدل (type) {
        حالة GGML_TYPE_Q1_0:    نتیجہ = quantize_q1_0   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q2_0:    نتیجہ = quantize_q2_0   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q4_0:    نتیجہ = quantize_q4_0   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q4_1:    نتیجہ = quantize_q4_1   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q5_0:    نتیجہ = quantize_q5_0   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q5_1:    نتیجہ = quantize_q5_1   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q8_0:    نتیجہ = quantize_q8_0   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_MXFP4:   نتیجہ = quantize_mxfp4  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_NVFP4:   نتیجہ = quantize_nvfp4  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q2_K:    نتیجہ = quantize_q2_K   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q3_K:    نتیجہ = quantize_q3_K   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q4_K:    نتیجہ = quantize_q4_K   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q5_K:    نتیجہ = quantize_q5_K   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_Q6_K:    نتیجہ = quantize_q6_K   (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_TQ1_0:   نتیجہ = quantize_tq1_0  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_TQ2_0:   نتیجہ = quantize_tq2_0  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_XXS: نتیجہ = quantize_iq2_xxs(src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_XS:  نتیجہ = quantize_iq2_xs (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ3_XXS: نتیجہ = quantize_iq3_xxs(src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ3_S:   نتیجہ = quantize_iq3_s  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_S:   نتیجہ = quantize_iq2_s  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ1_S:   نتیجہ = quantize_iq1_s  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ1_M:   نتیجہ = quantize_iq1_m  (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ4_NL:  نتیجہ = quantize_iq4_nl (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_IQ4_XS:  نتیجہ = quantize_iq4_xs (src + شروع, (حرف *) dst + شروع_صف * صف_ناپ, nrows, n_فی_صف, imatrix); اكسر;
        حالة GGML_TYPE_F16:
            {
                size_t elemsize = حجم(ggml_fp16_t);
                ggml_fp32_منزل_fp16_صف(src + شروع, (ggml_fp16_t *)dst + شروع, n);
                نتیجہ = n * elemsize;
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                size_t elemsize = حجم(ggml_bf16_t);
                ggml_fp32_منزل_bf16_صف_ref(src + شروع, (ggml_bf16_t *)dst + شروع, n);
                نتیجہ = n * elemsize;
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                size_t elemsize = حجم(عائم);
                نتیجہ = n * elemsize;
                حافظہ_نقل((uint8_t *)dst + شروع * elemsize, src + شروع, نتیجہ);
            } اكسر;
        افتراضي:
            أكد(false);
    }

    GGML_ASSERT(نتیجہ == nrows * صف_ناپ);

    أرجع نتیجہ;
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_log_get(ggml_log_callback * log_callback, فراغ ** user_انکڑے) {
    *log_callback = g_logger_state.log_callback;
    *user_انکڑے    = g_logger_state.log_callback_user_انکڑے;
}

فراغ ggml_log_سیٹ(ggml_log_callback log_callback, فراغ * user_انکڑے) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_انکڑے = user_انکڑے;
}

فراغ ggml_threadpool_params_init(بنية ggml_threadpool_params * p, صحيح n_دھاگے) {
    p->n_دھاگے  = n_دھاگے;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_دھاگے); // all-zero means use the default affinity (usually inherited)
}

بنية ggml_threadpool_params ggml_threadpool_params_default(صحيح n_دھاگے) {
    بنية ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_دھاگے);
    أرجع p;
}

bool ggml_threadpool_params_میل(ثابت بنية ggml_threadpool_params * p0, ثابت بنية ggml_threadpool_params * p1) {
    إذا (p0->n_دھاگے  != p1->n_دھاگے  ) أرجع false;
    إذا (p0->prio       != p1->prio       ) أرجع false;
    إذا (p0->poll       != p1->poll       ) أرجع false;
    إذا (p0->strict_cpu != p1->strict_cpu ) أرجع false;
    أرجع memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_دھاگے) == 0;
}
