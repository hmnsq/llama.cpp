#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_MATH_ਨਿਰਧਾਰਤ_ਕਰਦਾ // For M_PI on MSVC

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-version.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-threading.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

// FIXME: required here for quantization functions
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_HBM
#ਅਨ੍ਤਰ੍ਭਾਵ <hbwmalloc.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
#ਅਨ੍ਤਰ੍ਭਾਵ <malloc.h> // using malloc.h with MSC/MINGW
#ਜਾਂਜੇ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ਅਨ੍ਤਰ੍ਭਾਵ <alloca.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <errno.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <inttypes.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <signal.h>
#ਜੇ defined(__gnu_linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <syscall.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <mach/mach.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <TargetConditionals.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(_WIN32)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WIN32_LEAN_AND_ਔਸਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NOMINMAX
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

uint64_t ggml_ਗ੍ਰਾਫ_ਅਗਲਾ_uid(ਖਾਲੀ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _MSC_VER
#ਜੇ defined(_WIN32)
    ਸਥਿਰ ਅਸਥਿਰ LONG counter = 1;
    ਵਾਪਸ (uint64_t) InterlockedIncrement(&counter) - 1;
#ਨਹੀਂਤਾਂ
    ਸਥਿਰ ਅਸਥਿਰ ਲੰਮਾ ਲੰਮਾ counter = 1;
    ਵਾਪਸ (uint64_t) _InterlockedIncrement64(&counter) - 1;
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
    ਸਥਿਰ uint64_t counter = 1;
    ਵਾਪਸ __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#ਜੇ_ਅੰਤ
}

// Needed for ggml_fp32_to_bf16_row()
#ਜੇ defined(__AVX512BF16__)
#ਜੇ defined(_MSC_VER)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512i(p) p
#ਨਹੀਂਤਾਂ
#ਅਨ੍ਤਰ੍ਭਾਵ <immintrin.h>
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512i(p) (__m512i)(p)
#ਜੇ_ਅੰਤ // defined(_MSC_VER)
#ਜੇ_ਅੰਤ // defined(__AVX512BF16__)

#ਜੇ defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !ਨਿਸ਼ਾਨਾ_OS_TV && !ਨਿਸ਼ਾਨਾ_OS_WATCH)

#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/wait.h>
#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/prctl.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__ANDROID__)
#ਅਨ੍ਤਰ੍ਭਾਵ <unwind.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dlfcn.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>

ਬਣਤਰ backtrace_state {
    ਖਾਲੀ ** current;
    ਖਾਲੀ ** ਸਿਰਾ;
};

ਸਥਿਰ Unwind_Reason_ਕੋਡ unwind_callback(ਬਣਤਰ _Unwind_Context* context, ਖਾਲੀ* arg) {
    ਬਣਤਰ backtrace_state * state = (ਬਣਤਰ backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    ਜੇ (pc) {
        ਜੇ (state->current == state->ਸਿਰਾ) {
            ਵਾਪਸ URC_ਸਿਰਾ_ਦਾ_ਢੇਰ;
        } ਨਹੀਂਤਾਂ {
            *state->current++ = (ਖਾਲੀ*)pc;
        }
    }
    ਵਾਪਸ _URC_NO_REASON;
}

ਸਥਿਰ ਖਾਲੀ ggml_print_backtrace_symbols(ਖਾਲੀ) {
    ਅਚਲ ਪੂਰਨ max = 100;
    ਖਾਲੀ* ਬਫਰ[max];

    ਬਣਤਰ backtrace_state state = {ਬਫਰ, ਬਫਰ + max};
    _Unwind_Backtrace(unwind_callback, &state);

    ਪੂਰਨ ਗਿਣਤੀ_2 = state.current - ਬਫਰ;

    ਲਈ (ਪੂਰਨ idx = 0; idx < ਗਿਣਤੀ_2; ++idx) {
        ਅਚਲ ਖਾਲੀ * addr = ਬਫਰ[idx];
        ਅਚਲ ਅੱਖਰ * ਚਿੰਨ੍ਹ = "";

        Dl_info info;
        ਜੇ (dladdr(addr, &info) && info.dli_sname) {
            ਚਿੰਨ੍ਹ = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, ਚਿੰਨ੍ਹ);
    }
}
#ਜਾਂਜੇ defined(__linux__) && defined(__GLIBC__)
#ਅਨ੍ਤਰ੍ਭਾਵ <execinfo.h>
ਸਥਿਰ ਖਾਲੀ ggml_print_backtrace_symbols(ਖਾਲੀ) {
    ਖਾਲੀ * trace[100];
    ਪੂਰਨ nptrs = backtrace(trace, ਆਕਾਰ(trace)/ਆਕਾਰ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ਜਾਂਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <execinfo.h>
ਸਥਿਰ ਖਾਲੀ ggml_print_backtrace_symbols(ਖਾਲੀ) {
    ਖਾਲੀ * trace[100];
    ਪੂਰਨ nptrs = backtrace(trace, ਆਕਾਰ(trace)/ਆਕਾਰ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਖਾਲੀ ggml_print_backtrace_symbols(ਖਾਲੀ) {
    // platform not supported
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_print_backtrace(ਖਾਲੀ) {
    ਅਚਲ ਅੱਖਰ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ਜੇ (GGML_NO_BACKTRACE) {
        ਵਾਪਸ;
    }
#ਜੇ defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ਅਚਲ ਅੱਖਰ * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    ਜੇ (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ਵਾਪਸ;
    }
#ਜੇ_ਅੰਤ
#ਜੇ defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t ਮਾਪ = 0;
    ਅੱਖਰ * ਸਤਰ = ਨਿਰਾ;
    ssize_t ਲੰਮਾਈ = 0;
    ਜਦੋਂ ((ਲੰਮਾਈ = getline(&ਸਤਰ, &ਮਾਪ, f)) > 0) {
        ਜੇ (!ਲੜੀ_ਤੁਲਨਾ(ਸਤਰ, "TracerPid:", ਆਕਾਰ("TracerPid:") - 1) &&
            (ਲੰਮਾਈ != ਆਕਾਰ("TracerPid:\t0\n") - 1 || ਸਤਰ[ਲੰਮਾਈ - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            ਮੁਕਤ(ਸਤਰ);
            fclose(f);
            ਵਾਪਸ;
        }
    }
    ਮੁਕਤ(ਸਤਰ);
    fclose(f);
    ਪੂਰਨ ਤਾਲਾ[2] = { -1, -1 };
    (ਖਾਲੀ) !pipe(ਤਾਲਾ); // Don't start gdb until after PR_SET_PTRACER
#ਜੇ_ਅੰਤ
    ਅਚਲ ਪੂਰਨ ਮਾਪਾ_pid = getpid();
    ਅਚਲ ਪੂਰਨ ਬੱਚਾ_pid = fork();
    ਜੇ (ਬੱਚਾ_pid < 0) { // error
#ਜੇ defined(__linux__)
        close(ਤਾਲਾ[1]);
        close(ਤਾਲਾ[0]);
#ਜੇ_ਅੰਤ
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ ਜੇ (ਬੱਚਾ_pid == 0) { //ਬੱਚਾ
        ਅੱਖਰ attach[32];
        snprintf(attach, ਆਕਾਰ(attach), "attach %d", ਮਾਪਾ_pid);
#ਜੇ defined(__linux__)
        close(ਤਾਲਾ[1]);
        (ਖਾਲੀ) !ਪਠ(ਤਾਲਾ[0], ਤਾਲਾ, 1);
        close(ਤਾਲਾ[0]);
#ਜੇ_ਅੰਤ
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (ਅੱਖਰ *) ਨਿਰਾ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[ਆਕਾਰ("attach ") - 1],
            (ਅੱਖਰ *) ਨਿਰਾ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } ਨਹੀਂਤਾਂ { //ਮਾਪਾ
#ਜੇ defined(__linux__)
        prctl(PR_ਸੈੱਟ_PTRACER, ਬੱਚਾ_pid);
        close(ਤਾਲਾ[1]);
        close(ਤਾਲਾ[0]);
#ਜੇ_ਅੰਤ
        waitpid(ਬੱਚਾ_pid, ਨਿਰਾ, 0);
    }
}
#ਨਹੀਂਤਾਂ
ਖਾਲੀ ggml_print_backtrace(ਖਾਲੀ) {
    // platform not supported
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ggml_abort_callback_t g_abort_callback = ਨਿਰਾ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_ਸੈੱਟ_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ਵਾਪਸ ret_val;
}

ਖਾਲੀ ggml_abort(ਅਚਲ ਅੱਖਰ * file, ਪੂਰਨ ਸਤਰ, ਅਚਲ ਅੱਖਰ * fmt, ...) {
    fflush(stdout);

    ਅੱਖਰ ਸੁਨੇਹਾ[2048];
    ਪੂਰਨ ਵਿੱਥ = snprintf(ਸੁਨੇਹਾ, ਆਕਾਰ(ਸੁਨੇਹਾ), "%s:%d: ", file, ਸਤਰ);

    va_list ਦਲੀਲਾਂ_2;
    va_start(ਦਲੀਲਾਂ_2, fmt);
    vsnprintf(ਸੁਨੇਹਾ + ਵਿੱਥ, ਆਕਾਰ(ਸੁਨੇਹਾ) - ਵਿੱਥ, fmt, ਦਲੀਲਾਂ_2);
    va_end(ਦਲੀਲਾਂ_2);

    ਜੇ (g_abort_callback) {
        g_abort_callback(ਸੁਨੇਹਾ);
    } ਨਹੀਂਤਾਂ {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", ਸੁਨੇਹਾ);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

ਬਣਤਰ ggml_logger_state {
    ggml_log_callback log_callback;
    ਖਾਲੀ * log_callback_user_ਅੰਕੜੇ;
};
ਸਥਿਰ ਬਣਤਰ ggml_logger_state g_logger_state = {ggml_log_callback_default, ਨਿਰਾ};

ਸਥਿਰ ਖਾਲੀ ggml_log_ਅੰਦਰੂਨੀ_v(ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * format, va_list ਦਲੀਲਾਂ_2) {
    ਜੇ (format == ਨਿਰਾ) {
        ਵਾਪਸ;
    }
    va_list ਦਲੀਲਾਂ_copy;
    va_copy(ਦਲੀਲਾਂ_copy, ਦਲੀਲਾਂ_2);
    ਅੱਖਰ ਬਫਰ[128];
    ਪੂਰਨ len = vsnprintf(ਬਫਰ, 128, format, ਦਲੀਲਾਂ_2);
    ਜੇ (len < 128) {
        g_logger_state.log_callback(level, ਬਫਰ, g_logger_state.log_callback_user_ਅੰਕੜੇ);
    } ਨਹੀਂਤਾਂ {
        ਅੱਖਰ * buffer2 = (ਅੱਖਰ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(len + 1, ਆਕਾਰ(ਅੱਖਰ));
        vsnprintf(buffer2, len + 1, format, ਦਲੀਲਾਂ_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_ਅੰਕੜੇ);
        ਮੁਕਤ(buffer2);
    }
    va_end(ਦਲੀਲਾਂ_copy);
}

ਖਾਲੀ ggml_log_ਅੰਦਰੂਨੀ(ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * format, ...) {
    va_list ਦਲੀਲਾਂ_2;
    va_start(ਦਲੀਲਾਂ_2, format);
    ggml_log_ਅੰਦਰੂਨੀ_v(level, format, ਦਲੀਲਾਂ_2);
    va_end(ਦਲੀਲਾਂ_2);
}

ਖਾਲੀ ggml_log_callback_default(ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * ਪਾਠ_2, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    (ਖਾਲੀ) level;
    (ਖਾਲੀ) user_ਅੰਕੜੇ;
    fputs(ਪਾਠ_2, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#ਜੇ_ਅੰਤ


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#ਜੇ defined(__has_feature)
#ਜੇ __has_feature(thread_sanitizer)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_TSAN_ENABLED 1
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
#ਜੇ defined(__SANITIZE_THREAD__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_TSAN_ENABLED 1
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

ਖਾਲੀ * ggml_aligned_malloc(size_t ਮਾਪ) {
#ਜੇ defined(__s390x__)
    ਅਚਲ ਪੂਰਨ alignment = 256;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ alignment = 64;
#ਜੇ_ਅੰਤ

#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
    ਵਾਪਸ _aligned_malloc(ਮਾਪ, alignment);
#ਨਹੀਂਤਾਂ
    ਜੇ (ਮਾਪ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਖਾਲੀ * aligned_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
  #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_HBM
    ਪੂਰਨ ਨਤੀਜਾ = hbw_posix_memalign(&aligned_ਸਮ੍ਰਿਤੀ, alignment, ਮਾਪ);
  #ਜਾਂਜੇ ਨਿਸ਼ਾਨਾ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_ਅਣਵਰਤਿਆ(alignment);
    kern_return_t alloc_ਸਥਿਤੀ = vm_allocate((vm_map_t) mach_task_self(), (vm_ਪਤਾ_t *) &aligned_ਸਮ੍ਰਿਤੀ, ਮਾਪ, VM_FLAGS_ANYWHERE);
    ਪੂਰਨ ਨਤੀਜਾ = EFAULT;
    ਚੋਣ (alloc_ਸਥਿਤੀ) {
        ਹਾਲਤ KERN_SUCCESS:
            ਨਤੀਜਾ = 0;
            ਰੋਕੋ;
        ਹਾਲਤ KERN_INVALID_ਪਤਾ:
            ਨਤੀਜਾ = EINVAL;
            ਰੋਕੋ;
        ਹਾਲਤ KERN_NO_SPACE:
            ਨਤੀਜਾ = ENOMEM;
            ਰੋਕੋ;
        ਮੂਲ:
            ਨਤੀਜਾ = EFAULT;
            ਰੋਕੋ;
    }
  #ਨਹੀਂਤਾਂ
    ਪੂਰਨ ਨਤੀਜਾ = posix_memalign(&aligned_ਸਮ੍ਰਿਤੀ, alignment, ਮਾਪ);
  #ਜੇ_ਅੰਤ
    ਜੇ (ਨਤੀਜਾ != 0) {
        // Handle allocation failure
        ਅਚਲ ਅੱਖਰ *error_desc = "unknown allocation error";
        ਚੋਣ (ਨਤੀਜਾ) {
            ਹਾਲਤ EINVAL:
                error_desc = "invalid alignment value";
                ਰੋਕੋ;
            ਹਾਲਤ ENOMEM:
                error_desc = "insufficient memory";
                ਰੋਕੋ;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, ਮਾਪ/(1024.0*1024.0));
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ aligned_ਸਮ੍ਰਿਤੀ;
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_aligned_free(ਖਾਲੀ * ptr, size_t ਮਾਪ) {
    GGML_ਅਣਵਰਤਿਆ(ਮਾਪ);
#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#ਜਾਂਜੇ GGML_USE_CPU_HBM
    ਜੇ (ptr != ਨਿਰਾ) {
        hbw_free(ptr);
    }
#ਜਾਂਜੇ ਨਿਸ਼ਾਨਾ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    ਜੇ (ptr != ਨਿਰਾ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_ਪਤਾ_t)ptr, ਮਾਪ);
    }
#ਨਹੀਂਤਾਂ
    ਮੁਕਤ(ptr);
#ਜੇ_ਅੰਤ
}


ਸਤਰਅੰਦਰ ਸਥਿਰ ਖਾਲੀ * ggml_malloc(size_t ਮਾਪ) {
    ਜੇ (ਮਾਪ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਖਾਲੀ * ਨਤੀਜਾ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਮਾਪ);
    ਜੇ (ਨਤੀਜਾ == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ਮਾਪ/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

// calloc
ਸਤਰਅੰਦਰ ਸਥਿਰ ਖਾਲੀ * ggml_calloc(size_t ਅੰਕ, size_t ਮਾਪ) {
    ਜੇ (ਅੰਕ == 0 || ਮਾਪ == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਖਾਲੀ * ਨਤੀਜਾ = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਅੰਕ, ਮਾਪ);
    ਜੇ (ਨਤੀਜਾ == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ਮਾਪ/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_MALLOC(ਮਾਪ)      ggml_malloc(ਮਾਪ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CALLOC(ਅੰਕ, ਮਾਪ) ggml_calloc(ਅੰਕ, ਮਾਪ)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FREE(ptr) ਮੁਕਤ(ptr)

ਅਚਲ ਅੱਖਰ * ggml_ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_string(ਗਿਣਤੀ ggml_ਸਥਿਤੀ ਸਥਿਤੀ) {
    ਚੋਣ (ਸਥਿਤੀ) {
        ਹਾਲਤ GGML_ਸਥਿਤੀ_ALLOC_FAILED: ਵਾਪਸ "GGML status: error (failed to allocate memory)";
        ਹਾਲਤ GGML_ਸਥਿਤੀ_FAILED:       ਵਾਪਸ "GGML status: error (operation failed)";
        ਹਾਲਤ GGML_ਸਥਿਤੀ_SUCCESS:      ਵਾਪਸ "GGML status: success";
        ਹਾਲਤ GGML_ਸਥਿਤੀ_ABORTED:      ਵਾਪਸ "GGML status: warning (operation aborted)";
    }

    ਵਾਪਸ "GGML status: unknown";
}

ਦਸ਼ਮਲਵ ggml_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t x) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_fp16_ਮੰਜ਼ਿਲ_fp32 do_not_use_ggml_fp16_ਮੰਜ਼ਿਲ_fp32_ਵਿੱਚ_ggml
    ਵਾਪਸ GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
}

ggml_fp16_t ggml_fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ x) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_fp32_ਮੰਜ਼ਿਲ_fp16 do_not_use_ggml_fp32_ਮੰਜ਼ਿਲ_fp16_ਵਿੱਚ_ggml
    ਵਾਪਸ GGML_FP32_ਮੰਜ਼ਿਲ_FP16(x);
}

ਦਸ਼ਮਲਵ ggml_bf16_ਮੰਜ਼ਿਲ_fp32(ggml_bf16_t x) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_bf16_ਮੰਜ਼ਿਲ_fp32 do_not_use_ggml_bf16_ਮੰਜ਼ਿਲ_fp32_ਵਿੱਚ_ggml
    ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_ਮੰਜ਼ਿਲ_bf16(ਦਸ਼ਮਲਵ x) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_fp32_ਮੰਜ਼ਿਲ_bf16 do_not_use_ggml_fp32_ਮੰਜ਼ਿਲ_bf16_ਵਿੱਚ_ggml
    ਵਾਪਸ GGML_FP32_ਮੰਜ਼ਿਲ_BF16(x);
}

ਖਾਲੀ ggml_fp16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ(ਅਚਲ ggml_fp16_t * x, ਦਸ਼ਮਲਵ * y, int64_t n) {
    ਲਈ (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }
}

ਖਾਲੀ ggml_fp32_ਮੰਜ਼ਿਲ_fp16_ਪੰਕਤੀ(ਅਚਲ ਦਸ਼ਮਲਵ * x, ggml_fp16_t * y, int64_t n) {
    ਪੂਰਨ i = 0;
    ਲਈ (; i < n; ++i) {
        y[i] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(x[i]);
    }
}

ਖਾਲੀ ggml_bf16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ(ਅਚਲ ggml_bf16_t * x, ਦਸ਼ਮਲਵ * y, int64_t n) {
    ਪੂਰਨ i = 0;
    ਲਈ (; i < n; ++i) {
        y[i] = GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }
}

ਖਾਲੀ ggml_fp32_ਮੰਜ਼ਿਲ_bf16_ਪੰਕਤੀ_ref(ਅਚਲ ਦਸ਼ਮਲਵ * x, ggml_bf16_t * y, int64_t n) {
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_ਮੰਜ਼ਿਲ_bf16(x[i]);
    }
}

ਖਾਲੀ ggml_fp32_ਮੰਜ਼ਿਲ_bf16_ਪੰਕਤੀ(ਅਚਲ ਦਸ਼ਮਲਵ * x, ggml_bf16_t * y, int64_t n) {
  ਪੂਰਨ i = 0;
#ਜੇ defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  ਲਈ (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; i++) {
        y[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ਵਾਪਸ memcmp(guid_a, guid_b, ਆਕਾਰ(ggml_guid)) == 0;
}

ਅਚਲ ਅੱਖਰ * ggml_ਸੰਸਕਰਣ(ਖਾਲੀ) {
    ਵਾਪਸ GGML_ਸੰਸਕਰਣ;
}

ਅਚਲ ਅੱਖਰ * ggml_commit(ਖਾਲੀ) {
    ਵਾਪਸ GGML_COMMIT;
}

//
// timing
//

#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
ਸਥਿਰ int64_t timer_freq, timer_ਸ਼ੁਰੂ;
ਸਥਿਰ BOOL CALLBACK ggml_ਸਮਾਂ_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    ਅਣਵਰਤਿਆ(once);
    ਅਣਵਰਤਿਆ(param);
    ਅਣਵਰਤਿਆ(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_ਸ਼ੁਰੂ = t.QuadPart;

    ਵਾਪਸ TRUE;
}
ਖਾਲੀ ggml_ਸਮਾਂ_init(ਖਾਲੀ) {
    ਸਥਿਰ INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_ਸਮਾਂ_init_once, ਨਿਰਾ, ਨਿਰਾ);
}
int64_t ggml_ਸਮਾਂ_ms(ਖਾਲੀ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ਵਾਪਸ ((t.QuadPart-timer_ਸ਼ੁਰੂ) * 1000) / timer_freq;
}
int64_t ggml_ਸਮਾਂ_us(ਖਾਲੀ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ਵਾਪਸ ((t.QuadPart-timer_ਸ਼ੁਰੂ) * 1000000) / timer_freq;
}
#ਨਹੀਂਤਾਂ
ਖਾਲੀ ggml_ਸਮਾਂ_init(ਖਾਲੀ) {}
int64_t ggml_ਸਮਾਂ_ms(ਖਾਲੀ) {
    ਬਣਤਰ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ਵਾਪਸ (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_ਸਮਾਂ_us(ਖਾਲੀ) {
    ਬਣਤਰ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ਵਾਪਸ (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#ਜੇ_ਅੰਤ

int64_t ggml_ਚੱਕਰਾਂ(ਖਾਲੀ) {
    ਵਾਪਸ clock();
}

int64_t ggml_ਚੱਕਰਾਂ_ਪ੍ਰਤੀ_ms(ਖਾਲੀ) {
    ਵਾਪਸ CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
ਸਥਿਰ wchar_t * ggml_mbstowcs(ਅਚਲ ਅੱਖਰ * mbs) {
    ਪੂਰਨ wlen = Multi_Byte_ਮੰਜ਼ਿਲ_Wide_Char(CP_UTF8, 0, mbs, -1, ਨਿਰਾ, 0);
    ਜੇ (!wlen) {
        errno = EINVAL;
        ਵਾਪਸ ਨਿਰਾ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * ਆਕਾਰ(wchar_t));
    wlen = Multi_Byte_ਮੰਜ਼ਿਲ_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    ਜੇ (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ wbuf;
}
#ਜੇ_ਅੰਤ

FILE * ggml_fopen(ਅਚਲ ਅੱਖਰ * fname, ਅਚਲ ਅੱਖਰ * ਬਹੁਲਕ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    FILE * file = ਨਿਰਾ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    ਜੇ (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(ਬਹੁਲਕ);
        ਜੇ (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    ਵਾਪਸ file;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ fopen(fname, ਬਹੁਲਕ);
#ਜੇ_ਅੰਤ

}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_type_traits type_traits[GGML_TYPE_ਗਿਣਤੀ] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(ਦੁੱਗਣਾ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(ਦਸ਼ਮਲਵ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(ggml_fp16_t),
        .is_quantized             = false,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) ggml_fp16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_ਮੰਜ਼ਿਲ_fp16_ਪੰਕਤੀ,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_ਮਾਪ                = QK1_0,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q1_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_ਮਾਪ                = QK2_0,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q2_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_ਮਾਪ                = QK4_0,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q4_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_ਮਾਪ                = QK4_1,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q4_1),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_ਮਾਪ                = QK5_0,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q5_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_ਮਾਪ                = QK5_1,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q5_1),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_ਮਾਪ                = QK8_0,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q8_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_ਮਾਪ                = QK8_1,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_ਮਾਪ                = QK_MXFP4,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_mxfp4),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_ਮਾਪ                = QK_NVFP4,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_nvfp4),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q2_K),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q3_K),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q4_K),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q5_K),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q6_K),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq2_xxs),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq2_xxs,
        .from_float_ref           = ਨਿਰਾ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq2_xs),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq2_xs,
        .from_float_ref           = ਨਿਰਾ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq3_xxs),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq3_s),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq2_s),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq1_s),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq1_s,
        .from_float_ref           = ਨਿਰਾ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq1_m),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq1_m,
        .from_float_ref           = ਨਿਰਾ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_ਮਾਪ                = QK4_NL,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq4_nl),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_iq4_xs),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_ਪੰਕਤੀ_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_ਮਾਪ                = 1,
        .type_ਮਾਪ                = ਆਕਾਰ(ggml_bf16_t),
        .is_quantized             = false,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) ggml_bf16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_ਮੰਜ਼ਿਲ_bf16_ਪੰਕਤੀ_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_tq1_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_ਮਾਪ                = QK_K,
        .type_ਮਾਪ                = ਆਕਾਰ(ਖੰਡ_tq2_0),
        .is_quantized             = true,
        .ਮੰਜ਼ਿਲ_float                 = (ggml_ਮੰਜ਼ਿਲ_float_t) dequantize_ਪੰਕਤੀ_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ਪੰਕਤੀ_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ਮਾਪ                = 0,
        .type_ਮਾਪ                = 0,
        .is_quantized             = false,
    },
};

ਅਚਲ ਬਣਤਰ ggml_type_traits * ggml_get_type_traits(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ &type_traits[type];
}

//
// ggml object
//

ਬਣਤਰ ggml_object {
    size_t offs;
    size_t ਮਾਪ;

    ਬਣਤਰ ggml_object * ਅਗਲਾ;

    ਗਿਣਤੀ ggml_object_type type;

    ਅੱਖਰ padding[4];
};

ਸਥਿਰ ਅਚਲ size_t GGML_OBJECT_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ ggml_object);

//
// ggml context
//

ਬਣਤਰ ggml_context {
    size_t mem_ਮਾਪ;
    ਖਾਲੀ * mem_ਬਫਰ;
    bool   mem_ਬਫਰ_owned;
    bool   no_alloc;

    ਪੂਰਨ    n_objects;

    ਬਣਤਰ ggml_object * objects_begin;
    ਬਣਤਰ ggml_object * objects_ਸਿਰਾ;
};

//
// data types
//

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * GGML_OP_NAME[GGML_OP_ਗਿਣਤੀ] = {
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

static_assert(GGML_OP_ਗਿਣਤੀ == 102, "GGML_OP_COUNT != 102");

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * GGML_OP_ਚਿੰਨ੍ਹ[GGML_OP_ਗਿਣਤੀ] = {
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

static_assert(GGML_OP_ਗਿਣਤੀ == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_ਗਿਣਤੀ == 2, "GGML_OP_POOL_COUNT != 2");

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * GGML_UNARY_OP_NAME[GGML_UNARY_OP_ਗਿਣਤੀ] = {
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

static_assert(GGML_UNARY_OP_ਗਿਣਤੀ == 22, "GGML_UNARY_OP_COUNT != 22");

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * GGML_GLU_OP_NAME[GGML_GLU_OP_ਗਿਣਤੀ] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_ਗਿਣਤੀ == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(ਆਕਾਰ(ਬਣਤਰ ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(ਆਕਾਰ(ਬਣਤਰ ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

ਖਾਲੀ ggml_print_object(ਅਚਲ ਬਣਤਰ ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->ਮਾਪ, (ਅਚਲ ਖਾਲੀ *) obj->ਅਗਲਾ);
}

ਖਾਲੀ ggml_print_objects(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    ਬਣਤਰ ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ਅਚਲ ਖਾਲੀ *) ctx);

    ਜਦੋਂ (obj != ਨਿਰਾ) {
        ggml_print_object(obj);
        obj = obj->ਅਗਲਾ;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
        ਜੇ (tensor->ne[i] <= 0) {
            ਵਾਪਸ 0;
        }
    }

    size_t nbytes;
    ਅਚਲ size_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(tensor->type);
    ਜੇ (blck_ਮਾਪ == 1) {
        nbytes = ggml_type_ਮਾਪ(tensor->type);
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    ਨਹੀਂਤਾਂ {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_ਮਾਪ;
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ਵਾਪਸ nbytes;
}

size_t ggml_nbytes_pad(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_ਮਾਪ(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ type_traits[type].blck_ਮਾਪ;
}

size_t ggml_type_ਮਾਪ(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ type_traits[type].type_ਮਾਪ;
}

size_t ggml_ਪੰਕਤੀ_ਮਾਪ(ਗਿਣਤੀ ggml_type type, int64_t ne) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne % ggml_blck_ਮਾਪ(type) == 0);
    ਵਾਪਸ ggml_type_ਮਾਪ(type)*ne/ggml_blck_ਮਾਪ(type);
}

ਦੁੱਗਣਾ ggml_type_sizef(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ ((ਦੁੱਗਣਾ)(type_traits[type].type_ਮਾਪ))/type_traits[type].blck_ਮਾਪ;
}

ਅਚਲ ਅੱਖਰ * ggml_type_name(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ type_traits[type].type_name;
}

bool ggml_is_quantized(ਗਿਣਤੀ ggml_type type) {
    ਪ੍ਰਤਿਜ੍ਞਾ(type >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(type < GGML_TYPE_ਗਿਣਤੀ);
    ਵਾਪਸ type_traits[type].is_quantized;
}

ਅਚਲ ਅੱਖਰ * ggml_op_name(ਗਿਣਤੀ ggml_op op) {
    ਵਾਪਸ GGML_OP_NAME[op];
}

ਅਚਲ ਅੱਖਰ * ggml_op_ਚਿੰਨ੍ਹ(ਗਿਣਤੀ ggml_op op) {
    ਵਾਪਸ GGML_OP_ਚਿੰਨ੍ਹ[op];
}

ਅਚਲ ਅੱਖਰ * ggml_unary_op_name(ਗਿਣਤੀ ggml_unary_op op) {
    ਵਾਪਸ GGML_UNARY_OP_NAME[op];
}

ਅਚਲ ਅੱਖਰ * ggml_glu_op_name(ਗਿਣਤੀ ggml_glu_op op) {
    ਵਾਪਸ GGML_GLU_OP_NAME[op];
}

ਅਚਲ ਅੱਖਰ * ggml_op_desc(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (t->op == GGML_OP_UNARY) {
        ਗਿਣਤੀ ggml_unary_op uop = ggml_get_unary_op(t);
        ਵਾਪਸ ggml_unary_op_name(uop);
    }
    ਜੇ (t->op == GGML_OP_GLU) {
        ਗਿਣਤੀ ggml_glu_op gop = ggml_get_glu_op(t);
        ਵਾਪਸ ggml_glu_op_name(gop);
    }
    ਵਾਪਸ ggml_op_name(t->op);
}

size_t ggml_element_ਮਾਪ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_type_ਮਾਪ(tensor->type);
}

bool ggml_is_scalar(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_ਸਦਿਸ਼(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_ਮੈਟ੍ਰਿਕਸ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ tensor->ne[3] == 1;
}

ਪੂਰਨ ggml_n_dims(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        ਜੇ (tensor->ne[i] > 1) {
            ਵਾਪਸ i + 1;
        }
    }
    ਵਾਪਸ 1;
}

ਗਿਣਤੀ ggml_type ggml_ftype_ਮੰਜ਼ਿਲ_ggml_type(ਗਿਣਤੀ ggml_ftype ftype) {
    ਗਿਣਤੀ ggml_type wtype = GGML_TYPE_ਗਿਣਤੀ;

    ਚੋਣ (ftype) {
        ਹਾਲਤ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_ਗਿਣਤੀ; ਰੋਕੋ;
        ਹਾਲਤ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_ਗਿਣਤੀ; ਰੋਕੋ;
    }

    GGML_ASSERT(wtype != GGML_TYPE_ਗਿਣਤੀ);

    ਵਾਪਸ wtype;
}

size_t ggml_tensor_overhead(ਖਾਲੀ) {
    ਵਾਪਸ GGML_OBJECT_ਮਾਪ + GGML_TENSOR_ਮਾਪ;
}

bool ggml_is_transposed(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ tensor->nb[0] > tensor->nb[1];
}

ਸਥਿਰ bool ggml_is_contiguous_m_n(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ m, ਪੂਰਨ n) {
    size_t ਅਗਲਾ_nb = ggml_type_ਮਾਪ(tensor->type);
    ਜੇ (tensor->ne[0] != ggml_blck_ਮਾਪ(tensor->type) && tensor->nb[0] != ਅਗਲਾ_nb) {
        ਵਾਪਸ false;
    }
    ਅਗਲਾ_nb *= tensor->ne[0]/ggml_blck_ਮਾਪ(tensor->type);
    ਲਈ (ਪੂਰਨ i = 1; i < n; i++) {
        ਜੇ (i > m) {
            ਜੇ (tensor->ne[i] != 1 && tensor->nb[i] != ਅਗਲਾ_nb) {
                ਵਾਪਸ false;
            }
            ਅਗਲਾ_nb *= tensor->ne[i];
        } ਨਹੀਂਤਾਂ {
            // this dimension does not need to be contiguous
            ਅਗਲਾ_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ਵਾਪਸ true;
}

bool ggml_is_contiguous(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_ਮੰਜ਼ਿਲ_1(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_ਮੰਜ਼ਿਲ_2(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_ਮੰਜ਼ਿਲ_3(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_ਮਾਪ(tensor->type)/ggml_blck_ਮਾਪ(tensor->type);
}

bool ggml_is_permuted(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_ਮਾਪ(tensor->type);
}

bool ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ
        tensor->ne[0] == ggml_blck_ਮਾਪ(tensor->type) ||
        tensor->nb[0] == ggml_type_ਮਾਪ(tensor->type);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_is_padded_1d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ
        tensor->nb[0] == ggml_type_ਮਾਪ(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
        ਜੇ (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ਵਾਪਸ true;
        }
    }
    ਵਾਪਸ false;
}

bool ggml_are_same_shape(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(t0) ? ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_repeat_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ਬਣਤਰ ggml_context * ggml_init(ਬਣਤਰ ggml_init_params params) {
    ਸਥਿਰ bool is_ਪਹਿਲਾ_call = true;

    ggml_critical_section_ਸ਼ੁਰੂ();

    ਜੇ (is_ਪਹਿਲਾ_call) {
        // initialize time system (required on Windows)
        ggml_ਸਮਾਂ_init();

        is_ਪਹਿਲਾ_call = false;
    }

    ggml_critical_section_ਸਿਰਾ();

    ਬਣਤਰ ggml_context * ctx = GGML_MALLOC(ਆਕਾਰ(ਬਣਤਰ ggml_context));

    // allow to call ggml_init with 0 size
    ਜੇ (params.mem_ਮਾਪ == 0) {
        params.mem_ਮਾਪ = GGML_MEM_ALIGN;
    }

    ਅਚਲ size_t mem_ਮਾਪ = params.mem_ਬਫਰ ? params.mem_ਮਾਪ : GGML_PAD(params.mem_ਮਾਪ, GGML_MEM_ALIGN);

    *ctx = (ਬਣਤਰ ggml_context) {
        /*.mem_size           =*/ mem_ਮਾਪ,
        /*.mem_buffer         =*/ params.mem_ਬਫਰ ? params.mem_ਬਫਰ : ggml_aligned_malloc(mem_ਮਾਪ),
        /*.mem_buffer_owned   =*/ params.mem_ਬਫਰ ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ਨਿਰਾ,
        /*.objects_end        =*/ ਨਿਰਾ,
    };

    GGML_ASSERT(ctx->mem_ਬਫਰ != ਨਿਰਾ);

    GGML_ASSERT_ALIGNED(ctx->mem_ਬਫਰ);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ਵਾਪਸ ctx;
}

ਖਾਲੀ ggml_reset(ਬਣਤਰ ggml_context * ctx) {
    ਜੇ (ctx == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ਨਿਰਾ;
    ctx->objects_ਸਿਰਾ   = ਨਿਰਾ;
}

ਖਾਲੀ ggml_free(ਬਣਤਰ ggml_context * ctx) {
    ਜੇ (ctx == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    ਜੇ (ctx->mem_ਬਫਰ_owned) {
        ggml_aligned_free(ctx->mem_ਬਫਰ, ctx->mem_ਮਾਪ);
    }

    GGML_FREE(ctx);
}

size_t ggml_ਵਰਤਿਆ_mem(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    ਵਾਪਸ ctx->objects_ਸਿਰਾ == ਨਿਰਾ ? 0 : ctx->objects_ਸਿਰਾ->offs + ctx->objects_ਸਿਰਾ->ਮਾਪ;
}

bool ggml_get_no_alloc(ਬਣਤਰ ggml_context * ctx) {
    ਵਾਪਸ ctx->no_alloc;
}

ਖਾਲੀ ggml_ਸੈੱਟ_no_alloc(ਬਣਤਰ ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

ਖਾਲੀ * ggml_get_mem_ਬਫਰ(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    ਵਾਪਸ ctx->mem_ਬਫਰ;
}

size_t ggml_get_mem_ਮਾਪ(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    ਵਾਪਸ ctx->mem_ਮਾਪ;
}

size_t ggml_get_max_tensor_ਮਾਪ(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    size_t max_ਮਾਪ = 0;

    ਲਈ (ਬਣਤਰ ggml_tensor * tensor = ggml_get_ਪਹਿਲਾ_tensor(ctx); tensor != ਨਿਰਾ; tensor = ggml_get_ਅਗਲਾ_tensor(ctx, tensor)) {
        size_t ਬਾਈਟਾਂ = ggml_nbytes(tensor);
        max_ਮਾਪ = MAX(max_ਮਾਪ, ਬਾਈਟਾਂ);
    }

    ਵਾਪਸ max_ਮਾਪ;
}

////////////////////////////////////////////////////////////////////////////////

ਸਥਿਰ ਬਣਤਰ ggml_object * ggml_new_object(ਬਣਤਰ ggml_context * ctx, ਗਿਣਤੀ ggml_object_type type, size_t ਮਾਪ) {
    // always insert objects at the end of the context's memory pool
    ਬਣਤਰ ggml_object * obj_cur = ctx->objects_ਸਿਰਾ;

    ਅਚਲ size_t cur_offs = obj_cur == ਨਿਰਾ ? 0 : obj_cur->offs;
    ਅਚਲ size_t cur_ਮਾਪ = obj_cur == ਨਿਰਾ ? 0 : obj_cur->ਮਾਪ;
    ਅਚਲ size_t cur_ਸਿਰਾ  = cur_offs + cur_ਮਾਪ;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(ਮਾਪ <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t ਮਾਪ_needed = GGML_PAD(ਮਾਪ, GGML_MEM_ALIGN);

    ਅੱਖਰ * ਅਚਲ mem_ਬਫਰ = ctx->mem_ਬਫਰ;
    ਬਣਤਰ ggml_object * ਅਚਲ obj_new = (ਬਣਤਰ ggml_object *)(mem_ਬਫਰ + cur_ਸਿਰਾ);

    // integer overflow checks
    ਜੇ (cur_ਸਿਰਾ > SIZE_MAX - ਮਾਪ_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_ਸਿਰਾ, ਮਾਪ_needed);
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਜੇ (cur_ਸਿਰਾ + ਮਾਪ_needed > SIZE_MAX - GGML_OBJECT_ਮਾਪ) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_ਸਿਰਾ, ਮਾਪ_needed, (size_t) GGML_OBJECT_ਮਾਪ);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (cur_ਸਿਰਾ + ਮਾਪ_needed + GGML_OBJECT_ਮਾਪ > ctx->mem_ਮਾਪ) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_ਸਿਰਾ + ਮਾਪ_needed + GGML_OBJECT_ਮਾਪ, ctx->mem_ਮਾਪ);
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਨਿਰਾ;
    }

    *obj_new = (ਬਣਤਰ ggml_object) {
        .offs = cur_ਸਿਰਾ + GGML_OBJECT_ਮਾਪ,
        .ਮਾਪ = ਮਾਪ_needed,
        .ਅਗਲਾ = ਨਿਰਾ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_ਬਫਰ + obj_new->offs);

    ਜੇ (obj_cur != ਨਿਰਾ) {
        obj_cur->ਅਗਲਾ = obj_new;
    } ਨਹੀਂਤਾਂ {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_ਸਿਰਾ = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ਵਾਪਸ obj_new;
}

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_new_tensor_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type      type,
        ਪੂਰਨ                   n_dims,
        ਅਚਲ int64_t       * ne,
        ਬਣਤਰ ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_ਗਿਣਤੀ);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    ਜੇ (view_src != ਨਿਰਾ && view_src->view_src != ਨਿਰਾ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    ਲਈ (ਪੂਰਨ i = 1; i < n_dims; i++) {
        ਜੇ (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t ਅੰਕੜੇ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(type, ne[0]);
    ਲਈ (ਪੂਰਨ i = 1; i < n_dims; i++) {
        ਅੰਕੜੇ_ਮਾਪ *= ne[i];
    }

    GGML_ASSERT(view_src == ਨਿਰਾ || ਅੰਕੜੇ_ਮਾਪ == 0 || ਅੰਕੜੇ_ਮਾਪ + view_offs <= ggml_nbytes(view_src));

    ਖਾਲੀ * ਅੰਕੜੇ = view_src != ਨਿਰਾ ? view_src->ਅੰਕੜੇ : ਨਿਰਾ;
    ਜੇ (ਅੰਕੜੇ != ਨਿਰਾ) {
        ਅੰਕੜੇ = (ਅੱਖਰ *) ਅੰਕੜੇ + view_offs;
    }

    size_t obj_alloc_ਮਾਪ = 0;

    ਜੇ (view_src == ਨਿਰਾ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_ਮਾਪ = ਅੰਕੜੇ_ਮਾਪ;
    }

    GGML_ASSERT(GGML_TENSOR_ਮਾਪ <= SIZE_MAX - obj_alloc_ਮਾਪ);

    ਬਣਤਰ ggml_object * ਅਚਲ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_ਮਾਪ + obj_alloc_ਮਾਪ);
    GGML_ASSERT(obj_new);

    ਬਣਤਰ ggml_tensor * ਅਚਲ ਨਤੀਜਾ = (ਬਣਤਰ ggml_tensor *)((ਅੱਖਰ *)ctx->mem_ਬਫਰ + obj_new->offs);

    *ਨਤੀਜਾ = (ਬਣਤਰ ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ਨਿਰਾ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ਨਿਰਾ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_ਮਾਪ > 0 ? (ਖਾਲੀ *)(ਨਤੀਜਾ + 1) : ਅੰਕੜੇ,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ਨਿਰਾ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    ਲਈ (ਪੂਰਨ i = 0; i < n_dims; i++) {
        ਨਤੀਜਾ->ne[i] = ne[i];
    }

    ਨਤੀਜਾ->nb[0] = ggml_type_ਮਾਪ(type);
    ਨਤੀਜਾ->nb[1] = ਨਤੀਜਾ->nb[0]*(ਨਤੀਜਾ->ne[0]/ggml_blck_ਮਾਪ(type));
    ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ->nb[i] = ਨਤੀਜਾ->nb[i - 1]*ਨਤੀਜਾ->ne[i - 1];
    }

    ctx->n_objects++;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_new_tensor(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type      type,
        ਪੂਰਨ                   n_dims,
        ਅਚਲ int64_t       * ne) {
    ਵਾਪਸ ggml_new_tensor_impl(ctx, type, n_dims, ne, ਨਿਰਾ, 0);
}

ਬਣਤਰ ggml_tensor * ggml_new_tensor_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type      type,
        int64_t ne0) {
    ਵਾਪਸ ggml_new_tensor(ctx, type, 1, &ne0);
}

ਬਣਤਰ ggml_tensor * ggml_new_tensor_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ਅਚਲ int64_t ne[2] = { ne0, ne1 };
    ਵਾਪਸ ggml_new_tensor(ctx, type, 2, ne);
}

ਬਣਤਰ ggml_tensor * ggml_new_tensor_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ਅਚਲ int64_t ne[3] = { ne0, ne1, ne2 };
    ਵਾਪਸ ggml_new_tensor(ctx, type, 3, ne);
}

ਬਣਤਰ ggml_tensor * ggml_new_tensor_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ਅਚਲ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ਵਾਪਸ ggml_new_tensor(ctx, type, 4, ne);
}

ਖਾਲੀ * ggml_new_ਬਫਰ(ਬਣਤਰ ggml_context * ctx, size_t nbytes) {
    ਬਣਤਰ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_ਬਫਰ, nbytes);

    ਵਾਪਸ (uint8_t *)ctx->mem_ਬਫਰ + obj->offs;
}

ਬਣਤਰ ggml_tensor * ggml_dup_tensor(ਬਣਤਰ ggml_context * ctx, ਅਚਲ ਬਣਤਰ ggml_tensor * src) {
    ਵਾਪਸ ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

ਖਾਲੀ ggml_unravel_ਸੂਚਕਾਂਕ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ਅਚਲ int64_t ne2 = tensor->ne[2];
    ਅਚਲ int64_t ne1 = tensor->ne[1];
    ਅਚਲ int64_t ne0 = tensor->ne[0];

    ਅਚਲ int64_t i3_ = (i/(ne2*ne1*ne0));
    ਅਚਲ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ਅਚਲ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ਅਚਲ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    ਜੇ (i0) {
        * i0 = i0_;
    }
    ਜੇ (i1) {
        * i1 = i1_;
    }
    ਜੇ (i2) {
        * i2 = i2_;
    }
    ਜੇ (i3) {
        * i3 = i3_;
    }
}

ਖਾਲੀ * ggml_get_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ tensor->ਅੰਕੜੇ;
}

ਦਸ਼ਮਲਵ * ggml_get_ਅੰਕੜੇ_f32(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਪ੍ਰਤਿਜ੍ਞਾ(tensor->type == GGML_TYPE_F32);
    ਵਾਪਸ (ਦਸ਼ਮਲਵ *)(tensor->ਅੰਕੜੇ);
}

ਗਿਣਤੀ ggml_unary_op ggml_get_unary_op(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ਵਾਪਸ (ਗਿਣਤੀ ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

ਗਿਣਤੀ ggml_glu_op ggml_get_glu_op(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ਵਾਪਸ (ਗਿਣਤੀ ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ਅਚਲ ਅੱਖਰ * ggml_get_name(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ tensor->name;
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_name(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਅੱਖਰ * name) {
    size_t i;
    ਲਈ (i = 0; i < ਆਕਾਰ(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ਵਾਪਸ tensor;
}

ਬਣਤਰ ggml_tensor * ggml_format_name(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਅੱਖਰ * fmt, ...) {
    va_list ਦਲੀਲਾਂ_2;
    va_start(ਦਲੀਲਾਂ_2, fmt);
    vsnprintf(tensor->name, ਆਕਾਰ(tensor->name), fmt, ਦਲੀਲਾਂ_2);
    va_end(ਦਲੀਲਾਂ_2);
    ਵਾਪਸ tensor;
}

ਬਣਤਰ ggml_tensor * ggml_view_tensor(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * src) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (view)", src->name);

    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ->nb[i] = src->nb[i];
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_get_ਪਹਿਲਾ_tensor(ਅਚਲ ਬਣਤਰ ggml_context * ctx) {
    ਬਣਤਰ ggml_object * obj = ctx->objects_begin;

    ਅੱਖਰ * ਅਚਲ mem_ਬਫਰ = ctx->mem_ਬਫਰ;

    ਜਦੋਂ (obj != ਨਿਰਾ) {
        ਜੇ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ਵਾਪਸ (ਬਣਤਰ ggml_tensor *)(mem_ਬਫਰ + obj->offs);
        }

        obj = obj->ਅਗਲਾ;
    }

    ਵਾਪਸ ਨਿਰਾ;
}

ਬਣਤਰ ggml_tensor * ggml_get_ਅਗਲਾ_tensor(ਅਚਲ ਬਣਤਰ ggml_context * ctx, ਬਣਤਰ ggml_tensor * tensor) {
    ਬਣਤਰ ggml_object * obj = (ਬਣਤਰ ggml_object *) ((ਅੱਖਰ *)tensor - GGML_OBJECT_ਮਾਪ);
    obj = obj->ਅਗਲਾ;

    ਅੱਖਰ * ਅਚਲ mem_ਬਫਰ = ctx->mem_ਬਫਰ;

    ਜਦੋਂ (obj != ਨਿਰਾ) {
        ਜੇ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ਵਾਪਸ (ਬਣਤਰ ggml_tensor *)(mem_ਬਫਰ + obj->offs);
        }

        obj = obj->ਅਗਲਾ;
    }

    ਵਾਪਸ ਨਿਰਾ;
}

ਬਣਤਰ ggml_tensor * ggml_get_tensor(ਬਣਤਰ ggml_context * ctx, ਅਚਲ ਅੱਖਰ * name) {
    ਬਣਤਰ ggml_object * obj = ctx->objects_begin;

    ਅੱਖਰ * ਅਚਲ mem_ਬਫਰ = ctx->mem_ਬਫਰ;

    ਜਦੋਂ (obj != ਨਿਰਾ) {
        ਜੇ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ਬਣਤਰ ggml_tensor * cur = (ਬਣਤਰ ggml_tensor *)(mem_ਬਫਰ + obj->offs);
            ਜੇ (strcmp(cur->name, name) == 0) {
                ਵਾਪਸ cur;
            }
        }

        obj = obj->ਅਗਲਾ;
    }

    ਵਾਪਸ ਨਿਰਾ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_dup_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_DUP;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_dup(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_dup_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_dup_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_dup_impl(ctx, a, true);
}

// ggml_add

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜੋ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_ਜੋੜੋ_impl(ctx, a, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_ਜੋੜੋ_impl(ctx, a, b, true);
}

// ggml_add_cast

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_cast_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਗਿਣਤੀ   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_ਪੰਕਤੀਆਂ(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜੋ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_cast(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਗਿਣਤੀ   ggml_type      type) {
    ਵਾਪਸ ggml_ਜੋੜੋ_cast_impl(ctx, a, b, type);
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_id(
            ਬਣਤਰ ggml_context * ctx,
            ਬਣਤਰ ggml_tensor  * a,
            ਬਣਤਰ ggml_tensor  * b,
            ਬਣਤਰ ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜੋ_ID;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;
    ਨਤੀਜਾ->src[2] = ids;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_add1

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_add1_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_ADD1;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_add1(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_add1_impl(ctx, a, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_add1_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_acc_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(ਵਿੱਥ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, ਵਿੱਥ, inplace ? 1 : 0 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_ACC;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_acc(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, ਵਿੱਥ, false);
}

ਬਣਤਰ ggml_tensor * ggml_acc_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, ਵਿੱਥ, true);
}

// ggml_sub

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_sub_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SUB;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_sub(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_sub_impl(ctx, a, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_sub_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_mul_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_MUL;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_mul(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_mul_impl(ctx, a, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_mul_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_div_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_DIV;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_div(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_div_impl(ctx, a, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_div_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_sqr_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SQR;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_sqr(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sqr_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_sqr_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_sqrt_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SQRT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_sqrt(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sqrt_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_sqrt_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_log_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_LOG;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_log(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_log_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_log_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_log_impl(ctx, a, true);
}

ਬਣਤਰ ggml_tensor * ggml_expm1(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ਬਣਤਰ ggml_tensor * ggml_expm1_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ਬਣਤਰ ggml_tensor * ggml_softplus(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ਬਣਤਰ ggml_tensor * ggml_softplus_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_sin_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SIN;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_sin(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sin_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_sin_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_cos_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_COS;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_cos(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_cos_impl(ctx, a, false);
}

ਬਣਤਰ ggml_tensor * ggml_cos_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, a->type, 1);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜ;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_sum_rows

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜ_ਪੰਕਤੀਆਂ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_cumsum

ਬਣਤਰ ggml_tensor * ggml_cumsum(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_CUMSUM;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_mean

ਬਣਤਰ ggml_tensor * ggml_ਔਸਤ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_ਔਸਤ;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_argmax

ਬਣਤਰ ggml_tensor * ggml_argmax(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    ਨਤੀਜਾ->op     = GGML_OP_ARGMAX;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_count_equal

ਬਣਤਰ ggml_tensor * ggml_ਗਿਣਤੀ_equal(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    ਨਤੀਜਾ->op     = GGML_OP_ਗਿਣਤੀ_EQUAL;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_repeat

ਬਣਤਰ ggml_tensor * ggml_repeat(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ਨਤੀਜਾ->op     = GGML_OP_REPEAT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_repeat_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ਅਚਲ bool can_repeat = ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ਨਤੀਜਾ->op     = GGML_OP_REPEAT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_repeat_back

ਬਣਤਰ ggml_tensor * ggml_repeat_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ਨਤੀਜਾ->op     = GGML_OP_REPEAT_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_concat

ਬਣਤਰ ggml_tensor * ggml_concat(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਬਣਤਰ ggml_tensor  * b,
    ਪੂਰਨ                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    ਲਈ (ਪੂਰਨ d = 0; d < GGML_MAX_DIMS; ++d) {
        ਜੇ (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ਜਾਰੀ;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, dim);

    ਨਤੀਜਾ->op     = GGML_OP_CONCAT;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_abs

ਬਣਤਰ ggml_tensor * ggml_abs(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ਬਣਤਰ ggml_tensor * ggml_abs_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ਬਣਤਰ ggml_tensor * ggml_sgn(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ਬਣਤਰ ggml_tensor * ggml_sgn_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ਬਣਤਰ ggml_tensor * ggml_neg(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ਬਣਤਰ ggml_tensor * ggml_neg_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ਬਣਤਰ ggml_tensor * ggml_ਕਦਮ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_ਕਦਮ);
}

ਬਣਤਰ ggml_tensor * ggml_ਕਦਮ_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ਕਦਮ);
}

// ggml_tanh

ਬਣਤਰ ggml_tensor * ggml_tanh(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ਬਣਤਰ ggml_tensor * ggml_tanh_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ਬਣਤਰ ggml_tensor * ggml_elu(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ਬਣਤਰ ggml_tensor * ggml_elu_inplace(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ਬਣਤਰ ggml_tensor * ggml_relu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ਬਣਤਰ ggml_tensor * ggml_relu_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ਬਣਤਰ ggml_tensor * ggml_leaky_relu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 negative_ਢਲਾਨ,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &negative_ਢਲਾਨ, ਆਕਾਰ(negative_ਢਲਾਨ));

    ਨਤੀਜਾ->op     = GGML_OP_LEAKY_RELU;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_sigmoid

ਬਣਤਰ ggml_tensor * ggml_sigmoid(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ਬਣਤਰ ggml_tensor * ggml_sigmoid_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ਬਣਤਰ ggml_tensor * ggml_gelu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ਬਣਤਰ ggml_tensor * ggml_gelu_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ਬਣਤਰ ggml_tensor * ggml_gelu_erf(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ਬਣਤਰ ggml_tensor * ggml_gelu_erf_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ਬਣਤਰ ggml_tensor * ggml_gelu_ਤੇਜ਼(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ਤੇਜ਼);
}

ਬਣਤਰ ggml_tensor * ggml_gelu_ਤੇਜ਼_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ਤੇਜ਼);
}

// ggml_silu

ਬਣਤਰ ggml_tensor * ggml_silu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ਬਣਤਰ ggml_tensor * ggml_silu_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ਬਣਤਰ ggml_tensor * ggml_xielu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ alpha_n,
        ਦਸ਼ਮਲਵ alpha_p,
        ਦਸ਼ਮਲਵ beta,
        ਦਸ਼ਮਲਵ eps) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 3, beta);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 4, eps);

    ਨਤੀਜਾ->op     = GGML_OP_UNARY;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_silu_back

ਬਣਤਰ ggml_tensor * ggml_silu_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SILU_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml hardswish

ਬਣਤਰ ggml_tensor * ggml_hardswish(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ਬਣਤਰ ggml_tensor * ggml_hardsigmoid(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ਬਣਤਰ ggml_tensor * ggml_exp(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ਬਣਤਰ ggml_tensor * ggml_exp_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_glu_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਗਿਣਤੀ ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    ਜੇ (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ਨਿਰਾ, 0);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) op);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, (int32_t) swapped);

    ਨਤੀਜਾ->op     = GGML_OP_GLU;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_floor

ਬਣਤਰ ggml_tensor * ggml_floor(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ਬਣਤਰ ggml_tensor * ggml_floor_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ਬਣਤਰ ggml_tensor * ggml_ceil(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ਬਣਤਰ ggml_tensor * ggml_ceil_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ਬਣਤਰ ggml_tensor * ggml_ਗੇੜਾ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_ਗੇੜਾ);
}

ਬਣਤਰ ggml_tensor * ggml_ਗੇੜਾ_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ਗੇੜਾ);
}

//ggml_trunc

ਬਣਤਰ ggml_tensor * ggml_trunc(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ਬਣਤਰ ggml_tensor * ggml_trunc_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ਬਣਤਰ ggml_tensor * ggml_glu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_glu_op      op,
        bool                  swapped) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, op, swapped);
}

ਬਣਤਰ ggml_tensor * ggml_glu_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਗਿਣਤੀ ggml_glu_op      op) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ਬਣਤਰ ggml_tensor * ggml_reglu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_REGLU, false);
}

ਬਣਤਰ ggml_tensor * ggml_reglu_swapped(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_REGLU, true);
}

ਬਣਤਰ ggml_tensor * ggml_reglu_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ਬਣਤਰ ggml_tensor * ggml_geglu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU, false);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_swapped(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU, true);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ਬਣਤਰ ggml_tensor * ggml_swiglu(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_SWIGLU, false);
}

ਬਣਤਰ ggml_tensor * ggml_swiglu_swapped(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_SWIGLU, true);
}

ਬਣਤਰ ggml_tensor * ggml_swiglu_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ਬਣਤਰ ggml_tensor * ggml_geglu_erf(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU_ERF, false);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_erf_swapped(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU_ERF, true);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_erf_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ਬਣਤਰ ggml_tensor * ggml_geglu_ਤੇਜ਼(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU_ਤੇਜ਼, false);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_ਤੇਜ਼_swapped(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, ਨਿਰਾ, GGML_GLU_OP_GEGLU_ਤੇਜ਼, true);
}

ਬਣਤਰ ggml_tensor * ggml_geglu_ਤੇਜ਼_ਵੰਡੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ਤੇਜ਼, false);
}

ਬਣਤਰ ggml_tensor * ggml_swiglu_oai(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 alpha,
        ਦਸ਼ਮਲਵ                 ਹੱਦਬੰਦੀ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 2, alpha);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 3, ਹੱਦਬੰਦੀ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_swiglu_clamp(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 ਹੱਦਬੰਦੀ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 3, ਹੱਦਬੰਦੀ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_norm

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_norm_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &eps, ਆਕਾਰ(eps));

    ਨਤੀਜਾ->op     = GGML_OP_NORM;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_norm(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_norm_impl(ctx, a, eps, false);
}

ਬਣਤਰ ggml_tensor * ggml_norm_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_rms_norm_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &eps, ਆਕਾਰ(eps));

    ਨਤੀਜਾ->op     = GGML_OP_RMS_NORM;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_rms_norm(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_rms_norm_impl(ctx, a, eps, false);
}

ਬਣਤਰ ggml_tensor * ggml_rms_norm_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ਬਣਤਰ ggml_tensor * ggml_rms_norm_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 eps) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &eps, ਆਕਾਰ(eps));

    ਨਤੀਜਾ->op     = GGML_OP_RMS_NORM_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_group_norm

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਸਮੂਹ_norm_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_groups,
        ਦਸ਼ਮਲਵ                 eps,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, n_groups);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 1, eps);

    ਨਤੀਜਾ->op     = GGML_OP_ਸਮੂਹ_NORM;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਸਮੂਹ_norm(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_groups,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_ਸਮੂਹ_norm_impl(ctx, a, n_groups, eps, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਸਮੂਹ_norm_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_groups,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_ਸਮੂਹ_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_l2_norm_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, eps);

    ਨਤੀਜਾ->op     = GGML_OP_L2_NORM;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_l2_norm(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_l2_norm_impl(ctx, a, eps, false);
}

ਬਣਤਰ ggml_tensor * ggml_l2_norm_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 eps) {
    ਵਾਪਸ ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_ਸੈੱਟ_acc(
        ਬਣਤਰ ggml_tensor * a,
        ਗਿਣਤੀ ggml_prec       prec) {
    ਚੋਣ (a->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ਅਚਲ int32_t prec_i32 = (int32_t) prec;
                ggml_ਸੈੱਟ_op_params_i32(a, 0, prec_i32);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                ਅਚਲ int32_t prec_i32 = (int32_t) prec;
                ggml_ਸੈੱਟ_op_params_i32(a, 3, prec_i32);
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ false;
    };

    ਵਾਪਸ true;
}

bool ggml_prec_ਸੈੱਟ_src(
        ਬਣਤਰ ggml_tensor * a,
        ਗਿਣਤੀ ggml_prec       prec,
        ਪੂਰਨ                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ਚੋਣ (a->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ਜੇ (idx != 1) {
                    ਵਾਪਸ false;
                }

                ਅਚਲ int32_t prec_i32 = (int32_t) prec;
                ggml_ਸੈੱਟ_op_params_i32(a, 2 + idx, prec_i32);
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ false;
    };

    ਵਾਪਸ true;
}

// ggml_mul_mat

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_mul_mat(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ਬਣਤਰ ggml_tensor * ggml_mul_mat(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ਅਚਲ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_MUL_MAT;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_mul_mat_ਸੈੱਟ_prec(
        ਬਣਤਰ ggml_tensor * a,
        ਗਿਣਤੀ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ਅਚਲ int32_t prec_i32 = (int32_t) prec;

    ggml_ਸੈੱਟ_op_params_i32(a, 0, prec_i32);
}

ਖਾਲੀ ggml_mul_mat_ਸੈੱਟ_hint(
        ਬਣਤਰ ggml_tensor * a,
        ਗਿਣਤੀ ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ਅਚਲ int32_t hint_i32 = (int32_t) hint;

    ggml_ਸੈੱਟ_op_params_i32(a, 1, hint_i32);
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
ਬਣਤਰ ggml_tensor * ggml_mul_mat_id(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * as,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ਅਚਲ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_MUL_MAT_ID;
    ਨਤੀਜਾ->src[0] = as;
    ਨਤੀਜਾ->src[1] = b;
    ਨਤੀਜਾ->src[2] = ids;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_out_prod

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_ਬਾਹਰਲਾ_prod(ਅਚਲ ਬਣਤਰ ggml_tensor * t0, ਅਚਲ ਬਣਤਰ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ਬਣਤਰ ggml_tensor * ggml_ਬਾਹਰਲਾ_prod(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_ਬਾਹਰਲਾ_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ਅਚਲ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_ਬਾਹਰਲਾ_PROD;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_scale

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_scale_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 s,
        ਦਸ਼ਮਲਵ                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਦਸ਼ਮਲਵ params[2] = { s, b };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_SCALE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_scale(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 s) {
    ਵਾਪਸ ggml_scale_impl(ctx, a, s, 0.0, false);
}

ਬਣਤਰ ggml_tensor * ggml_scale_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 s) {
    ਵਾਪਸ ggml_scale_impl(ctx, a, s, 0.0, true);
}

ਬਣਤਰ ggml_tensor * ggml_scale_bias(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 s,
        ਦਸ਼ਮਲਵ                 b) {
    ਵਾਪਸ ggml_scale_impl(ctx, a, s, b, false);
}

ਬਣਤਰ ggml_tensor * ggml_scale_bias_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 s,
        ਦਸ਼ਮਲਵ                 b) {
    ਵਾਪਸ ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(ਵਿੱਥ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, ਵਿੱਥ, inplace ? 1 : 0 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_ਸੈੱਟ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, nb1, nb2, nb3, ਵਿੱਥ, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, nb1, nb2, nb3, ਵਿੱਥ, true);
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], ਵਿੱਥ, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_1d_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], ਵਿੱਥ, true);
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], ਵਿੱਥ, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_2d_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        size_t                nb1,
        size_t                ਵਿੱਥ) {
    ਵਾਪਸ ggml_ਸੈੱਟ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], ਵਿੱਥ, true);
}

// ggml_cpy

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_cpy_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, b);
    ਜੇ (ਲੜੀ_ਲੰਬਾਈ(b->name) > 0) {
        ggml_format_name(ਨਤੀਜਾ, "%s (copy of %s)", b->name, a->name);
    } ਨਹੀਂਤਾਂ {
        ggml_format_name(ਨਤੀਜਾ, "%s (copy)", a->name);
    }

    ਨਤੀਜਾ->op     = GGML_OP_CPY;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_cpy(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor * a,
        ਬਣਤਰ ggml_tensor * b) {
    ਵਾਪਸ ggml_cpy_impl(ctx, a, b);
}

ਬਣਤਰ ggml_tensor * ggml_cast(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ   ggml_type      type) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(ਨਤੀਜਾ, "%s (copy)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_CPY;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = ਨਤੀਜਾ; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_cont

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_cont_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);
    ggml_format_name(ਨਤੀਜਾ, "%s (cont)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_CONT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_cont(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor * a) {
    ਵਾਪਸ ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ਬਣਤਰ ggml_tensor * ggml_cont_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0) {
    ਵਾਪਸ ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ਬਣਤਰ ggml_tensor * ggml_cont_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ਵਾਪਸ ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ਬਣਤਰ ggml_tensor * ggml_cont_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ਵਾਪਸ ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ਬਣਤਰ ggml_tensor * ggml_cont_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(ਨਤੀਜਾ, "%s (cont)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_CONT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_reshape

ਬਣਤਰ ggml_tensor * ggml_reshape(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor * a,
        ਬਣਤਰ ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (reshaped)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_RESHAPE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_reshape_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ਅਚਲ int64_t ne[1] = { ne0 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (reshaped)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_RESHAPE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_reshape_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ਅਚਲ int64_t ne[2] = { ne0, ne1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (reshaped)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_RESHAPE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_reshape_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ਅਚਲ int64_t ne[3] = { ne0, ne1, ne2 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (reshaped)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_RESHAPE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_reshape_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ਅਚਲ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(ਨਤੀਜਾ, "%s (reshaped)", a->name);

    ਨਤੀਜਾ->op     = GGML_OP_RESHAPE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_view_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_dims,
        ਅਚਲ int64_t       * ne,
        size_t                ਵਿੱਥ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, ਵਿੱਥ);
    ggml_format_name(ਨਤੀਜਾ, "%s (view)", a->name);

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &ਵਿੱਥ, ਆਕਾਰ(ਵਿੱਥ));

    ਨਤੀਜਾ->op     = GGML_OP_VIEW;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_view_1d

ਬਣਤਰ ggml_tensor * ggml_view_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        size_t                ਵਿੱਥ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_impl(ctx, a, 1, &ne0, ਵਿੱਥ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_view_2d

ਬਣਤਰ ggml_tensor * ggml_view_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                ਵਿੱਥ) {
    ਅਚਲ int64_t ne[2] = { ne0, ne1 };

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_impl(ctx, a, 2, ne, ਵਿੱਥ);

    ਨਤੀਜਾ->nb[1] = nb1;
    ਨਤੀਜਾ->nb[2] = ਨਤੀਜਾ->nb[1]*ne1;
    ਨਤੀਜਾ->nb[3] = ਨਤੀਜਾ->nb[2];

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_view_3d

ਬਣਤਰ ggml_tensor * ggml_view_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                ਵਿੱਥ) {
    ਅਚਲ int64_t ne[3] = { ne0, ne1, ne2 };

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_impl(ctx, a, 3, ne, ਵਿੱਥ);

    ਨਤੀਜਾ->nb[1] = nb1;
    ਨਤੀਜਾ->nb[2] = nb2;
    ਨਤੀਜਾ->nb[3] = ਨਤੀਜਾ->nb[2]*ne2;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_view_4d

ਬਣਤਰ ggml_tensor * ggml_view_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ਵਿੱਥ) {
    ਅਚਲ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_impl(ctx, a, 4, ne, ਵਿੱਥ);

    ਨਤੀਜਾ->nb[1] = nb1;
    ਨਤੀਜਾ->nb[2] = nb2;
    ਨਤੀਜਾ->nb[3] = nb3;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_permute

ਬਣਤਰ ggml_tensor * ggml_ਕ੍ਰਮ_ਬਦਲੋ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   axis0,
        ਪੂਰਨ                   axis1,
        ਪੂਰਨ                   axis2,
        ਪੂਰਨ                   axis3) {
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

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);
    ggml_format_name(ਨਤੀਜਾ, "%s (permuted)", a->name);

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

    ਨਤੀਜਾ->ne[0] = ne[0];
    ਨਤੀਜਾ->ne[1] = ne[1];
    ਨਤੀਜਾ->ne[2] = ne[2];
    ਨਤੀਜਾ->ne[3] = ne[3];

    ਨਤੀਜਾ->nb[0] = nb[0];
    ਨਤੀਜਾ->nb[1] = nb[1];
    ਨਤੀਜਾ->nb[2] = nb[2];
    ਨਤੀਜਾ->nb[3] = nb[3];

    ਨਤੀਜਾ->op     = GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ;
    ਨਤੀਜਾ->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_transpose

ਬਣਤਰ ggml_tensor * ggml_transpose(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);
    ggml_format_name(ਨਤੀਜਾ, "%s (transposed)", a->name);

    ਨਤੀਜਾ->ne[0] = a->ne[1];
    ਨਤੀਜਾ->ne[1] = a->ne[0];

    ਨਤੀਜਾ->nb[0] = a->nb[1];
    ਨਤੀਜਾ->nb[1] = a->nb[0];

    ਨਤੀਜਾ->op     = GGML_OP_TRANSPOSE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_get_rows

ਬਣਤਰ ggml_tensor * ggml_get_ਪੰਕਤੀਆਂ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    ਗਿਣਤੀ ggml_type type = GGML_TYPE_F32;
    ਜੇ (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    ਨਤੀਜਾ->op     = GGML_OP_GET_ਪੰਕਤੀਆਂ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_get_rows_back

ਬਣਤਰ ggml_tensor * ggml_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(a) && ggml_is_ਸਦਿਸ਼(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    ਨਤੀਜਾ->op     = GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_set_rows

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_ਪੰਕਤੀਆਂ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(a));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(b));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
    ਨਤੀਜਾ->src[0] = b;
    ਨਤੀਜਾ->src[1] = c;
    ਨਤੀਜਾ->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_diag

ਬਣਤਰ ggml_tensor * ggml_diag(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ਅਚਲ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_DIAG;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_diag_mask_inf

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_inf_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_DIAG_ਨਕਾਬ_INF;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_inf(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past) {
    ਵਾਪਸ ggml_diag_ਨਕਾਬ_inf_impl(ctx, a, n_past, false);
}

ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_inf_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past) {
    ਵਾਪਸ ggml_diag_ਨਕਾਬ_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_zero_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_DIAG_ਨਕਾਬ_ZERO;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_zero(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past) {
    ਵਾਪਸ ggml_diag_ਨਕਾਬ_zero_impl(ctx, a, n_past, false);
}

ਬਣਤਰ ggml_tensor * ggml_diag_ਨਕਾਬ_zero_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_past) {
    ਵਾਪਸ ggml_diag_ਨਕਾਬ_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_clamp_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਦਸ਼ਮਲਵ                 min,
        ਦਸ਼ਮਲਵ                 max,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਦਸ਼ਮਲਵ params[] = { min, max };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_CLAMP;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_clamp(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਦਸ਼ਮਲਵ                 min,
    ਦਸ਼ਮਲਵ                 max) {
    ਵਾਪਸ ggml_clamp_impl(ctx, a, min, max, false);
}

ਬਣਤਰ ggml_tensor * ggml_clamp_inplace(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਦਸ਼ਮਲਵ                 min,
    ਦਸ਼ਮਲਵ                 max) {
    ਵਾਪਸ ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_soft_max_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    ਜੇ (ਨਕਾਬ) {
        GGML_ASSERT(ਨਕਾਬ->type == GGML_TYPE_F16 || ਨਕਾਬ->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(ਨਕਾਬ));
        GGML_ASSERT(ਨਕਾਬ->ne[0] == a->ne[0]);
        GGML_ASSERT(ਨਕਾਬ->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%ਨਕਾਬ->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%ਨਕਾਬ->ne[3] == 0);
    }

    ਜੇ (max_bias > 0.0f) {
        GGML_ASSERT(ਨਕਾਬ);
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਦਸ਼ਮਲਵ params[] = { scale, max_bias };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_SOFT_MAX;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = ਨਕਾਬ;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_soft_max(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_soft_max_impl(ctx, a, ਨਿਰਾ, 1.0f, 0.0f, false);
}

ਬਣਤਰ ggml_tensor * ggml_soft_max_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a) {
    ਵਾਪਸ ggml_soft_max_impl(ctx, a, ਨਿਰਾ, 1.0f, 0.0f, true);
}

ਬਣਤਰ ggml_tensor * ggml_soft_max_ext(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias) {
    ਵਾਪਸ ggml_soft_max_impl(ctx, a, ਨਕਾਬ, scale, max_bias, false);
}

ਬਣਤਰ ggml_tensor * ggml_soft_max_ext_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias) {
    ਵਾਪਸ ggml_soft_max_impl(ctx, a, ਨਕਾਬ, scale, max_bias, true);
}

ਖਾਲੀ ggml_soft_max_ਜੋੜੋ_sinks(
        ਬਣਤਰ ggml_tensor * a,
        ਬਣਤਰ ggml_tensor * sinks) {
    ਜੇ (!sinks) {
        a->src[2] = ਨਿਰਾ;
        ਵਾਪਸ;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ਨਿਰਾ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_soft_max_ext_ਪਿੱਠ_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias,
        bool                  inplace) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_SOFT_MAX_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਦਸ਼ਮਲਵ *) ਨਤੀਜਾ->op_params + 0, &scale,    ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਦਸ਼ਮਲਵ *) ਨਤੀਜਾ->op_params + 1, &max_bias, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_soft_max_ext_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias) {
    ਵਾਪਸ ggml_soft_max_ext_ਪਿੱਠ_impl(ctx, a, b, scale, max_bias, false);
}

ਬਣਤਰ ggml_tensor * ggml_soft_max_ext_ਪਿੱਠ_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias) {
    ਵਾਪਸ ggml_soft_max_ext_ਪਿੱਠ_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_rope_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   sections[GGML_MROPE_SECTIONS],
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((ਬਹੁਲਕ & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_ਸਦਿਸ਼(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_ਵਰਤਿਆ = ਬਹੁਲਕ & GGML_ROPE_TYPE_MROPE;
    ਜੇ (mrope_ਵਰਤਿਆ) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    ਜੇ (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, ਬਹੁਲਕ, /*n_ctx*/ 0, n_ctx_orig };
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params +  5, &freq_ਅਧਾਰ,    ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params +  6, &freq_scale,   ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params +  7, &ext_ਗੁਣਕ,   ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params +  8, &attn_ਗੁਣਕ,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params +  9, &beta_fast,    ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(params + 10, &beta_slow,    ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (mrope_ਵਰਤਿਆ && sections) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(params + 11, sections,  ਆਕਾਰ(int32_t) * GGML_MROPE_SECTIONS);
    } ਨਹੀਂਤਾਂ {
        memset(params + 11, 0,         ਆਕਾਰ(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_ROPE;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;
    ਨਤੀਜਾ->src[2] = c;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_rope(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, ਨਿਰਾ, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_multi(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   sections[GGML_MROPE_SECTIONS],
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, false
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_multi_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   sections[GGML_MROPE_SECTIONS],
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, true
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, ਨਿਰਾ, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_ext(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, c, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, false
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_ext_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, c, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, true
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_custom(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, ਨਿਰਾ, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, false
    );
}

ਬਣਤਰ ggml_tensor * ggml_rope_custom_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਵਾਪਸ ggml_rope_impl(
        ctx, a, b, ਨਿਰਾ, n_dims, ਨਿਰਾ, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale,
        ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ਸਥਿਰ ਦਸ਼ਮਲਵ ggml_rope_yarn_corr_dim(ਪੂਰਨ n_dims, ਪੂਰਨ n_ctx_orig, ਦਸ਼ਮਲਵ n_rot, ਦਸ਼ਮਲਵ ਅਧਾਰ) {
    ਵਾਪਸ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ਦਸ਼ਮਲਵ)M_PI)) / (2 * logf(ਅਧਾਰ));
}

ਖਾਲੀ ggml_rope_yarn_corr_dims(
    ਪੂਰਨ n_dims, ਪੂਰਨ n_ctx_orig, ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, ਦਸ਼ਮਲਵ beta_fast, ਦਸ਼ਮਲਵ beta_slow, ਦਸ਼ਮਲਵ dims[2]
) {
    // start and end correction dims
    ਦਸ਼ਮਲਵ ਸ਼ੁਰੂ = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ਅਧਾਰ));
    ਦਸ਼ਮਲਵ ਸਿਰਾ   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ਅਧਾਰ));
    dims[0] = MAX(0, ਸ਼ੁਰੂ);
    dims[1] = MIN(n_dims - 1, ਸਿਰਾ);
}

// ggml_rope_back

ਬਣਤਰ ggml_tensor * ggml_rope_ext_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_rope_ext(
        ctx, a, b, c, n_dims, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
    ਨਤੀਜਾ->op = GGML_OP_ROPE_ਪਿੱਠ;
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_rope_multi_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c,
        ਪੂਰਨ                   n_dims,
        ਪੂਰਨ                   sections[4],
        ਪੂਰਨ                   ਬਹੁਲਕ,
        ਪੂਰਨ                   n_ctx_orig,
        ਦਸ਼ਮਲਵ                 freq_ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 freq_scale,
        ਦਸ਼ਮਲਵ                 ext_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 attn_ਗੁਣਕ,
        ਦਸ਼ਮਲਵ                 beta_fast,
        ਦਸ਼ਮਲਵ                 beta_slow) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
    ਨਤੀਜਾ->op = GGML_OP_ROPE_ਪਿੱਠ;
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_rope_ਸੈੱਟ_ਵਿੱਥ(
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_ਪਿੱਠ);
    GGML_ASSERT(n_offs >= 0);

    ਅਚਲ int32_t ਬਹੁਲਕ = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(ਬਹੁਲਕ != GGML_ROPE_TYPE_VISION);

    ggml_ਸੈੱਟ_op_params_i32(a, 15, n_offs);
    ਵਾਪਸ a;
}

ਸਥਿਰ int64_t ggml_calc_conv_output_ਮਾਪ(int64_t ins, int64_t ks, ਪੂਰਨ s, ਪੂਰਨ p, ਪੂਰਨ d) {
    ਵਾਪਸ (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ਬਣਤਰ ggml_tensor * ggml_im2col(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   d0,
        ਪੂਰਨ                   d1,
        bool                  is_2D,
        ਗਿਣਤੀ ggml_type        dst_type) {
    ਜੇ (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } ਨਹੀਂਤਾਂ {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ਅਚਲ int64_t OH = is_2D ? ggml_calc_conv_output_ਮਾਪ(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ਅਚਲ int64_t OW =         ggml_calc_conv_output_ਮਾਪ(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ਅਚਲ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_IM2_ਖਾਨਾ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_im2col_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        int64_t             * ne,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   d0,
        ਪੂਰਨ                   d1,
        bool                  is_2D) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_1d

ਬਣਤਰ ggml_tensor * ggml_conv_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   d0) {
    ਬਣਤਰ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    ਨਤੀਜਾ = ggml_reshape_3d(ctx, ਨਤੀਜਾ, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    ਨਤੀਜਾ = ggml_cont(ctx, ggml_ਕ੍ਰਮ_ਬਦਲੋ(ctx, ਨਤੀਜਾ, 0, 2, 1, 3));                // [N, OC, OL]

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_1d_ph

ਬਣਤਰ ggml_tensor* ggml_conv_1d_ph(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s,
        ਪੂਰਨ                   d) {
    ਵਾਪਸ ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ਬਣਤਰ ggml_tensor * ggml_conv_1d_dw(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   d0) {
    ਬਣਤਰ ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ਬਣਤਰ ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_mul_mat(ctx, im2col, a);

    ਨਤੀਜਾ = ggml_reshape_3d(ctx, ਨਤੀਜਾ, ਨਤੀਜਾ->ne[0], ਨਤੀਜਾ->ne[2], 1);

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_1d_dw_ph

ਬਣਤਰ ggml_tensor * ggml_conv_1d_dw_ph(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   d0) {
    ਵਾਪਸ ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ਬਣਤਰ ggml_tensor * ggml_col2im_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   oc,
        ਪੂਰਨ                   p0) {
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ਅਚਲ int64_t K_OC = a->ne[0];
    ਅਚਲ int64_t T_ਵਿੱਚ = a->ne[1];
    ਅਚਲ int64_t K = K_OC / oc;
    ਅਚਲ int64_t T_ਬਾਹਰਲਾ = (T_ਵਿੱਚ - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_ਬਾਹਰਲਾ > 0);

    ਅਚਲ int64_t ne[4] = { T_ਬਾਹਰਲਾ, oc, 1, 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_COL2IM_1D;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_transpose_1d

ਸਥਿਰ int64_t ggml_calc_conv_transpose_1d_output_ਮਾਪ(int64_t ins, int64_t ks, ਪੂਰਨ s, ਪੂਰਨ p, ਪੂਰਨ d) {
    ਵਾਪਸ (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ਬਣਤਰ ggml_tensor * ggml_conv_transpose_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   d0) {
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ਅਚਲ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_ਮਾਪ(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_CONV_TRANSPOSE_1D;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ਬਣਤਰ ggml_tensor * ggml_conv_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   d0,
        ਪੂਰਨ                   d1) {
    ਬਣਤਰ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    ਨਤੀਜਾ = ggml_reshape_4d(ctx, ਨਤੀਜਾ, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    ਨਤੀਜਾ = ggml_cont(ctx, ggml_ਕ੍ਰਮ_ਬਦਲੋ(ctx, ਨਤੀਜਾ, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ਵਾਪਸ ਨਤੀਜਾ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ਬਣਤਰ ggml_tensor * ggml_im2col_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        int64_t               IC,
        ਪੂਰਨ                   s0, // stride width
        ਪੂਰਨ                   s1, // stride height
        ਪੂਰਨ                   s2, // stride depth
        ਪੂਰਨ                   p0, // padding width
        ਪੂਰਨ                   p1, // padding height
        ਪੂਰਨ                   p2, // padding depth
        ਪੂਰਨ                   d0, // dilation width
        ਪੂਰਨ                   d1, // dilation height
        ਪੂਰਨ                   d2, // dilation depth
        ਗਿਣਤੀ ggml_type        dst_type) {
    ਅਚਲ int64_t N = b->ne[3] / IC;
    ਅਚਲ int64_t ID = b->ne[2];
    ਅਚਲ int64_t IH = b->ne[1];
    ਅਚਲ int64_t IW = b->ne[0];

    ਅਚਲ int64_t OC = a->ne[3] / IC;
    ਅਣਵਰਤਿਆ(OC);
    ਅਚਲ int64_t KD = a->ne[2];
    ਅਚਲ int64_t KH = a->ne[1];
    ਅਚਲ int64_t KW = a->ne[0];
    ਅਚਲ int64_t OD = ggml_calc_conv_output_ਮਾਪ(ID, KD, s2, p2, d2);
    ਅਚਲ int64_t OH = ggml_calc_conv_output_ਮਾਪ(IH, KH, s1, p1, d1);
    ਅਚਲ int64_t OW = ggml_calc_conv_output_ਮਾਪ(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ਅਚਲ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_IM2_ਖਾਨਾ_3_D;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ਬਣਤਰ ggml_tensor * ggml_conv_3d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        int64_t               IC,
        ਪੂਰਨ                   s0, // stride width
        ਪੂਰਨ                   s1, // stride height
        ਪੂਰਨ                   s2, // stride depth
        ਪੂਰਨ                   p0, // padding width
        ਪੂਰਨ                   p1, // padding height
        ਪੂਰਨ                   p2, // padding depth
        ਪੂਰਨ                   d0, // dilation width
        ਪੂਰਨ                   d1, // dilation height
        ਪੂਰਨ                   d2  // dilation depth
        ) {
    ਬਣਤਰ ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    ਨਤੀਜਾ = ggml_reshape_4d(ctx, ਨਤੀਜਾ, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    ਨਤੀਜਾ = ggml_cont(ctx, ggml_ਕ੍ਰਮ_ਬਦਲੋ(ctx, ਨਤੀਜਾ, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    ਨਤੀਜਾ = ggml_reshape_4d(ctx, ਨਤੀਜਾ, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_2d_sk_p0

ਬਣਤਰ ggml_tensor * ggml_conv_2d_sk_p0(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ਬਣਤਰ ggml_tensor * ggml_conv_2d_s1_ph(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    ਵਾਪਸ ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ਬਣਤਰ ggml_tensor * ggml_conv_2d_dw(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   d0,
        ਪੂਰਨ                   d1) {
    ਬਣਤਰ ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ਬਣਤਰ ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ਬਣਤਰ ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_mul_mat(ctx, new_a, new_b);
    ਨਤੀਜਾ = ggml_reshape_4d(ctx, ਨਤੀਜਾ, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_2d_dw_direct

ਬਣਤਰ ggml_tensor * ggml_conv_2d_dw_ਸਿੱਧਾ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   stride0,
        ਪੂਰਨ                   stride1,
        ਪੂਰਨ                   pad0,
        ਪੂਰਨ                   pad1,
        ਪੂਰਨ                   dilation0,
        ਪੂਰਨ                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ਮਾਪ(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_ਮਾਪ(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, b->type, 4, ne);

    ਜੇ (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ਅਚਲ int64_t type_ਮਾਪ = ggml_type_ਮਾਪ(ਨਤੀਜਾ->type);
        GGML_ASSERT(ggml_blck_ਮਾਪ(ਨਤੀਜਾ->type) == 1);
        ਨਤੀਜਾ->nb[0] = ਨਤੀਜਾ->ne[2] * type_ਮਾਪ;
        ਨਤੀਜਾ->nb[1] = ਨਤੀਜਾ->ne[0] * ਨਤੀਜਾ->nb[0];
        ਨਤੀਜਾ->nb[2] = type_ਮਾਪ;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_CONV_2D_DW;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;
    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_2d_direct

ਬਣਤਰ ggml_tensor * ggml_conv_2d_ਸਿੱਧਾ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ਬਣਤਰ ggml_tensor  * b,   // input data [W, H, C, N]
        ਪੂਰਨ                   s0,  // stride dimension 0
        ਪੂਰਨ                   s1,  // stride dimension 1
        ਪੂਰਨ                   p0,  // padding dimension 0
        ਪੂਰਨ                   p1,  // padding dimension 1
        ਪੂਰਨ                   d0,  // dilation dimension 0
        ਪੂਰਨ                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ਮਾਪ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ਮਾਪ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, s0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, s1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 2, p0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 3, p1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 4, d0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 5, d1);

    ਨਤੀਜਾ->op = GGML_OP_CONV_2D;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_3d_direct

ਬਣਤਰ ggml_tensor * ggml_conv_3d_ਸਿੱਧਾ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਪੂਰਨ                   s2,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   p2,
        ਪੂਰਨ                   d0,
        ਪੂਰਨ                   d1,
        ਪੂਰਨ                   d2,
        ਪੂਰਨ                   c,
        ਪੂਰਨ                   n,
        ਪੂਰਨ                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ਮਾਪ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ਮਾਪ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_ਮਾਪ(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0,  s0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1,  s1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 2,  s2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 3,  p0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 4,  p1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 5,  p2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 6,  d0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 7,  d1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 8,  d2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 9,  c);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 10, n);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 11, oc);

    ਨਤੀਜਾ->op = GGML_OP_CONV_3D;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_conv_transpose_2d_p0

ਸਥਿਰ int64_t ggml_calc_conv_transpose_output_ਮਾਪ(int64_t ins, int64_t ks, ਪੂਰਨ s, ਪੂਰਨ p) {
    ਵਾਪਸ (ins - 1) * s - 2 * p + ks;
}

ਬਣਤਰ ggml_tensor * ggml_conv_transpose_2d_p0(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਪੂਰਨ                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ਅਚਲ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_ਮਾਪ(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_ਮਾਪ(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ਬਣਤਰ ggml_tensor* ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, stride);

    ਨਤੀਜਾ->op     = GGML_OP_CONV_TRANSPOSE_2D;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_pool_*

ਸਥਿਰ int64_t ggml_calc_pool_output_ਮਾਪ(int64_t ins, ਪੂਰਨ ks, ਪੂਰਨ s, ਦਸ਼ਮਲਵ p) {
    ਵਾਪਸ (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ਬਣਤਰ ggml_tensor * ggml_pool_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_op_pool     op,
        ਪੂਰਨ                   k0,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   p0) {
    ਅਚਲ int64_t ne[4] = {
        ggml_calc_pool_output_ਮਾਪ(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_POOL_1D;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_pool_2d

ਬਣਤਰ ggml_tensor * ggml_pool_2d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_op_pool     op,
        ਪੂਰਨ                   k0,
        ਪੂਰਨ                   k1,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਦਸ਼ਮਲਵ                 p0,
        ਦਸ਼ਮਲਵ                 p1) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ;
    ਅਚਲ int64_t ne[4] = {
        ggml_calc_pool_output_ਮਾਪ(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_ਮਾਪ(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_POOL_2D;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_pool_2d_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * af,
        ਗਿਣਤੀ ggml_op_pool     op,
        ਪੂਰਨ                   k0,
        ਪੂਰਨ                   k1,
        ਪੂਰਨ                   s0,
        ਪੂਰਨ                   s1,
        ਦਸ਼ਮਲਵ                 p0,
        ਦਸ਼ਮਲਵ                 p1) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ;
    ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_POOL_2_D_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = af;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_upscale / ggml_interpolate

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_interpolate_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              ਬਹੁਲਕ) {
    GGML_ASSERT((ਬਹੁਲਕ & 0xFF) < GGML_SCALE_ਬਹੁਲਕ_ਗਿਣਤੀ);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(ਬਹੁਲਕ & GGML_SCALE_FLAG_ANTIALIAS) || (ਬਹੁਲਕ & 0xFF) == GGML_SCALE_ਬਹੁਲਕ_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t)ਬਹੁਲਕ);

    ਨਤੀਜਾ->op     = GGML_OP_UPSCALE;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_upscale(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   scale_ਗੁਣਕ,
        ਗਿਣਤੀ ggml_scale_ਬਹੁਲਕ  ਬਹੁਲਕ) {
    GGML_ASSERT(scale_ਗੁਣਕ > 1);
    ਵਾਪਸ ggml_interpolate_impl(ctx, a, a->ne[0] * scale_ਗੁਣਕ, a->ne[1] * scale_ਗੁਣਕ, a->ne[2], a->ne[3], ਬਹੁਲਕ);
}

ਬਣਤਰ ggml_tensor * ggml_upscale_ext(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   ne0,
        ਪੂਰਨ                   ne1,
        ਪੂਰਨ                   ne2,
        ਪੂਰਨ                   ne3,
        ਗਿਣਤੀ ggml_scale_ਬਹੁਲਕ  ਬਹੁਲਕ) {
    ਵਾਪਸ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, ਬਹੁਲਕ);
}

ਬਣਤਰ ggml_tensor * ggml_interpolate(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              ਬਹੁਲਕ) {
    ਵਾਪਸ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, ਬਹੁਲਕ);
}

// ggml_pad

ਬਣਤਰ ggml_tensor * ggml_pad(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   p2,
        ਪੂਰਨ                   p3) {
    ਵਾਪਸ ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ਬਣਤਰ ggml_tensor * ggml_pad_circular(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1,
        ਪੂਰਨ                   p2,
        ਪੂਰਨ                   p3) {
    ਵਾਪਸ ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ਬਣਤਰ ggml_tensor * ggml_pad_ext(
            ਬਣਤਰ ggml_context * ctx,
            ਬਣਤਰ ggml_tensor  * a,
            ਪੂਰਨ                  lp0,
            ਪੂਰਨ                  rp0,
            ਪੂਰਨ                  lp1,
            ਪੂਰਨ                  rp1,
            ਪੂਰਨ                  lp2,
            ਪੂਰਨ                  rp2,
            ਪੂਰਨ                  lp3,
            ਪੂਰਨ                  rp3
            ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, lp0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, rp0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 2, lp1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 3, rp1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 4, lp2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 5, rp2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 6, lp3);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 7, rp3);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 8, 0); // not circular by default


    ਨਤੀਜਾ->op     = GGML_OP_PAD;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_pad_ext_circular

ਬਣਤਰ ggml_tensor * ggml_pad_ext_circular(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                  lp0,
        ਪੂਰਨ                  rp0,
        ਪੂਰਨ                  lp1,
        ਪੂਰਨ                  rp1,
        ਪੂਰਨ                  lp2,
        ਪੂਰਨ                  rp2,
        ਪੂਰਨ                  lp3,
        ਪੂਰਨ                  rp3
        ) {
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 8, 1); // circular
    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_pad_reflect_1d

ਬਣਤਰ ggml_tensor * ggml_pad_reflect_1d(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   p0,
        ਪੂਰਨ                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_PAD_REFLECT_1D;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_roll

ਬਣਤਰ ggml_tensor * ggml_roll(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   shift0,
        ਪੂਰਨ                   shift1,
        ਪੂਰਨ                   shift2,
        ਪੂਰਨ                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_ਮਾਪ(a->type));
    GGML_ASSERT(ਨਿਰਪੇਖ(shift0) < a->ne[0]);
    GGML_ASSERT(ਨਿਰਪੇਖ(shift1) < a->ne[1]);
    GGML_ASSERT(ਨਿਰਪੇਖ(shift2) < a->ne[2]);
    GGML_ASSERT(ਨਿਰਪੇਖ(shift3) < a->ne[3]);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, shift0);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, shift1);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 2, shift2);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 3, shift3);

    ਨਤੀਜਾ->op     = GGML_OP_ROLL;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_timestep_embedding

ਬਣਤਰ ggml_tensor * ggml_timestep_embedding(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * timesteps,
        ਪੂਰਨ                   dim,
        ਪੂਰਨ                   max_period) {

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, dim);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, max_period);

    ਨਤੀਜਾ->op     = GGML_OP_TIMESTEP_EMBEDDING;
    ਨਤੀਜਾ->src[0] = timesteps;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_tri

ਬਣਤਰ ggml_tensor * ggml_tri(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਗਿਣਤੀ ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, type);

    ਨਤੀਜਾ->op = GGML_OP_TRI;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_fill

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_fill_impl(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਦਸ਼ਮਲਵ                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, c);

    ਨਤੀਜਾ->op = GGML_OP_FILL;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_fill(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਦਸ਼ਮਲਵ                 c) {
    ਵਾਪਸ ggml_fill_impl(ctx, a, c, false);
}

ਬਣਤਰ ggml_tensor * ggml_fill_inplace(
    ਬਣਤਰ ggml_context * ctx,
    ਬਣਤਰ ggml_tensor  * a,
    ਦਸ਼ਮਲਵ                 c) {
    ਵਾਪਸ ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ਬਣਤਰ ggml_tensor * ggml_argsort(
        ਬਣਤਰ ggml_context  * ctx,
        ਬਣਤਰ ggml_tensor   * a,
        ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ   ਕ੍ਰਮ) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) ਕ੍ਰਮ);

    ਨਤੀਜਾ->op     = GGML_OP_ARGSORT;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_argsort_top_k

ਬਣਤਰ ggml_tensor * ggml_argsort_ਸਿਖਰ_k(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_argsort(ctx, a, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC);

    ਨਤੀਜਾ = ggml_view_4d(ctx, ਨਤੀਜਾ,
                k, ਨਤੀਜਾ->ne[1], ਨਤੀਜਾ->ne[2], ਨਤੀਜਾ->ne[3],
                   ਨਤੀਜਾ->nb[1], ਨਤੀਜਾ->nb[2], ਨਤੀਜਾ->nb[3],
                0);

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_top_k

ਬਣਤਰ ggml_tensor * ggml_ਸਿਖਰ_k(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    ਨਤੀਜਾ->op     = GGML_OP_ਸਿਖਰ_K;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_arange

ਬਣਤਰ ggml_tensor * ggml_arange(
        ਬਣਤਰ ggml_context * ctx,
        ਦਸ਼ਮਲਵ                 ਸ਼ੁਰੂ,
        ਦਸ਼ਮਲਵ                 stop,
        ਦਸ਼ਮਲਵ                 ਕਦਮ) {
    GGML_ASSERT(stop > ਸ਼ੁਰੂ);

    ਅਚਲ int64_t ਕਦਮਾਂ = (int64_t) ceilf((stop - ਸ਼ੁਰੂ) / ਕਦਮ);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ਕਦਮਾਂ);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, ਸ਼ੁਰੂ);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 1, stop);
    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 2, ਕਦਮ);

    ਨਤੀਜਾ->op = GGML_OP_ARANGE;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_flash_attn_ext

ਬਣਤਰ ggml_tensor * ggml_flash_attn_ext(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * q,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ,
        ਦਸ਼ਮਲਵ                 scale,
        ਦਸ਼ਮਲਵ                 max_bias,
        ਦਸ਼ਮਲਵ                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    ਜੇ (ਨਕਾਬ) {
        GGML_ASSERT(ਨਕਾਬ->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(ਨਕਾਬ));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % ਨਕਾਬ->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % ਨਕਾਬ->ne[3] == 0);
    }

    ਜੇ (max_bias > 0.0f) {
        GGML_ASSERT(ਨਕਾਬ);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਦਸ਼ਮਲਵ params[] = { scale, max_bias, logit_softcap };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_FLASH_ATTN_EXT;
    ਨਤੀਜਾ->src[0] = q;
    ਨਤੀਜਾ->src[1] = k;
    ਨਤੀਜਾ->src[2] = v;
    ਨਤੀਜਾ->src[3] = ਨਕਾਬ;

    ਵਾਪਸ ਨਤੀਜਾ;
}


ਖਾਲੀ ggml_flash_attn_ext_ਸੈੱਟ_prec(
        ਬਣਤਰ ggml_tensor * a,
        ਗਿਣਤੀ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ਅਚਲ int32_t prec_i32 = (int32_t) prec;

    ggml_ਸੈੱਟ_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

ਗਿਣਤੀ ggml_prec ggml_flash_attn_ext_get_prec(
        ਅਚਲ ਬਣਤਰ ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ਅਚਲ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ਵਾਪਸ (ਗਿਣਤੀ ggml_prec) prec_i32;
}

ਖਾਲੀ ggml_flash_attn_ext_ਸੈੱਟ_n_kv_max(
        ਬਣਤਰ ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_ਸੈੱਟ_op_params_i32(a, 4, n_kv_max);
}

ਖਾਲੀ ggml_flash_attn_ext_ਜੋੜੋ_sinks(
        ਬਣਤਰ ggml_tensor * a,
        ਬਣਤਰ ggml_tensor * sinks) {
    ਜੇ (!sinks) {
        a->src[4] = ਨਿਰਾ;
        ਵਾਪਸ;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ਨਿਰਾ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ਬਣਤਰ ggml_tensor * ggml_flash_attn_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * q,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ਅਚਲ int64_t     D = q->ne[0];
    ਅਚਲ int64_t     N = q->ne[1];
    ਅਚਲ int64_t     M = k->ne[1];
    ਅਚਲ int64_t   ne2 = q->ne[2];
    ਅਚਲ int64_t   ne3 = q->ne[3];
    ਅਚਲ int64_t kvne2 = k->ne[2];

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
    ਅਚਲ int64_t elem_q = ggml_nelements(q);
    ਅਚਲ int64_t elem_k = ggml_nelements(k);
    ਅਚਲ int64_t elem_v = ggml_nelements(v);

    ਗਿਣਤੀ ggml_type ਨਤੀਜਾ_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_ਮਾਪ(ਨਤੀਜਾ_type) == 1);
    ਅਚਲ size_t tsize = ggml_type_ਮਾਪ(ਨਤੀਜਾ_type);

    ਅਚਲ size_t offs_q = 0;
    ਅਚਲ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ਅਚਲ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ਅਚਲ size_t ਸਿਰਾ    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ਅਚਲ size_t nelements = (ਸਿਰਾ + tsize - 1)/tsize;

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &masked_i, ਆਕਾਰ(masked_i));

    ਨਤੀਜਾ->op     = GGML_OP_FLASH_ATTN_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = q;
    ਨਤੀਜਾ->src[1] = k;
    ਨਤੀਜਾ->src[2] = v;
    ਨਤੀਜਾ->src[3] = d;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_ssm_conv

ਬਣਤਰ ggml_tensor * ggml_ssm_conv(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * sx,
        ਬਣਤਰ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(c));

    ਅਚਲ int64_t d_conv  = c->ne[0];
    ਅਚਲ int64_t d_inner = c->ne[1];
    ਅਚਲ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ਅਚਲ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    ਨਤੀਜਾ->op     = GGML_OP_SSM_CONV;
    ਨਤੀਜਾ->src[0] = sx;
    ਨਤੀਜਾ->src[1] = c;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_ssm_scan

ਬਣਤਰ ggml_tensor * ggml_ssm_ਜਾਂਚ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * s,
        ਬਣਤਰ ggml_tensor  * x,
        ਬਣਤਰ ggml_tensor  * dt,
        ਬਣਤਰ ggml_tensor  * A,
        ਬਣਤਰ ggml_tensor  * B,
        ਬਣਤਰ ggml_tensor  * C,
        ਬਣਤਰ ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_ਮਾਪ(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_ਮਾਪ(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_ਮਾਪ(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ਅਚਲ int64_t d_state      = s->ne[0];
        ਅਚਲ int64_t ਸਿਰ_dim     = x->ne[0];
        ਅਚਲ int64_t n_ਸਿਰ       = x->ne[1];
        ਅਚਲ int64_t n_seq_ਟੋਕਨ = x->ne[2];
        ਅਚਲ int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_ਸਿਰ);
        GGML_ASSERT(dt->ne[1] == n_seq_ਟੋਕਨ);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == ਸਿਰ_dim);
        GGML_ASSERT(s->ne[2] == n_ਸਿਰ);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_ਟੋਕਨ);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_ਸਦਿਸ਼(ids));
        GGML_ASSERT(A->ne[1] == n_ਸਿਰ);
        GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(A));

        ਜੇ (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    ਨਤੀਜਾ->op   = GGML_OP_SSM_ਜਾਂਚ;
    ਨਤੀਜਾ->src[0] = s;
    ਨਤੀਜਾ->src[1] = x;
    ਨਤੀਜਾ->src[2] = dt;
    ਨਤੀਜਾ->src[3] = A;
    ਨਤੀਜਾ->src[4] = B;
    ਨਤੀਜਾ->src[5] = C;
    ਨਤੀਜਾ->src[6] = ids;

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) K);

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_win_part

ਬਣਤਰ ggml_tensor * ggml_win_part(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ਅਚਲ ਪੂਰਨ px = (w - a->ne[1]%w)%w;
    ਅਚਲ ਪੂਰਨ py = (w - a->ne[2]%w)%w;

    ਅਚਲ ਪੂਰਨ npx = (px + a->ne[1])/w;
    ਅਚਲ ਪੂਰਨ npy = (py + a->ne[2])/w;
    ਅਚਲ ਪੂਰਨ np  = npx*npy;

    ਅਚਲ int64_t ne[4] = { a->ne[0], w, w, np, };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_WIN_PART;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_win_unpart

ਬਣਤਰ ggml_tensor * ggml_win_unpart(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   w0,
        ਪੂਰਨ                   h0,
        ਪੂਰਨ                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ਅਚਲ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_WIN_UNPART;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_get_rel_pos

ਬਣਤਰ ggml_tensor * ggml_get_rel_pos(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਪੂਰਨ                   qh,
        ਪੂਰਨ                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ਅਚਲ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    ਨਤੀਜਾ->op     = GGML_OP_GET_REL_POS;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_add_rel_pos

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_rel_pos_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * pw,
        ਬਣਤਰ ggml_tensor  * ph,
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

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, inplace ? 1 : 0);

    ਨਤੀਜਾ->op     = GGML_OP_ਜੋੜੋ_REL_POS;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = pw;
    ਨਤੀਜਾ->src[2] = ph;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_rel_pos(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * pw,
        ਬਣਤਰ ggml_tensor  * ph) {
    ਵਾਪਸ ggml_ਜੋੜੋ_rel_pos_impl(ctx, a, pw, ph, false);
}

ਬਣਤਰ ggml_tensor * ggml_ਜੋੜੋ_rel_pos_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * pw,
        ਬਣਤਰ ggml_tensor  * ph) {
    ਵਾਪਸ ggml_ਜੋੜੋ_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ਬਣਤਰ ggml_tensor * ggml_rwkv_wkv6(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * r,
        ਬਣਤਰ ggml_tensor  * tf,
        ਬਣਤਰ ggml_tensor  * td,
        ਬਣਤਰ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ਅਚਲ int64_t S = k->ne[0];
    ਅਚਲ int64_t H = k->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = k->ne[2];
    ਅਚਲ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ਅਚਲ int64_t ne[4] = { S * H, n_ਟੋਕਨ + S * n_seqs, 1, 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_RWKV_WKV6;
    ਨਤੀਜਾ->src[0] = k;
    ਨਤੀਜਾ->src[1] = v;
    ਨਤੀਜਾ->src[2] = r;
    ਨਤੀਜਾ->src[3] = tf;
    ਨਤੀਜਾ->src[4] = td;
    ਨਤੀਜਾ->src[5] = state;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_gated_linear_attn

ਬਣਤਰ ggml_tensor * ggml_gated_ਰੇਖਿਕ_attn(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * q,
        ਬਣਤਰ ggml_tensor  * g,
        ਬਣਤਰ ggml_tensor  * state,
        ਦਸ਼ਮਲਵ scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ਅਚਲ int64_t S = k->ne[0];
    ਅਚਲ int64_t H = k->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = k->ne[2];
    ਅਚਲ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ਅਚਲ int64_t ne[4] = { S * H, n_ਟੋਕਨ + S * n_seqs, 1, 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, scale);

    ਨਤੀਜਾ->op     = GGML_OP_GATED_ਰੇਖਿਕ_ATTN;
    ਨਤੀਜਾ->src[0] = k;
    ਨਤੀਜਾ->src[1] = v;
    ਨਤੀਜਾ->src[2] = q;
    ਨਤੀਜਾ->src[3] = g;
    ਨਤੀਜਾ->src[4] = state;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_rwkv_wkv7

ਬਣਤਰ ggml_tensor * ggml_rwkv_wkv7(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * r,
        ਬਣਤਰ ggml_tensor  * w,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ਅਚਲ int64_t S = k->ne[0];
    ਅਚਲ int64_t H = k->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = k->ne[2];
    ਅਚਲ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ਅਚਲ int64_t ne[4] = { S * H, n_ਟੋਕਨ + S * n_seqs, 1, 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op     = GGML_OP_RWKV_WKV7;
    ਨਤੀਜਾ->src[0] = r;
    ਨਤੀਜਾ->src[1] = w;
    ਨਤੀਜਾ->src[2] = k;
    ਨਤੀਜਾ->src[3] = v;
    ਨਤੀਜਾ->src[4] = a;
    ਨਤੀਜਾ->src[5] = b;
    ਨਤੀਜਾ->src[6] = state;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_unary

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_unary_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(a));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) op);

    ਨਤੀਜਾ->op     = GGML_OP_UNARY;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_unary(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_unary_op    op) {
    ਵਾਪਸ ggml_unary_impl(ctx, a, op, false);
}

ਬਣਤਰ ggml_tensor * ggml_unary_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਗਿਣਤੀ ggml_unary_op    op) {
    ਵਾਪਸ ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_map_custom1_impl(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਅਚਲ  ggml_custom1_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਬਣਤਰ ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_MAP_CUSTOM1;
    ਨਤੀਜਾ->src[0] = a;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_map_custom1(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਅਚਲ  ggml_custom1_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ਬਣਤਰ ggml_tensor * ggml_map_custom1_inplace(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਅਚਲ  ggml_custom1_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_map_custom2_impl(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਅਚਲ  ggml_custom2_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਬਣਤਰ ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_MAP_CUSTOM2;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_map_custom2(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਅਚਲ  ggml_custom2_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ਬਣਤਰ ggml_tensor * ggml_map_custom2_inplace(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਅਚਲ  ggml_custom2_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_map_custom3_impl(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਬਣਤਰ ggml_tensor       * c,
        ਅਚਲ  ggml_custom3_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ਬਣਤਰ ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op     = GGML_OP_MAP_CUSTOM3;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;
    ਨਤੀਜਾ->src[2] = c;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_map_custom3(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਬਣਤਰ ggml_tensor       * c,
        ਅਚਲ  ggml_custom3_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ਬਣਤਰ ggml_tensor * ggml_map_custom3_inplace(
        ਬਣਤਰ ggml_context      * ctx,
        ਬਣਤਰ ggml_tensor       * a,
        ਬਣਤਰ ggml_tensor       * b,
        ਬਣਤਰ ggml_tensor       * c,
        ਅਚਲ  ggml_custom3_op_t   fun,
        ਪੂਰਨ                        n_tasks,
        ਖਾਲੀ                     * userdata) {
    ਵਾਪਸ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ਬਣਤਰ ggml_tensor * ggml_custom_4d(
        ਬਣਤਰ ggml_context * ctx,
        ਗਿਣਤੀ ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ਬਣਤਰ ggml_tensor ** ਦਲੀਲਾਂ_2,
        ਪੂਰਨ                   n_ਦਲੀਲਾਂ,
        ggml_custom_op_t      fun,
        ਪੂਰਨ                   n_tasks,
        ਖਾਲੀ                * userdata) {

    GGML_ASSERT(n_ਦਲੀਲਾਂ < GGML_MAX_SRC);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ਬਣਤਰ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op = GGML_OP_CUSTOM;
    ਲਈ (ਪੂਰਨ i = 0; i < n_ਦਲੀਲਾਂ; i++) {
        ਨਤੀਜਾ->src[i] = ਦਲੀਲਾਂ_2[i];
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_custom_inplace(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor ** ਦਲੀਲਾਂ_2,
        ਪੂਰਨ                   n_ਦਲੀਲਾਂ,
        ggml_custom_op_t      fun,
        ਪੂਰਨ                   n_tasks,
        ਖਾਲੀ                * userdata) {

    GGML_ASSERT(n_ਦਲੀਲਾਂ < GGML_MAX_SRC - 1);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);

    ਬਣਤਰ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ਸੈੱਟ_op_params(ਨਤੀਜਾ, &params, ਆਕਾਰ(params));

    ਨਤੀਜਾ->op = GGML_OP_CUSTOM;
    ਨਤੀਜਾ->src[0] = a;
    ਲਈ (ਪੂਰਨ i = 0; i < n_ਦਲੀਲਾਂ; i++) {
        ਨਤੀਜਾ->src[i + 1] = ਦਲੀਲਾਂ_2[i];
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}
// ggml_cross_entropy_loss

ਬਣਤਰ ggml_tensor * ggml_cross_entropy_loss(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, a->type, 1);

    ਨਤੀਜਾ->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_cross_entropy_loss_back

ਬਣਤਰ ggml_tensor * ggml_cross_entropy_loss_ਪਿੱਠ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        ਬਣਤਰ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_dup_tensor(ctx, b);

    ਨਤੀਜਾ->op     = GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;
    ਨਤੀਜਾ->src[2] = c;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// opt_step_adamw

ਬਣਤਰ ggml_tensor * ggml_opt_ਕਦਮ_adamw(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * grad,
        ਬਣਤਰ ggml_tensor  * m,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_OPT_ਕਦਮ_ADAMW;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = grad;
    ਨਤੀਜਾ->src[2] = m;
    ਨਤੀਜਾ->src[3] = v;
    ਨਤੀਜਾ->src[4] = adamw_params;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// opt_step_sgd

ਬਣਤਰ ggml_tensor * ggml_opt_ਕਦਮ_sgd(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * grad,
        ਬਣਤਰ ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_view_tensor(ctx, a);

    ਨਤੀਜਾ->op     = GGML_OP_OPT_ਕਦਮ_SGD;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = grad;
    ਨਤੀਜਾ->src[2] = params;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// solve_tri

ਬਣਤਰ ggml_tensor * ggml_solve_tri(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * a,
        ਬਣਤਰ ggml_tensor  * b,
        bool                  ਖੱਬਾ,
        bool                  ਹੇਠਲਾ,
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

    GGML_ASSERT(ਹੇਠਲਾ && ਖੱਬਾ && !uni); // TODO: support other variants

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    ਨਤੀਜਾ->op     = GGML_OP_SOLVE_TRI;
    ਨਤੀਜਾ->src[0] = a;
    ਨਤੀਜਾ->src[1] = b;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_gated_delta_net

ਬਣਤਰ ggml_tensor * ggml_gated_delta_net(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * q,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * v,
        ਬਣਤਰ ggml_tensor  * g,
        ਬਣਤਰ ggml_tensor  * beta,
        ਬਣਤਰ ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(q));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(k));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    ਅਚਲ int64_t S_v      = v->ne[0];
    ਅਚਲ int64_t H        = v->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = v->ne[2];
    ਅਚਲ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ਅਚਲ int64_t state_ਪੰਕਤੀਆਂ = K * S_v * n_seqs;
    ਅਚਲ int64_t ne[4] = { S_v * H, n_ਟੋਕਨ * n_seqs + state_ਪੰਕਤੀਆਂ, 1, 1 };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 0, (int32_t) K);

    ਨਤੀਜਾ->op     = GGML_OP_GATED_DELTA_NET;
    ਨਤੀਜਾ->src[0] = q;
    ਨਤੀਜਾ->src[1] = k;
    ਨਤੀਜਾ->src[2] = v;
    ਨਤੀਜਾ->src[3] = g;
    ਨਤੀਜਾ->src[4] = beta;
    ਨਤੀਜਾ->src[5] = state;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_lightning_indexer

ਬਣਤਰ ggml_tensor * ggml_lightning_indexer(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * q,
        ਬਣਤਰ ggml_tensor  * k,
        ਬਣਤਰ ggml_tensor  * weights,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    ਨਕਾਬ->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   ਨਕਾਬ->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   ਨਕਾਬ->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   ਨਕਾਬ->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % ਨਕਾਬ->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ਨਤੀਜਾ->op   = GGML_OP_LIGHTNING_INDEXER;
    ਨਤੀਜਾ->src[0] = q;
    ਨਤੀਜਾ->src[1] = k;
    ਨਤੀਜਾ->src[2] = weights;
    ਨਤੀਜਾ->src[3] = ਨਕਾਬ;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_dsv4_hc_comb

ਬਣਤਰ ggml_tensor * ggml_dsv4_hc_comb(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * mixes,
        ਬਣਤਰ ggml_tensor  * scale,
        ਬਣਤਰ ggml_tensor  * ਅਧਾਰ,
        ਦਸ਼ਮਲਵ                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ਅਧਾਰ->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ਅਚਲ int64_t hc_mix_dim = mixes->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ   = mixes->ne[1];

    int64_t hc = 0;
    ਲਈ (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        ਜੇ ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ਰੋਕੋ;
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
    GGML_ASSERT(ਅਧਾਰ->ne[0] == hc_mix_dim);
    GGML_ASSERT(ਅਧਾਰ->ne[1] == 1);
    GGML_ASSERT(ਅਧਾਰ->ne[2] == 1);
    GGML_ASSERT(ਅਧਾਰ->ne[3] == 1);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_ਟੋਕਨ);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, eps);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, n_iter);

    ਨਤੀਜਾ->op     = GGML_OP_DSV4_HC_COMB;
    ਨਤੀਜਾ->src[0] = mixes;
    ਨਤੀਜਾ->src[1] = scale;
    ਨਤੀਜਾ->src[2] = ਅਧਾਰ;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_dsv4_hc_pre

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_dsv4_hc_pre_impl(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * x,
        ਬਣਤਰ ggml_tensor  * weights,
        ਦਸ਼ਮਲਵ                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t hc       = x->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    ਜੇ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ਟੋਕਨ);
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ਟੋਕਨ);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ਟੋਕਨ);

    ggml_ਸੈੱਟ_op_params_f32(ਨਤੀਜਾ, 0, scale);
    ggml_ਸੈੱਟ_op_params_i32(ਨਤੀਜਾ, 1, gated ? 1 : 0);

    ਨਤੀਜਾ->op     = GGML_OP_DSV4_HC_PRE;
    ਨਤੀਜਾ->src[0] = x;
    ਨਤੀਜਾ->src[1] = weights;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_dsv4_hc_pre(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * x,
        ਬਣਤਰ ggml_tensor  * weights) {
    ਵਾਪਸ ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ਬਣਤਰ ggml_tensor * ggml_dsv4_hc_pre_gated(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * x,
        ਬਣਤਰ ggml_tensor  * gate,
        ਦਸ਼ਮਲਵ                 scale) {
    ਵਾਪਸ ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ਬਣਤਰ ggml_tensor * ggml_dsv4_hc_post(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * x,
        ਬਣਤਰ ggml_tensor  * residual,
        ਬਣਤਰ ggml_tensor  * post,
        ਬਣਤਰ ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[1];
    ਅਚਲ int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ਟੋਕਨ);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ਟੋਕਨ);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    ਜੇ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ਟੋਕਨ);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_ਟੋਕਨ);

    ਨਤੀਜਾ->op     = GGML_OP_DSV4_HC_POST;
    ਨਤੀਜਾ->src[0] = x;
    ਨਤੀਜਾ->src[1] = residual;
    ਨਤੀਜਾ->src[2] = post;
    ਨਤੀਜਾ->src[3] = comb;

    ਵਾਪਸ ਨਤੀਜਾ;
}

// ggml_top_k_mask

ਬਣਤਰ ggml_tensor * ggml_ਸਿਖਰ_k_ਨਕਾਬ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_tensor  * ਨਕਾਬ,
        ਬਣਤਰ ggml_tensor  * ਸਿਖਰ_k,
        ਬਣਤਰ ggml_tensor  * score,
        ਬਣਤਰ ggml_tensor  * ਖੰਡਾਂ,
        ਬਣਤਰ ggml_tensor  * extra) {
    GGML_ASSERT(ਨਕਾਬ->type == GGML_TYPE_F16 || ਨਕਾਬ->type == GGML_TYPE_F32);
    GGML_ASSERT(ਨਕਾਬ->nb[0] == ggml_type_ਮਾਪ(ਨਕਾਬ->type));
    GGML_ASSERT(ਸਿਖਰ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(ਸਿਖਰ_k->ne[1] % ਨਕਾਬ->ne[1] == 0 && ਸਿਖਰ_k->ne[2] % ਨਕਾਬ->ne[2] == 0 && ਸਿਖਰ_k->ne[3] % ਨਕਾਬ->ne[3] == 0);

    ਜੇ (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == ਸਿਖਰ_k->ne[1] && score->ne[2] == ਸਿਖਰ_k->ne[2] && score->ne[3] == ਸਿਖਰ_k->ne[3]);
    }

    ਜੇ (ਖੰਡਾਂ) {
        GGML_ASSERT(ਖੰਡਾਂ->type == GGML_TYPE_I32);
        GGML_ASSERT(ਖੰਡਾਂ->ne[2] == 1 && ਖੰਡਾਂ->ne[3] == 1);
        GGML_ASSERT(!score || ਖੰਡਾਂ->ne[1] == score->ne[0]);
    }

    ਜੇ (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == ਸਿਖਰ_k->ne[1] && extra->ne[2] == ਸਿਖਰ_k->ne[2] && extra->ne[3] == ਸਿਖਰ_k->ne[3]);
    }

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, ਨਕਾਬ->type, ਨਕਾਬ->ne[0], ਸਿਖਰ_k->ne[1], ਸਿਖਰ_k->ne[2], ਸਿਖਰ_k->ne[3]);

    ਨਤੀਜਾ->op     = GGML_OP_ਸਿਖਰ_K_ਨਕਾਬ;
    ਨਤੀਜਾ->src[0] = ਨਕਾਬ;
    ਨਤੀਜਾ->src[1] = ਸਿਖਰ_k;
    ਨਤੀਜਾ->src[2] = score;
    ਨਤੀਜਾ->src[3] = ਖੰਡਾਂ;
    ਨਤੀਜਾ->src[4] = extra;

    ਵਾਪਸ ਨਤੀਜਾ;
}

////////////////////////////////////////////////////////////////////////////////

ਬਣਤਰ ggml_hash_ਸੈੱਟ ggml_hash_ਸੈੱਟ_new(size_t ਮਾਪ) {
    ਮਾਪ = ggml_hash_ਮਾਪ(ਮਾਪ);
    ਬਣਤਰ ggml_hash_ਸੈੱਟ ਨਤੀਜਾ;
    ਨਤੀਜਾ.ਮਾਪ = ਮਾਪ;
    ਨਤੀਜਾ.ਕੁੰਜੀਆਂ = GGML_MALLOC(ਆਕਾਰ(ਬਣਤਰ ggml_tensor *) * ਮਾਪ);
    ਨਤੀਜਾ.ਵਰਤਿਆ = GGML_CALLOC(ggml_bitset_ਮਾਪ(ਮਾਪ), ਆਕਾਰ(ggml_bitset_t));
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_hash_ਸੈੱਟ_reset(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ) {
    memset(hash_ਸੈੱਟ->ਵਰਤਿਆ, 0, ਆਕਾਰ(ggml_bitset_t) * ggml_bitset_ਮਾਪ(hash_ਸੈੱਟ->ਮਾਪ));
}

ਖਾਲੀ ggml_hash_ਸੈੱਟ_free(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ) {
    GGML_FREE(hash_ਸੈੱਟ->ਵਰਤਿਆ);
    GGML_FREE(hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ);
}

size_t ggml_hash_ਮਾਪ(size_t min_sz) {
    // next primes after powers of two
    ਸਥਿਰ ਅਚਲ size_t ਅਭਾਜ_ਸੰਖਿਆਵਾਂ[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ਸਥਿਰ ਅਚਲ size_t n_ਅਭਾਜ_ਸੰਖਿਆਵਾਂ = ਆਕਾਰ(ਅਭਾਜ_ਸੰਖਿਆਵਾਂ)/ਆਕਾਰ(ਅਭਾਜ_ਸੰਖਿਆਵਾਂ[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_ਅਭਾਜ_ਸੰਖਿਆਵਾਂ;
    ਜਦੋਂ (l < r) {
        size_t m = (l + r)/2;
        ਜੇ (ਅਭਾਜ_ਸੰਖਿਆਵਾਂ[m] < min_sz) {
            l = m + 1;
        } ਨਹੀਂਤਾਂ {
            r = m;
        }
    }
    size_t sz = l < n_ਅਭਾਜ_ਸੰਖਿਆਵਾਂ ? ਅਭਾਜ_ਸੰਖਿਆਵਾਂ[l] : min_sz | 1;
    ਵਾਪਸ sz;
}

ਬਣਤਰ hash_map {
    ਬਣਤਰ ggml_hash_ਸੈੱਟ ਸੈੱਟ;
    ਬਣਤਰ ggml_tensor ** vals;
};

ਸਥਿਰ ਬਣਤਰ hash_map * ggml_new_hash_map(size_t ਮਾਪ) {
    ਬਣਤਰ hash_map * ਨਤੀਜਾ = GGML_MALLOC(ਆਕਾਰ(ਬਣਤਰ hash_map));
    ਨਤੀਜਾ->ਸੈੱਟ = ggml_hash_ਸੈੱਟ_new(ਮਾਪ);
    ਨਤੀਜਾ->vals = GGML_CALLOC(ਨਤੀਜਾ->ਸੈੱਟ.ਮਾਪ, ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hash_map_free(ਬਣਤਰ hash_map * map) {
    ggml_hash_ਸੈੱਟ_free(&map->ਸੈੱਟ);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

ਸਥਿਰ ਖਾਲੀ ggml_ਜੋੜੋ_or_ਸੈੱਟ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_cgraph  * cgraph,
        size_t                isrc,
        ਬਣਤਰ ggml_tensor  * tensor) {
    ਬਣਤਰ ggml_tensor * src = cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[isrc];
    GGML_ASSERT(src);
    ਜੇ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_ਜੋੜੋ_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } ਨਹੀਂਤਾਂ {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ਸਥਿਰ ਖਾਲੀ ggml_acc_or_ਸੈੱਟ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_cgraph  * cgraph,
        size_t                isrc,
        ਬਣਤਰ ggml_tensor  * tensor,
        ਅਚਲ  size_t         nb1,
        ਅਚਲ  size_t         nb2,
        ਅਚਲ  size_t         nb3,
        ਅਚਲ  size_t         ਵਿੱਥ) {
    ਬਣਤਰ ggml_tensor * src = cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[isrc];
    GGML_ASSERT(src);
    ਜੇ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, ਵਿੱਥ, cgraph->grad_accs[isrc]);
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, ਵਿੱਥ, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ਸਥਿਰ ਖਾਲੀ ggml_add1_or_ਸੈੱਟ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_cgraph  * cgraph,
        size_t                isrc,
        ਬਣਤਰ ggml_tensor  * tensor) {
    ਬਣਤਰ ggml_tensor * src = cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[isrc];
    GGML_ASSERT(src);
    ਜੇ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ਨਹੀਂਤਾਂ {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ਸਥਿਰ ਖਾਲੀ ggml_sub_or_ਸੈੱਟ(
        ਬਣਤਰ ggml_context * ctx,
        ਬਣਤਰ ggml_cgraph  * cgraph,
        size_t                isrc,
        ਬਣਤਰ ggml_tensor  * tensor) {
    ਬਣਤਰ ggml_tensor * src = cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[isrc];
    GGML_ASSERT(src);
    ਜੇ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ਨਹੀਂਤਾਂ {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_backward(
        ਬਣਤਰ ggml_context * ctx, ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ i, ਅਚਲ bool * grads_needed) {
    ਬਣਤਰ ggml_tensor * tensor = cgraph->nodes[i];
    ਬਣਤਰ ggml_tensor * grad   = ggml_ਗ੍ਰਾਫ_get_grad(cgraph, tensor);

    ਜੇ (!grad) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ ggml_tensor * src0 = tensor->src[0];
    ਬਣਤਰ ggml_tensor * src1 = tensor->src[1];
    ਬਣਤਰ ggml_tensor * src2 = tensor->src[2];
    ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ = &cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ;
    ਅਚਲ size_t isrc0 = src0 ? ggml_hash_ਲੱਭੋ(hash_ਸੈੱਟ, src0) : (size_t) -1;
    ਅਚਲ size_t isrc1 = src1 ? ggml_hash_ਲੱਭੋ(hash_ਸੈੱਟ, src1) : (size_t) -1;
    ਅਚਲ size_t isrc2 = src2 ? ggml_hash_ਲੱਭੋ(hash_ਸੈੱਟ, src2) : (size_t) -1;
    ਅਚਲ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, isrc0) && grads_needed[isrc0];
    ਅਚਲ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, isrc1) && grads_needed[isrc1];
    ਅਚਲ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, isrc2) && grads_needed[isrc2];

    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_DUP: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
            ਜੇ (src1_needs_grads) {
                ਬਣਤਰ ggml_tensor * tmp = grad;
                ਜੇ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_ਪਿੱਠ(ctx, tmp, src1);
                }
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, tmp);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ADD1: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
            ਜੇ (src1_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_ਔਸਤ(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ACC: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
            ਜੇ (src1_needs_grads) {
                ਅਚਲ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ਅਚਲ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ਅਚਲ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ਅਚਲ size_t ਵਿੱਥ = ((int32_t *) tensor->op_params)[3];

                ਬਣਤਰ ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, ਵਿੱਥ);

                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
            ਜੇ (src1_needs_grads) {
                ggml_sub_or_ਸੈੱਟ(ctx, cgraph, isrc1, grad);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            ਜੇ (src1_needs_grads) {
                ਬਣਤਰ ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                ਜੇ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_ਪਿੱਠ(ctx, tmp, src1);
                }
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, tmp);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            ਜੇ (src1_needs_grads) {
                ggml_sub_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS: {
            ਜੇ (src0_needs_grads) {
                ggml_sub_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ: {
            ਜੇ (src0_needs_grads) {
                ggml_add1_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ: {
            ਜੇ (src0_needs_grads) {
                ggml_add1_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_repeat_ਪਿੱਠ(ctx, grad, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM: {
            ਜੇ (src0_needs_grads) {
                ਦਸ਼ਮਲਵ eps;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, tensor->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_rms_norm_ਪਿੱਠ(ctx, grad, src0, eps));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT: {
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

            ਜੇ (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ਬਣਤਰ ggml_tensor * tmp =
                    ggml_ਬਾਹਰਲਾ_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                ਜੇ (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ਅਚਲ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ਅਚਲ size_t nb2 = tmp->nb[2] * nr2;
                    ਅਚਲ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_ਪਿੱਠ(ctx, tmp, src0);
                }
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, tmp);
            }
            ਜੇ (src1_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_ਬਾਹਰਲਾ_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE: {
            ਜੇ (src0_needs_grads) {
                ਦਸ਼ਮਲਵ s;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&s, tensor->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ: {
            ਅਚਲ size_t nb1    = ((ਅਚਲ int32_t *) tensor->op_params)[0];
            ਅਚਲ size_t nb2    = ((ਅਚਲ int32_t *) tensor->op_params)[1];
            ਅਚਲ size_t nb3    = ((ਅਚਲ int32_t *) tensor->op_params)[2];
            ਅਚਲ size_t ਵਿੱਥ = ((ਅਚਲ int32_t *) tensor->op_params)[3];

            ਬਣਤਰ ggml_tensor * tensor_grad_view = ਨਿਰਾ;

            ਜੇ (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, ਵਿੱਥ);
            }

            ਜੇ (src0_needs_grads) {
                ਬਣਤਰ ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, ਵਿੱਥ, false));
            }

            ਜੇ (src1_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            ਜੇ (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            ਜੇ (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT: {
            // same as cpy
            ਜੇ (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RESHAPE: {
            ਜੇ (src0_needs_grads) {
                ਬਣਤਰ ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_VIEW: {
            ਜੇ (src0_needs_grads) {
                size_t ਵਿੱਥ;

                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਵਿੱਥ, tensor->op_params, ਆਕਾਰ(ਵਿੱਥ));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                ਜੇ (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_ਮਾਪ(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_ਮਾਪ(src0);
                    GGML_ASSERT(ਵਿੱਥ % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    ਵਿੱਥ = (ਵਿੱਥ / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, ਵਿੱਥ);
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ: {
            ਜੇ (src0_needs_grads) {
                ਅਚਲ int32_t * axes = (ਅਚਲ int32_t *) tensor->op_params;
                ਅਚਲ ਪੂਰਨ axis0 = axes[0] & 0x3;
                ਅਚਲ ਪੂਰਨ axis1 = axes[1] & 0x3;
                ਅਚਲ ਪੂਰਨ axis2 = axes[2] & 0x3;
                ਅਚਲ ਪੂਰਨ axis3 = axes[3] & 0x3;
                ਪੂਰਨ axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_ਕ੍ਰਮ_ਬਦਲੋ(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRANSPOSE: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ctx, grad, src1, src0));
            }
            ਜੇ (src1_needs_grads) {
                // noop
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF: {
            ਜੇ (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ਅਚਲ ਪੂਰਨ n_past = ((ਅਚਲ int32_t *) tensor->op_params)[0];
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_diag_ਨਕਾਬ_zero_impl(ctx, grad, n_past, false));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_ZERO: {
            ਜੇ (src0_needs_grads) {
                ਅਚਲ ਪੂਰਨ n_past = ((ਅਚਲ int32_t *) tensor->op_params)[0];
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_diag_ਨਕਾਬ_zero_impl(ctx, grad, n_past, false));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX: {
            ਜੇ (src0_needs_grads) {
                ਦਸ਼ਮਲਵ scale    = 1.0f;
                ਦਸ਼ਮਲਵ max_bias = 0.0f;

                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_soft_max_ext_ਪਿੱਠ(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE: {
            ਜੇ (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ਅਚਲ ਪੂਰਨ n_dims     = ((ਅਚਲ int32_t *) tensor->op_params)[1];
                ਅਚਲ ਪੂਰਨ ਬਹੁਲਕ       = ((ਅਚਲ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ਅਚਲ ਪੂਰਨ n_ctx_orig = ((ਅਚਲ int32_t *) tensor->op_params)[4];
                ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow;
                ਪੂਰਨ sections[4] = {0, 0, 0, 0};

                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ,   (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params +  5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale,  (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params +  6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ,  (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params +  7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params +  8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast,   (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params +  9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow,   (ਅਚਲ ਦਸ਼ਮਲਵ *) tensor->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&sections,                    tensor->op_params + 11, ਆਕਾਰ(sections));

                ਬਣਤਰ ggml_tensor * rope_ਪਿੱਠ = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_ਪਿੱਠ(ctx, grad, src1, src2, n_dims,
                        ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow) :
                    ggml_rope_multi_ਪਿੱਠ(ctx, grad, src1, src2, n_dims, sections,
                        ਬਹੁਲਕ, n_ctx_orig, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, rope_ਪਿੱਠ);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ: {
            ਜੇ (src1_needs_grads) {
                ਅਚਲ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ਅਚਲ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ਅਚਲ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ਅਚਲ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ਅਚਲ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ਅਚਲ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ਅਚਲ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_im2col_ਪਿੱਠ(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2D: {
            ਜੇ (src0_needs_grads) {
                ਅਚਲ ਗਿਣਤੀ ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ਅਚਲ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ਅਚਲ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ਅਚਲ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ਅਚਲ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ਅਚਲ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ਅਚਲ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_pool_2d_ਪਿੱਠ(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_WIN_PART:
        ਹਾਲਤ GGML_OP_WIN_UNPART:
        ਹਾਲਤ GGML_OP_UNARY: {
            ਚੋਣ (ggml_get_unary_op(tensor)) {
                ਹਾਲਤ GGML_UNARY_OP_ABS: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SGN: {
                    // noop
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_NEG: {
                    ਜੇ (src0_needs_grads) {
                        ggml_sub_or_ਸੈੱਟ(ctx, cgraph, isrc0, grad);
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ: {
                    // noop
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_RELU: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_ਕਦਮ(ctx, src0), grad));
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SILU: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_silu_ਪਿੱਠ(ctx, grad, src0));
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXP: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXPM1: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS: {
                    ਜੇ (src0_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ਰੋਕੋ;
                ਮੂਲ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS: {
            ਜੇ (src0_needs_grads) {
                ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_cross_entropy_loss_ਪਿੱਠ(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU: {
            ਚੋਣ (ggml_get_glu_op(tensor)) {
                ਹਾਲਤ GGML_GLU_OP_SWIGLU: {
                    ਜੇ (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc0, ggml_silu_ਪਿੱਠ(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    ਜੇ (src1_needs_grads) {
                        ggml_ਜੋੜੋ_or_ਸੈੱਟ(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ਰੋਕੋ;
                ਮੂਲ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE: {
            // noop
        } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ:
        ਮੂਲ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ਸਥਿਰ size_t ggml_ਦੇਖੋ_parents_ਗ੍ਰਾਫ(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_tensor * ਗੰਢ, bool compute) {
    ਜੇ (ਗੰਢ->op != GGML_OP_NONE && compute) {
        ਗੰਢ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ਅਚਲ size_t ਗੰਢ_hash_pos = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ);
    GGML_ASSERT(ਗੰਢ_hash_pos != GGML_HASHSET_ਭਰਿਆ);

    ਜੇ (ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, ਗੰਢ_hash_pos)) {
        // already visited

        ਜੇ (compute) {
            // update the compute flag regardless
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; ++i) {
                ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[i];
                ਜੇ (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_ਦੇਖੋ_parents_ਗ੍ਰਾਫ(cgraph, src, true);
                }
            }
        }

        ਵਾਪਸ ਗੰਢ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[ਗੰਢ_hash_pos] = ਗੰਢ;
    ggml_bitset_ਸੈੱਟ(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, ਗੰਢ_hash_pos);
    cgraph->use_counts[ਗੰਢ_hash_pos] = 0;

    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; ++i) {
        ਅਚਲ ਪੂਰਨ k =
            (cgraph->ਕ੍ਰਮ == GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਖੱਬਾ_ਮੰਜ਼ਿਲ_ਸੱਜਾ) ? i :
            (cgraph->ਕ੍ਰਮ == GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਸੱਜਾ_ਮੰਜ਼ਿਲ_ਖੱਬਾ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[k];
        ਜੇ (src) {
            ਅਚਲ size_t src_hash_pos = ggml_ਦੇਖੋ_parents_ਗ੍ਰਾਫ(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    ਜੇ (ਗੰਢ->op == GGML_OP_NONE && !(ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->ਮਾਪ);

        ਜੇ (ਲੜੀ_ਲੰਬਾਈ(ਗੰਢ->name) == 0) {
            ggml_format_name(ਗੰਢ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ਗੰਢ;
        cgraph->n_leafs++;
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(cgraph->n_nodes < cgraph->ਮਾਪ);

        ਜੇ (ਲੜੀ_ਲੰਬਾਈ(ਗੰਢ->name) == 0) {
            ggml_format_name(ਗੰਢ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ਗੰਢ;
        cgraph->n_nodes++;
    }

    ਵਾਪਸ ਗੰਢ_hash_pos;
}

ਸਥਿਰ ਖਾਲੀ ggml_build_forward_impl(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_tensor * tensor, bool expand, bool compute) {
    ਜੇ (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_ਗ੍ਰਾਫ_clear(cgraph);
    }

    ਅਚਲ ਪੂਰਨ n_old = cgraph->n_nodes;

    ggml_ਦੇਖੋ_parents_ਗ੍ਰਾਫ(cgraph, tensor, compute);

    ਅਚਲ ਪੂਰਨ n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    ਜੇ (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ਬਣਤਰ ggml_tensor * ggml_build_forward_select(
        ਬਣਤਰ ggml_cgraph  * cgraph,
        ਬਣਤਰ ggml_tensor ** tensors,
        ਪੂਰਨ                   n_tensors,
        ਪੂਰਨ                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    ਲਈ (ਪੂਰਨ i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ਵਾਪਸ tensors[idx];
}

ਖਾਲੀ ggml_build_forward_expand(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

ਖਾਲੀ ggml_build_forward_ਕ੍ਰਮ(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

ਖਾਲੀ ggml_build_backward_expand(
        ਬਣਤਰ ggml_context *  ctx,
        ਬਣਤਰ ggml_cgraph  *  cgraph,
        ਬਣਤਰ ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ਅਚਲ ਪੂਰਨ n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    bool * grads_needed = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ, ਆਕਾਰ(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        ਲਈ (ਪੂਰਨ i = 0; i < n_nodes_f; ++i) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];
            any_params = any_params || (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    ਲਈ (ਪੂਰਨ i = 0; i < n_nodes_f; ++i) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਜੇ (ਗੰਢ->type == GGML_TYPE_I32) {
            ਜਾਰੀ;
        }

        bool ਗੰਢ_needs_grad = (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM) || (ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ਚੋਣ (ਗੰਢ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:      // only used for its shape
            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ: // same as IM2COL
                ignore_src[0] = true;
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_UNARY: {
                ਅਚਲ ਗਿਣਤੀ ggml_unary_op uop = ggml_get_unary_op(ਗੰਢ);
                // SGN and STEP unary ops are piecewise constant
                ਜੇ (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_ਕਦਮ) {
                    ignore_src[0] = true;
                }
            } ਰੋਕੋ;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            ਹਾਲਤ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:      // row indices not differentiable
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ: // same as for GET_ROWS
            ਹਾਲਤ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ਰੋਕੋ;

            ਮੂਲ:
                ਰੋਕੋ;
        }
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; ++j) {
            ਜੇ (!ਗੰਢ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ->src[j])]) {
                ਜਾਰੀ;
            }
            GGML_ASSERT(ਗੰਢ->src[j]->type == GGML_TYPE_F32 || ਗੰਢ->src[j]->type == GGML_TYPE_F16);
            ਗੰਢ_needs_grad = true;
            ਰੋਕੋ;
        }
        ਜੇ (!ਗੰਢ_needs_grad) {
            ਜਾਰੀ;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ਗੰਢ->view_src || ਗੰਢ->op == GGML_OP_CPY || ਗੰਢ->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ || ਗੰਢ->op == GGML_OP_VIEW ||
            ਗੰਢ->op == GGML_OP_RESHAPE || ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || ਗੰਢ->op == GGML_OP_TRANSPOSE);

        ਅਚਲ size_t ihash = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ);
        GGML_ASSERT(ihash != GGML_HASHSET_ਭਰਿਆ);
        GGML_ASSERT(ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, ihash));
        ਜੇ (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ਗੰਢ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    ਲਈ (ਪੂਰਨ i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    ਮੁਕਤ(grads_needed);
}

ਸਥਿਰ ਖਾਲੀ * incr_ptr_aligned(ਖਾਲੀ ** p, size_t ਮਾਪ, size_t align) {
    ਖਾਲੀ * ptr = *p;
    ptr = (ਖਾਲੀ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ਖਾਲੀ *) ((ਅੱਖਰ *) ptr + ਮਾਪ);
    ਵਾਪਸ ptr;
}

ਸਥਿਰ size_t ggml_ਗ੍ਰਾਫ_nbytes(size_t ਮਾਪ, bool grads) {
    size_t hash_ਮਾਪ = ggml_hash_ਮਾਪ(ਮਾਪ * 2);
    ਖਾਲੀ * p = (ਅੱਖਰ *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, ਆਕਾਰ(ਬਣਤਰ ggml_cgraph), 1);
    incr_ptr_aligned(&p, ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(int32_t), ਆਕਾਰ(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)); // hash keys
    ਜੇ (grads) {
        incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_ਮਾਪ(hash_ਮਾਪ) * ਆਕਾਰ(ggml_bitset_t), ਆਕਾਰ(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ਵਾਪਸ nbytes - 1024;
}

size_t ggml_ਗ੍ਰਾਫ_overhead_custom(size_t ਮਾਪ, bool grads) {
    ਵਾਪਸ GGML_OBJECT_ਮਾਪ + GGML_PAD(ggml_ਗ੍ਰਾਫ_nbytes(ਮਾਪ, grads), GGML_MEM_ALIGN);
}

size_t ggml_ਗ੍ਰਾਫ_overhead(ਖਾਲੀ) {
    ਵਾਪਸ ggml_ਗ੍ਰਾਫ_overhead_custom(GGML_DEFAULT_ਗ੍ਰਾਫ_ਮਾਪ, false);
}

ਬਣਤਰ ggml_cgraph * ggml_new_ਗ੍ਰਾਫ_custom(ਬਣਤਰ ggml_context * ctx, size_t ਮਾਪ, bool grads) {
    ਅਚਲ size_t obj_ਮਾਪ = ggml_ਗ੍ਰਾਫ_nbytes(ਮਾਪ, grads);
    ਬਣਤਰ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_ਗ੍ਰਾਫ, obj_ਮਾਪ);
    ਬਣਤਰ ggml_cgraph * cgraph = (ਬਣਤਰ ggml_cgraph *) ((ਅੱਖਰ *) ctx->mem_ਬਫਰ + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_ਮਾਪ = ggml_hash_ਮਾਪ(ਮਾਪ * 2);

    ਖਾਲੀ * p = cgraph + 1;

    ਬਣਤਰ ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, ਮਾਪ      * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਬਣਤਰ ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, ਮਾਪ      * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(int32_t), ਆਕਾਰ(int32_t));
    ਬਣਤਰ ggml_tensor ** hash_ਕੁੰਜੀਆਂ_ptr  =         incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਬਣਤਰ ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)) : ਨਿਰਾ;
    ਬਣਤਰ ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *), ਆਕਾਰ(ਬਣਤਰ ggml_tensor *)) : ਨਿਰਾ;

    ggml_bitset_t * hash_ਵਰਤਿਆ = incr_ptr_aligned(&p, ggml_bitset_ਮਾਪ(hash_ਮਾਪ) * ਆਕਾਰ(ggml_bitset_t), ਆਕਾਰ(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ਪ੍ਰਤਿਜ੍ਞਾ(obj_ਮਾਪ == (size_t)((ਅੱਖਰ *)p - (ਅੱਖਰ *)cgraph));

    *cgraph = (ਬਣਤਰ ggml_cgraph) {
        /*.size         =*/ ਮਾਪ,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_ਮਾਪ, hash_ਵਰਤਿਆ, hash_ਕੁੰਜੀਆਂ_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਖੱਬਾ_ਮੰਜ਼ਿਲ_ਸੱਜਾ,
        /*.uid          =*/ 0,
    };

    ggml_hash_ਸੈੱਟ_reset(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ);
    ਜੇ (grads) {
        memset(cgraph->grads,     0, hash_ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    }

    ਵਾਪਸ cgraph;
}

ਬਣਤਰ ggml_cgraph * ggml_new_ਗ੍ਰਾਫ(ਬਣਤਰ ggml_context * ctx) {
    ਵਾਪਸ ggml_new_ਗ੍ਰਾਫ_custom(ctx, GGML_DEFAULT_ਗ੍ਰਾਫ_ਮਾਪ, false);
}

ਬਣਤਰ ggml_cgraph ggml_ਗ੍ਰਾਫ_view(ਬਣਤਰ ggml_cgraph * cgraph0, ਪੂਰਨ i0, ਪੂਰਨ i1) {
    ਬਣਤਰ ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ਨਿਰਾ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ਨਿਰਾ,
        /*.leafs            =*/ ਨਿਰਾ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->ਦੇਖਿਆ_hash_ਸੈੱਟ,
        /*.order            =*/ cgraph0->ਕ੍ਰਮ,
        /*.uid              =*/ 0
    };

    ਵਾਪਸ cgraph;
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_cpy(ਬਣਤਰ ggml_cgraph * src, ਬਣਤਰ ggml_cgraph * dst) {
    GGML_ASSERT(dst->ਮਾਪ >= src->n_leafs);
    GGML_ASSERT(dst->ਮਾਪ >= src->n_nodes);
    GGML_ASSERT(dst->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ >= src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->ਕ੍ਰਮ   = src->ਕ੍ਰਮ;

    ਲਈ (ਪੂਰਨ i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    ਲਈ (ਪੂਰਨ i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    ਲਈ (size_t i = 0; i < src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ; ++i) {
        // copy all hashset keys (tensors) that are in use
        ਜੇ (ggml_bitset_get(src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, i)) {
            size_t new_hash_pos = ggml_hash_ਪਾਓ(&dst->ਦੇਖਿਆ_hash_ਸੈੱਟ, src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਕੁੰਜੀਆਂ[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    ਜੇ (dst->grads) {
        memset(dst->grads,     0, dst->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        memset(dst->grad_accs, 0, dst->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਮਾਪ*ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    }
    ਜੇ (src->grads) {
        GGML_ASSERT(dst->grads     != ਨਿਰਾ);
        GGML_ASSERT(dst->grad_accs != ਨਿਰਾ);
        ਲਈ (ਪੂਰਨ i = 0; i < src->n_nodes; ++i) {
            ਅਚਲ size_t igrad_src = ggml_hash_ਲੱਭੋ(&src->ਦੇਖਿਆ_hash_ਸੈੱਟ, src->nodes[i]);
            ਅਚਲ size_t igrad_dst = ggml_hash_ਲੱਭੋ(&dst->ਦੇਖਿਆ_hash_ਸੈੱਟ, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_ਭਰਿਆ);
            GGML_ASSERT(ggml_bitset_get(src->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_ਭਰਿਆ);
            GGML_ASSERT(ggml_bitset_get(dst->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ਬਣਤਰ ggml_cgraph * ggml_ਗ੍ਰਾਫ_dup(ਬਣਤਰ ggml_context * ctx, ਬਣਤਰ ggml_cgraph * cgraph, bool force_grads) {
    ਬਣਤਰ ggml_cgraph * ਨਤੀਜਾ = ggml_new_ਗ੍ਰਾਫ_custom(ctx, cgraph->ਮਾਪ, cgraph->grads || force_grads);
    ggml_ਗ੍ਰਾਫ_cpy(cgraph, ਨਤੀਜਾ);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_zero(ਬਣਤਰ ggml_tensor * tensor) {
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(tensor)) {
        ਵਾਪਸ tensor;
    }
    ਜੇ (tensor->ਬਫਰ) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(tensor->ਅੰਕੜੇ);
        memset(tensor->ਅੰਕੜੇ, 0, ggml_nbytes(tensor));
    }
    ਵਾਪਸ tensor;
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_reset(ਬਣਤਰ ggml_cgraph * cgraph) {
    ਜੇ (!cgraph) {
        ਵਾਪਸ;
    }
    GGML_ASSERT(cgraph->grads != ਨਿਰਾ);

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ     = cgraph->nodes[i];
        ਬਣਤਰ ggml_tensor * grad_acc = ggml_ਗ੍ਰਾਫ_get_grad_acc(cgraph, ਗੰਢ);

        ਜੇ (ਗੰਢ->op == GGML_OP_OPT_ਕਦਮ_ADAMW) {
            // clear momenta
            ggml_ਸੈੱਟ_zero(ਗੰਢ->src[2]);
            ggml_ਸੈੱਟ_zero(ਗੰਢ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        ਜੇ (grad_acc) {
            ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ਅਚਲ ਦਸ਼ਮਲਵ onef = 1.0f;
                ਜੇ (grad_acc->ਬਫਰ) {
                    ggml_backend_tensor_ਸੈੱਟ(grad_acc, &onef, 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                } ਨਹੀਂਤਾਂ {
                    GGML_ASSERT(grad_acc->ਅੰਕੜੇ);
                    *((ਦਸ਼ਮਲਵ *) grad_acc->ਅੰਕੜੇ) = onef;
                }
            } ਨਹੀਂਤਾਂ {
                ggml_ਸੈੱਟ_zero(grad_acc);
            }
        }
    }
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_clear(ਬਣਤਰ ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_ਸੈੱਟ_reset(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ);
}

ਪੂਰਨ ggml_ਗ੍ਰਾਫ_ਮਾਪ(ਬਣਤਰ ggml_cgraph * cgraph) {
    ਵਾਪਸ cgraph->ਮਾਪ;
}

ਬਣਤਰ ggml_tensor * ggml_ਗ੍ਰਾਫ_ਗੰਢ(ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ i) {
    ਜੇ (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ਵਾਪਸ cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ਵਾਪਸ cgraph->nodes[i];
}

ਬਣਤਰ ggml_tensor ** ggml_ਗ੍ਰਾਫ_nodes(ਬਣਤਰ ggml_cgraph * cgraph) {
    ਵਾਪਸ cgraph->nodes;
}

ਪੂਰਨ ggml_ਗ੍ਰਾਫ_n_nodes(ਬਣਤਰ ggml_cgraph * cgraph) {
    ਵਾਪਸ cgraph->n_nodes;
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_ਜੋੜੋ_ਗੰਢ(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->ਮਾਪ > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ਬਣਤਰ ggml_tensor * ggml_ਗ੍ਰਾਫ_get_tensor(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਅੱਖਰ * name) {
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = cgraph->leafs[i];

        ਜੇ (strcmp(ਪੱਤੀ->name, name) == 0) {
            ਵਾਪਸ ਪੱਤੀ;
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਜੇ (strcmp(ਗੰਢ->name, name) == 0) {
            ਵਾਪਸ ਗੰਢ;
        }
    }

    ਵਾਪਸ ਨਿਰਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਗ੍ਰਾਫ_get_grad(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    ਅਚਲ size_t igrad = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ);
    ਵਾਪਸ igrad != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad) && cgraph->grads ? cgraph->grads[igrad] : ਨਿਰਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਗ੍ਰਾਫ_get_grad_acc(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    ਅਚਲ size_t igrad = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ);
    ਵਾਪਸ igrad != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ਨਿਰਾ;
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_print(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ਗੰਢ->ne[0], ਗੰਢ->ne[1], ਗੰਢ->ne[2],
                ggml_op_name(ਗੰਢ->op), (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_ਗ੍ਰਾਫ_get_grad(cgraph, ਗੰਢ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ਗੰਢ->ne[0], ਗੰਢ->ne[1],
                ggml_op_name(ਗੰਢ->op),
                ggml_get_name(ਗੰਢ));
    }

    GGML_LOG_INFO("========================================\n");
}

ਸਥਿਰ ਪੂਰਨ ggml_ਗੰਢ_list_ਲੱਭੋ_tensor(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                      ਅਚਲ ਪੂਰਨ *                idxs,
                                      ਪੂਰਨ                        ਗਿਣਤੀ_2,
                                      ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    ਲਈ (ਪੂਰਨ i = 0; i < ਗਿਣਤੀ_2; ++i) {
        ਅਚਲ ਪੂਰਨ ਗੰਢ_idx = idxs[i];

        ਜੇ (ਗੰਢ_idx >= cgraph->n_nodes) {
            ਵਾਪਸ -1;
        }
        ਜੇ (cgraph->nodes[ਗੰਢ_idx] == tensor) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ -1;
}

ਸਥਿਰ bool ggml_is_constant(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ tensor->ਬਫਰ != ਨਿਰਾ && ggml_backend_ਬਫਰ_get_usage(tensor->ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                ਅਚਲ ਪੂਰਨ *                ਗੰਢ_idxs,
                                ਪੂਰਨ                        ਗਿਣਤੀ_2,
                                ਅਚਲ ਗਿਣਤੀ ggml_op *       ops,
                                ਅਚਲ ਪੂਰਨ *                outputs,
                                ਪੂਰਨ                        ਅੰਕ_outputs) {
    GGML_ASSERT(outputs && ਅੰਕ_outputs > 0);

    ਲਈ (ਪੂਰਨ i = 0; i < ਗਿਣਤੀ_2; ++i) {
        ਜੇ (ਗੰਢ_idxs[i] >= cgraph->n_nodes) {
            ਵਾਪਸ false;
        }

        ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idxs[i]];

        ਜੇ (ਗੰਢ->op != ops[i]) {
            ਵਾਪਸ false;
        }

        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਵਾਪਸ false;
        }

        ਜੇ (ggml_ਗੰਢ_list_ਲੱਭੋ_tensor(cgraph, outputs, ਅੰਕ_outputs, ਗੰਢ) != -1) {
            ਜਾਰੀ;
        }

        ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ਵਾਪਸ false;
        }

        ਪੂਰਨ subgraph_ਵਰਤਦਾ = 0;
        ਲਈ (ਪੂਰਨ j = i + 1; j < ਗਿਣਤੀ_2; ++j) {
            ਅਚਲ ਬਣਤਰ ggml_tensor * other_ਗੰਢ = cgraph->nodes[ਗੰਢ_idxs[j]];
            ਲਈ (ਪੂਰਨ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ਜੇ (other_ਗੰਢ->src[src_idx] == ਗੰਢ) {
                    subgraph_ਵਰਤਦਾ++;
                }
            }
        }

        ਜੇ (subgraph_ਵਰਤਦਾ != ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, ਗੰਢ_idxs[i])) {
            ਵਾਪਸ false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ਬਣਤਰ ggml_tensor * view_src = ਗੰਢ->view_src;
        ਜਦੋਂ (view_src) {
            ਜੇ (ggml_ਗੰਢ_list_ਲੱਭੋ_tensor(cgraph, ਗੰਢ_idxs, ਗਿਣਤੀ_2, view_src) == -1 && !ggml_is_constant(view_src)) {
                ਵਾਪਸ false;
            }
            view_src = view_src->view_src;
        }
    }

    ਵਾਪਸ true;
}

// check if node is part of the graph
ਸਥਿਰ bool ggml_ਗ੍ਰਾਫ_ਲੱਭੋ(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    ਜੇ (cgraph == ਨਿਰਾ) {
        ਵਾਪਸ true;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਜੇ (cgraph->nodes[i] == ਗੰਢ) {
            ਵਾਪਸ true;
        }
    }

    ਵਾਪਸ false;
}

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_ਗ੍ਰਾਫ_get_ਮਾਪਾ(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਮਾਪਾ = cgraph->nodes[i];
        ਬਣਤਰ ggml_tensor * grad = ggml_ਗ੍ਰਾਫ_get_grad(cgraph, ਮਾਪਾ);

        ਜੇ (grad == ਗੰਢ) {
            ਵਾਪਸ ਮਾਪਾ;
        }
    }

    ਵਾਪਸ ਨਿਰਾ;
}

ਸਥਿਰ ਖਾਲੀ ggml_ਗ੍ਰਾਫ_dump_dot_ਗੰਢ_ਕਿਨਾਰਾ(FILE * fp, ਅਚਲ ਬਣਤਰ ggml_cgraph * gb, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਬਣਤਰ ggml_tensor * ਮਾਪਾ, ਅਚਲ ਅੱਖਰ * label)  {
    ਬਣਤਰ ggml_tensor * gparent = ggml_ਗ੍ਰਾਫ_get_ਮਾਪਾ(gb, ਗੰਢ);
    ਬਣਤਰ ggml_tensor * gparent0 = ggml_ਗ੍ਰਾਫ_get_ਮਾਪਾ(gb, ਮਾਪਾ);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (ਖਾਲੀ *) gparent0 : (ਖਾਲੀ *) ਮਾਪਾ,
            gparent ? (ਖਾਲੀ *) gparent : (ਖਾਲੀ *) ਗੰਢ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ਸਥਿਰ ਖਾਲੀ ggml_ਗ੍ਰਾਫ_dump_dot_ਪੱਤੀ_ਕਿਨਾਰਾ(FILE * fp, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਬਣਤਰ ggml_tensor * ਮਾਪਾ, ਅਚਲ ਅੱਖਰ * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (ਖਾਲੀ *) ਮਾਪਾ,
            (ਖਾਲੀ *) ਗੰਢ,
            label);
}

ਖਾਲੀ ggml_ਗ੍ਰਾਫ_dump_dot(ਅਚਲ ਬਣਤਰ ggml_cgraph * gb, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਅੱਖਰ * filename) {
    ਅੱਖਰ color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    ਲਈ (ਪੂਰਨ i = 0; i < gb->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = gb->nodes[i];
        ਬਣਤਰ ggml_tensor * grad = ggml_ਗ੍ਰਾਫ_get_grad(gb, ਗੰਢ);

        ਜੇ (ggml_ਗ੍ਰਾਫ_get_ਮਾਪਾ(gb, ਗੰਢ) != ਨਿਰਾ) {
            ਜਾਰੀ;
        }

        ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, ਆਕਾਰ(color), "yellow");
        } ਨਹੀਂਤਾਂ ਜੇ (grad) {
            ਜੇ (ggml_ਗ੍ਰਾਫ_ਲੱਭੋ(cgraph, ਗੰਢ)) {
                snprintf(color, ਆਕਾਰ(color), "green");
            } ਨਹੀਂਤਾਂ {
                snprintf(color, ਆਕਾਰ(color), "lightblue");
            }
        } ਨਹੀਂਤਾਂ {
            snprintf(color, ਆਕਾਰ(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (ਖਾਲੀ *) ਗੰਢ, color);

        ਜੇ (ਲੜੀ_ਲੰਬਾਈ(ਗੰਢ->name) > 0) {
            fprintf(fp, "%s (%s)|", ਗੰਢ->name, ggml_type_name(ਗੰਢ->type));
        } ਨਹੀਂਤਾਂ {
            fprintf(fp, "(%s)|", ggml_type_name(ਗੰਢ->type));
        }

        ਜੇ (ggml_is_ਮੈਟ੍ਰਿਕਸ(ਗੰਢ)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ਗੰਢ->ne[0], ਗੰਢ->ne[1], ggml_op_ਚਿੰਨ੍ਹ(ਗੰਢ->op));
        } ਨਹੀਂਤਾਂ {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ਗੰਢ->ne[0], ਗੰਢ->ne[1], ਗੰਢ->ne[2], ggml_op_ਚਿੰਨ੍ਹ(ਗੰਢ->op));
        }

        ਜੇ (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_ਚਿੰਨ੍ਹ(grad->op));
        } ਨਹੀਂਤਾਂ {
            fprintf(fp, "\"; ]\n");
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < gb->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = gb->leafs[i];

        snprintf(color, ਆਕਾਰ(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (ਖਾਲੀ *) ਗੰਢ, color);

        ਜੇ (ਲੜੀ_ਲੰਬਾਈ(ਗੰਢ->name) > 0) {
            fprintf(fp, "%s (%s)|", ਗੰਢ->name, ggml_type_name(ਗੰਢ->type));
        } ਨਹੀਂਤਾਂ {
            fprintf(fp, "(%s)|", ggml_type_name(ਗੰਢ->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ਗੰਢ->ne[0], ਗੰਢ->ne[1]);
        ਜੇ (ggml_nelements(ਗੰਢ) < 5 && ਗੰਢ->ਅੰਕੜੇ != ਨਿਰਾ) {
            fprintf(fp, " | (");
            ਲਈ (ਪੂਰਨ j = 0; j < ggml_nelements(ਗੰਢ); j++) {
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
                ਜੇ (j < ggml_nelements(ਗੰਢ) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    ਲਈ (ਪੂਰਨ i = 0; i < gb->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = gb->nodes[i];

        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਜੇ (ਗੰਢ->src[j]) {
                ਅੱਖਰ label[16];
                snprintf(label, ਆਕਾਰ(label), "src %d", j);
                ggml_ਗ੍ਰਾਫ_dump_dot_ਗੰਢ_ਕਿਨਾਰਾ(fp, gb, ਗੰਢ, ਗੰਢ->src[j], label);
            }
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < gb->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = gb->leafs[i];

        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਜੇ (ਗੰਢ->src[j]) {
                ਅੱਖਰ label[16];
                snprintf(label, ਆਕਾਰ(label), "src %d", j);
                ggml_ਗ੍ਰਾਫ_dump_dot_ਪੱਤੀ_ਕਿਨਾਰਾ(fp, ਗੰਢ, ਗੰਢ->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

ਖਾਲੀ ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_ਨਿਵੇਸ਼;
}

ਖਾਲੀ ggml_ਸੈੱਟ_output(ਬਣਤਰ ggml_tensor * tensor) {
    ਲਈ (ਬਣਤਰ ggml_tensor * cur = tensor; cur != ਨਿਰਾ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

ਖਾਲੀ ggml_ਸੈੱਟ_param(ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

ਖਾਲੀ ggml_ਸੈੱਟ_loss(ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

ਖਾਲੀ ggml_quantize_init(ਗਿਣਤੀ ggml_type type) {
    ggml_critical_section_ਸ਼ੁਰੂ();

    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ਰੋਕੋ;
        ਮੂਲ: // nothing
            ਰੋਕੋ;
    }

    ggml_critical_section_ਸਿਰਾ();
}

ਖਾਲੀ ggml_quantize_free(ਖਾਲੀ) {
    ggml_critical_section_ਸ਼ੁਰੂ();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_ਸਿਰਾ();
}

bool ggml_quantize_requires_imatrix(ਗਿਣਤੀ ggml_type type) {
    ਵਾਪਸ
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        ਗਿਣਤੀ ggml_type   type,
           ਅਚਲ ਦਸ਼ਮਲਵ * src,
                  ਖਾਲੀ * dst,
               int64_t   ਸ਼ੁਰੂ,
               int64_t   nrows,
               int64_t   n_ਪ੍ਰਤੀ_ਪੰਕਤੀ,
           ਅਚਲ ਦਸ਼ਮਲਵ * imatrix) {
    ਅਚਲ int64_t n = nrows * n_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਜੇ (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ਨਿਰਾ);
    }

    GGML_ASSERT(ਸ਼ੁਰੂ % type_traits[type].blck_ਮਾਪ == 0);
    GGML_ASSERT(ਸ਼ੁਰੂ % n_ਪ੍ਰਤੀ_ਪੰਕਤੀ == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ਅਚਲ size_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਸ਼ੁਰੂ / n_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ  = ggml_ਪੰਕਤੀ_ਮਾਪ(type, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);

    size_t ਨਤੀਜਾ = 0;

    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:    ਨਤੀਜਾ = quantize_q1_0   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:    ਨਤੀਜਾ = quantize_q2_0   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:    ਨਤੀਜਾ = quantize_q4_0   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:    ਨਤੀਜਾ = quantize_q4_1   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:    ਨਤੀਜਾ = quantize_q5_0   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:    ਨਤੀਜਾ = quantize_q5_1   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:    ਨਤੀਜਾ = quantize_q8_0   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:   ਨਤੀਜਾ = quantize_mxfp4  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_NVFP4:   ਨਤੀਜਾ = quantize_nvfp4  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:    ਨਤੀਜਾ = quantize_q2_K   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:    ਨਤੀਜਾ = quantize_q3_K   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:    ਨਤੀਜਾ = quantize_q4_K   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:    ਨਤੀਜਾ = quantize_q5_K   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:    ਨਤੀਜਾ = quantize_q6_K   (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_TQ1_0:   ਨਤੀਜਾ = quantize_tq1_0  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_TQ2_0:   ਨਤੀਜਾ = quantize_tq2_0  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS: ਨਤੀਜਾ = quantize_iq2_xxs(src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:  ਨਤੀਜਾ = quantize_iq2_xs (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS: ਨਤੀਜਾ = quantize_iq3_xxs(src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:   ਨਤੀਜਾ = quantize_iq3_s  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:   ਨਤੀਜਾ = quantize_iq2_s  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:   ਨਤੀਜਾ = quantize_iq1_s  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:   ਨਤੀਜਾ = quantize_iq1_m  (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:  ਨਤੀਜਾ = quantize_iq4_nl (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:  ਨਤੀਜਾ = quantize_iq4_xs (src + ਸ਼ੁਰੂ, (ਅੱਖਰ *) dst + ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ, nrows, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, imatrix); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                size_t elemsize = ਆਕਾਰ(ggml_fp16_t);
                ggml_fp32_ਮੰਜ਼ਿਲ_fp16_ਪੰਕਤੀ(src + ਸ਼ੁਰੂ, (ggml_fp16_t *)dst + ਸ਼ੁਰੂ, n);
                ਨਤੀਜਾ = n * elemsize;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                size_t elemsize = ਆਕਾਰ(ggml_bf16_t);
                ggml_fp32_ਮੰਜ਼ਿਲ_bf16_ਪੰਕਤੀ_ref(src + ਸ਼ੁਰੂ, (ggml_bf16_t *)dst + ਸ਼ੁਰੂ, n);
                ਨਤੀਜਾ = n * elemsize;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                size_t elemsize = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
                ਨਤੀਜਾ = n * elemsize;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *)dst + ਸ਼ੁਰੂ * elemsize, src + ਸ਼ੁਰੂ, ਨਤੀਜਾ);
            } ਰੋਕੋ;
        ਮੂਲ:
            ਪ੍ਰਤਿਜ੍ਞਾ(false);
    }

    GGML_ASSERT(ਨਤੀਜਾ == nrows * ਪੰਕਤੀ_ਮਾਪ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

////////////////////////////////////////////////////////////////////////////////

ਖਾਲੀ ggml_log_get(ggml_log_callback * log_callback, ਖਾਲੀ ** user_ਅੰਕੜੇ) {
    *log_callback = g_logger_state.log_callback;
    *user_ਅੰਕੜੇ    = g_logger_state.log_callback_user_ਅੰਕੜੇ;
}

ਖਾਲੀ ggml_log_ਸੈੱਟ(ggml_log_callback log_callback, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_ਅੰਕੜੇ = user_ਅੰਕੜੇ;
}

ਖਾਲੀ ggml_threadpool_params_init(ਬਣਤਰ ggml_threadpool_params * p, ਪੂਰਨ n_ਧਾਗੇ) {
    p->n_ਧਾਗੇ  = n_ਧਾਗੇ;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_ਧਾਗੇ); // all-zero means use the default affinity (usually inherited)
}

ਬਣਤਰ ggml_threadpool_params ggml_threadpool_params_default(ਪੂਰਨ n_ਧਾਗੇ) {
    ਬਣਤਰ ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_ਧਾਗੇ);
    ਵਾਪਸ p;
}

bool ggml_threadpool_params_ਮੇਲ(ਅਚਲ ਬਣਤਰ ggml_threadpool_params * p0, ਅਚਲ ਬਣਤਰ ggml_threadpool_params * p1) {
    ਜੇ (p0->n_ਧਾਗੇ  != p1->n_ਧਾਗੇ  ) ਵਾਪਸ false;
    ਜੇ (p0->prio       != p1->prio       ) ਵਾਪਸ false;
    ਜੇ (p0->poll       != p1->poll       ) ਵਾਪਸ false;
    ਜੇ (p0->strict_cpu != p1->strict_cpu ) ਵਾਪਸ false;
    ਵਾਪਸ memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_ਧਾਗੇ) == 0;
}
