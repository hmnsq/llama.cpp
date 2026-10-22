#ವ್ಯಾಖ್ಯಾನಿಸು _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#ವ್ಯಾಖ್ಯಾನಿಸು USE_MATH_ನಿರೂಪಿಸುತ್ತದೆ // For M_PI on MSVC

#ಅನ್ತರ್ಭಾವ "ggml-version.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

// FIXME: required here for quantization functions
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_HBM
#ಅನ್ತರ್ಭಾವ <hbwmalloc.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
#ಅನ್ತರ್ಭಾವ <malloc.h> // using malloc.h with MSC/MINGW
#ಅಥವಾಆದರೆ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ಅನ್ತರ್ಭಾವ <alloca.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <errno.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <inttypes.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <signal.h>
#ಆದರೆ defined(__gnu_linux__)
#ಅನ್ತರ್ಭಾವ <syscall.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <mach/mach.h>
#ಅನ್ತರ್ಭಾವ <TargetConditionals.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_WIN32)
#ವ್ಯಾಖ್ಯಾನಿಸು WIN32_LEAN_AND_ಸರಾಸರಿ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NOMINMAX
    #ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <windows.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ GGML_ಬಳಸದ

uint64_t ggml_ನಕ್ಷೆ_ಮುಂದಿನ_uid(ಶೂನ್ಯ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
#ಆದರೆ defined(_WIN32)
    ಸ್ಥಿರ ಅಸ್ಥಿರ LONG counter = 1;
    ಹಿಂತಿರುಗಿ (uint64_t) InterlockedIncrement(&counter) - 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರ ಅಸ್ಥಿರ ದೀರ್ಘ ದೀರ್ಘ counter = 1;
    ಹಿಂತಿರುಗಿ (uint64_t) _InterlockedIncrement64(&counter) - 1;
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರ uint64_t counter = 1;
    ಹಿಂತಿರುಗಿ __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#ಆದರೆ_ಅಂತ್ಯ
}

// Needed for ggml_fp32_to_bf16_row()
#ಆದರೆ defined(__AVX512BF16__)
#ಆದರೆ defined(_MSC_VER)
#ವ್ಯಾಖ್ಯಾನಿಸು m512i(p) p
#ಇಲ್ಲದಿದ್ದರೆ
#ಅನ್ತರ್ಭಾವ <immintrin.h>
#ವ್ಯಾಖ್ಯಾನಿಸು m512i(p) (__m512i)(p)
#ಆದರೆ_ಅಂತ್ಯ // defined(_MSC_VER)
#ಆದರೆ_ಅಂತ್ಯ // defined(__AVX512BF16__)

#ಆದರೆ defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !ಗುರಿ_OS_TV && !ಗುರಿ_OS_WATCH)

#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/wait.h>
#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/prctl.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__ANDROID__)
#ಅನ್ತರ್ಭಾವ <unwind.h>
#ಅನ್ತರ್ಭಾವ <dlfcn.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>

ರಚನೆ backtrace_state {
    ಶೂನ್ಯ ** current;
    ಶೂನ್ಯ ** ಕೊನೆ;
};

ಸ್ಥಿರ Unwind_Reason_ಸಂಕೇತ unwind_callback(ರಚನೆ _Unwind_Context* context, ಶೂನ್ಯ* arg) {
    ರಚನೆ backtrace_state * state = (ರಚನೆ backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    ಆದರೆ (pc) {
        ಆದರೆ (state->current == state->ಕೊನೆ) {
            ಹಿಂತಿರುಗಿ URC_ಕೊನೆ_ದ_ರಾಶಿ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            *state->current++ = (ಶೂನ್ಯ*)pc;
        }
    }
    ಹಿಂತಿರುಗಿ _URC_NO_REASON;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max = 100;
    ಶೂನ್ಯ* ಬಫರ್[max];

    ರಚನೆ backtrace_state state = {ಬಫರ್, ಬಫರ್ + max};
    _Unwind_Backtrace(unwind_callback, &state);

    ಪೂರ್ಣಾಂಕ ಎಣಿಕೆ_2 = state.current - ಬಫರ್;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = 0; idx < ಎಣಿಕೆ_2; ++idx) {
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr = ಬಫರ್[idx];
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಚಿಹ್ನೆ = "";

        Dl_info info;
        ಆದರೆ (dladdr(addr, &info) && info.dli_sname) {
            ಚಿಹ್ನೆ = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, ಚಿಹ್ನೆ);
    }
}
#ಅಥವಾಆದರೆ defined(__linux__) && defined(__GLIBC__)
#ಅನ್ತರ್ಭಾವ <execinfo.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಶೂನ್ಯ * trace[100];
    ಪೂರ್ಣಾಂಕ nptrs = backtrace(trace, ಗಾತ್ರ(trace)/ಗಾತ್ರ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ಅಥವಾಆದರೆ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <execinfo.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಶೂನ್ಯ * trace[100];
    ಪೂರ್ಣಾಂಕ nptrs = backtrace(trace, ಗಾತ್ರ(trace)/ಗಾತ್ರ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    // platform not supported
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ಆದರೆ (GGML_NO_BACKTRACE) {
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    ಆದರೆ (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t ಗಾತ್ರ_2 = 0;
    ಅಕ್ಷರ * ಸಾಲು = ಖಾಲಿ;
    ssize_t ಉದ್ದಳತೆ = 0;
    ಆಗ ((ಉದ್ದಳತೆ = getline(&ಸಾಲು, &ಗಾತ್ರ_2, f)) > 0) {
        ಆದರೆ (!ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(ಸಾಲು, "TracerPid:", ಗಾತ್ರ("TracerPid:") - 1) &&
            (ಉದ್ದಳತೆ != ಗಾತ್ರ("TracerPid:\t0\n") - 1 || ಸಾಲು[ಉದ್ದಳತೆ - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            ಬಿಡುಗಡೆ(ಸಾಲು);
            fclose(f);
            ಹಿಂತಿರುಗಿ;
        }
    }
    ಬಿಡುಗಡೆ(ಸಾಲು);
    fclose(f);
    ಪೂರ್ಣಾಂಕ ಬೀಗ[2] = { -1, -1 };
    (ಶೂನ್ಯ) !pipe(ಬೀಗ); // Don't start gdb until after PR_SET_PTRACER
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಪೋಷಕ_pid = getpid();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಮಗು_pid = fork();
    ಆದರೆ (ಮಗು_pid < 0) { // error
#ಆದರೆ defined(__linux__)
        close(ಬೀಗ[1]);
        close(ಬೀಗ[0]);
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಮಗು_pid == 0) { //ಮಗು
        ಅಕ್ಷರ attach[32];
        snprintf(attach, ಗಾತ್ರ(attach), "attach %d", ಪೋಷಕ_pid);
#ಆದರೆ defined(__linux__)
        close(ಬೀಗ[1]);
        (ಶೂನ್ಯ) !ಪಠ(ಬೀಗ[0], ಬೀಗ, 1);
        close(ಬೀಗ[0]);
#ಆದರೆ_ಅಂತ್ಯ
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (ಅಕ್ಷರ *) ಖಾಲಿ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[ಗಾತ್ರ("attach ") - 1],
            (ಅಕ್ಷರ *) ಖಾಲಿ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } ಇಲ್ಲದಿದ್ದರೆ { //ಪೋಷಕ
#ಆದರೆ defined(__linux__)
        prctl(PR_ಗಣ_PTRACER, ಮಗು_pid);
        close(ಬೀಗ[1]);
        close(ಬೀಗ[0]);
#ಆದರೆ_ಅಂತ್ಯ
        waitpid(ಮಗು_pid, ಖಾಲಿ, 0);
    }
}
#ಇಲ್ಲದಿದ್ದರೆ
ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ) {
    // platform not supported
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ggml_abort_callback_t g_abort_callback = ಖಾಲಿ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_ಗಣ_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ಹಿಂತಿರುಗಿ ret_val;
}

ಶೂನ್ಯ ggml_abort(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * file, ಪೂರ್ಣಾಂಕ ಸಾಲು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fmt, ...) {
    fflush(stdout);

    ಅಕ್ಷರ ಸಂದೇಶ[2048];
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = snprintf(ಸಂದೇಶ, ಗಾತ್ರ(ಸಂದೇಶ), "%s:%d: ", file, ಸಾಲು);

    va_list ವಾದಗಳು_2;
    va_start(ವಾದಗಳು_2, fmt);
    vsnprintf(ಸಂದೇಶ + ಸ್ಥಾನಾಂತರ, ಗಾತ್ರ(ಸಂದೇಶ) - ಸ್ಥಾನಾಂತರ, fmt, ವಾದಗಳು_2);
    va_end(ವಾದಗಳು_2);

    ಆದರೆ (g_abort_callback) {
        g_abort_callback(ಸಂದೇಶ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", ಸಂದೇಶ);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

ರಚನೆ ggml_logger_state {
    ggml_log_callback log_callback;
    ಶೂನ್ಯ * log_callback_user_ದತ್ತಾಂಶ;
};
ಸ್ಥಿರ ರಚನೆ ggml_logger_state g_logger_state = {ggml_log_callback_default, ಖಾಲಿ};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_log_ಆಂತರಿಕ_v(ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * format, va_list ವಾದಗಳು_2) {
    ಆದರೆ (format == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }
    va_list ವಾದಗಳು_copy;
    va_copy(ವಾದಗಳು_copy, ವಾದಗಳು_2);
    ಅಕ್ಷರ ಬಫರ್[128];
    ಪೂರ್ಣಾಂಕ len = vsnprintf(ಬಫರ್, 128, format, ವಾದಗಳು_2);
    ಆದರೆ (len < 128) {
        g_logger_state.log_callback(level, ಬಫರ್, g_logger_state.log_callback_user_ದತ್ತಾಂಶ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಅಕ್ಷರ * buffer2 = (ಅಕ್ಷರ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(len + 1, ಗಾತ್ರ(ಅಕ್ಷರ));
        vsnprintf(buffer2, len + 1, format, ವಾದಗಳು_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_ದತ್ತಾಂಶ);
        ಬಿಡುಗಡೆ(buffer2);
    }
    va_end(ವಾದಗಳು_copy);
}

ಶೂನ್ಯ ggml_log_ಆಂತರಿಕ(ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * format, ...) {
    va_list ವಾದಗಳು_2;
    va_start(ವಾದಗಳು_2, format);
    ggml_log_ಆಂತರಿಕ_v(level, format, ವಾದಗಳು_2);
    va_end(ವಾದಗಳು_2);
}

ಶೂನ್ಯ ggml_log_callback_default(ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಪಠ್ಯ_2, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) level;
    (ಶೂನ್ಯ) user_ದತ್ತಾಂಶ;
    fputs(ಪಠ್ಯ_2, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#ಆದರೆ_ಅಂತ್ಯ


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#ಆದರೆ defined(__has_feature)
#ಆದರೆ __has_feature(thread_sanitizer)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_TSAN_ENABLED 1
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ defined(__SANITIZE_THREAD__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_TSAN_ENABLED 1
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ * ggml_aligned_malloc(size_t ಗಾತ್ರ_2) {
#ಆದರೆ defined(__s390x__)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ alignment = 256;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ alignment = 64;
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
    ಹಿಂತಿರುಗಿ _aligned_malloc(ಗಾತ್ರ_2, alignment);
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (ಗಾತ್ರ_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಶೂನ್ಯ * aligned_ಸ್ಮರಣೆ = ಖಾಲಿ;
  #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_HBM
    ಪೂರ್ಣಾಂಕ ಫಲಿತಾಂಶ = hbw_posix_memalign(&aligned_ಸ್ಮರಣೆ, alignment, ಗಾತ್ರ_2);
  #ಅಥವಾಆದರೆ ಗುರಿ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_ಬಳಸದ(alignment);
    kern_return_t alloc_ಸ್ಥಿತಿ = vm_allocate((vm_map_t) mach_task_self(), (vm_ವಿಳಾಸ_t *) &aligned_ಸ್ಮರಣೆ, ಗಾತ್ರ_2, VM_FLAGS_ANYWHERE);
    ಪೂರ್ಣಾಂಕ ಫಲಿತಾಂಶ = EFAULT;
    ಆಯ್ಕೆ (alloc_ಸ್ಥಿತಿ) {
        ಸ್ಥಿತಿ KERN_SUCCESS:
            ಫಲಿತಾಂಶ = 0;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ KERN_INVALID_ವಿಳಾಸ:
            ಫಲಿತಾಂಶ = EINVAL;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ KERN_NO_SPACE:
            ಫಲಿತಾಂಶ = ENOMEM;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಫಲಿತಾಂಶ = EFAULT;
            ನಿಲ್ಲಿಸು;
    }
  #ಇಲ್ಲದಿದ್ದರೆ
    ಪೂರ್ಣಾಂಕ ಫಲಿತಾಂಶ = posix_memalign(&aligned_ಸ್ಮರಣೆ, alignment, ಗಾತ್ರ_2);
  #ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (ಫಲಿತಾಂಶ != 0) {
        // Handle allocation failure
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *error_desc = "unknown allocation error";
        ಆಯ್ಕೆ (ಫಲಿತಾಂಶ) {
            ಸ್ಥಿತಿ EINVAL:
                error_desc = "invalid alignment value";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ ENOMEM:
                error_desc = "insufficient memory";
                ನಿಲ್ಲಿಸು;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, ಗಾತ್ರ_2/(1024.0*1024.0));
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ aligned_ಸ್ಮರಣೆ;
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_aligned_free(ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2) {
    GGML_ಬಳಸದ(ಗಾತ್ರ_2);
#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#ಅಥವಾಆದರೆ GGML_USE_CPU_HBM
    ಆದರೆ (ptr != ಖಾಲಿ) {
        hbw_free(ptr);
    }
#ಅಥವಾಆದರೆ ಗುರಿ_OS_OSX && !defined(GGML_TSAN_ENABLED)
    ಆದರೆ (ptr != ಖಾಲಿ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_ವಿಳಾಸ_t)ptr, ಗಾತ್ರ_2);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಿಡುಗಡೆ(ptr);
#ಆದರೆ_ಅಂತ್ಯ
}


ಸಾಲೊಳಗೆ ಸ್ಥಿರ ಶೂನ್ಯ * ggml_malloc(size_t ಗಾತ್ರ_2) {
    ಆದರೆ (ಗಾತ್ರ_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಶೂನ್ಯ * ಫಲಿತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ_2);
    ಆದರೆ (ಫಲಿತಾಂಶ == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ಗಾತ್ರ_2/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// calloc
ಸಾಲೊಳಗೆ ಸ್ಥಿರ ಶೂನ್ಯ * ggml_calloc(size_t ಅಂಕೆ, size_t ಗಾತ್ರ_2) {
    ಆದರೆ (ಅಂಕೆ == 0 || ಗಾತ್ರ_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಶೂನ್ಯ * ಫಲಿತಾಂಶ = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ಅಂಕೆ, ಗಾತ್ರ_2);
    ಆದರೆ (ಫಲಿತಾಂಶ == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, ಗಾತ್ರ_2/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_MALLOC(ಗಾತ್ರ_2)      ggml_malloc(ಗಾತ್ರ_2)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CALLOC(ಅಂಕೆ, ಗಾತ್ರ_2) ggml_calloc(ಅಂಕೆ, ಗಾತ್ರ_2)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FREE(ptr) ಬಿಡುಗಡೆ(ptr)

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_ಸ್ಥಿತಿ_ಗಮ್ಯ_string(ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2) {
    ಆಯ್ಕೆ (ಸ್ಥಿತಿ_2) {
        ಸ್ಥಿತಿ GGML_ಸ್ಥಿತಿ_ALLOC_FAILED: ಹಿಂತಿರುಗಿ "GGML status: error (failed to allocate memory)";
        ಸ್ಥಿತಿ GGML_ಸ್ಥಿತಿ_FAILED:       ಹಿಂತಿರುಗಿ "GGML status: error (operation failed)";
        ಸ್ಥಿತಿ GGML_ಸ್ಥಿತಿ_SUCCESS:      ಹಿಂತಿರುಗಿ "GGML status: success";
        ಸ್ಥಿತಿ GGML_ಸ್ಥಿತಿ_ABORTED:      ಹಿಂತಿರುಗಿ "GGML status: warning (operation aborted)";
    }

    ಹಿಂತಿರುಗಿ "GGML status: unknown";
}

ದಶಮಾಂಶ ggml_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t x) {
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_fp16_ಗಮ್ಯ_fp32 do_not_use_ggml_fp16_ಗಮ್ಯ_fp32_ಒಳ_ggml
    ಹಿಂತಿರುಗಿ GGML_FP16_ಗಮ್ಯ_FP32(x);
}

ggml_fp16_t ggml_fp32_ಗಮ್ಯ_fp16(ದಶಮಾಂಶ x) {
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_fp32_ಗಮ್ಯ_fp16 do_not_use_ggml_fp32_ಗಮ್ಯ_fp16_ಒಳ_ggml
    ಹಿಂತಿರುಗಿ GGML_FP32_ಗಮ್ಯ_FP16(x);
}

ದಶಮಾಂಶ ggml_bf16_ಗಮ್ಯ_fp32(ggml_bf16_t x) {
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_bf16_ಗಮ್ಯ_fp32 do_not_use_ggml_bf16_ಗಮ್ಯ_fp32_ಒಳ_ggml
    ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_ಗಮ್ಯ_bf16(ದಶಮಾಂಶ x) {
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_fp32_ಗಮ್ಯ_bf16 do_not_use_ggml_fp32_ಗಮ್ಯ_bf16_ಒಳ_ggml
    ಹಿಂತಿರುಗಿ GGML_FP32_ಗಮ್ಯ_BF16(x);
}

ಶೂನ್ಯ ggml_fp16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x, ದಶಮಾಂಶ * y, int64_t n) {
    ಗಾಗಿ (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_ಗಮ್ಯ_FP32(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_ಗಮ್ಯ_fp16_ಅಡ್ಡಸಾಲು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ggml_fp16_t * y, int64_t n) {
    ಪೂರ್ಣಾಂಕ i = 0;
    ಗಾಗಿ (; i < n; ++i) {
        y[i] = GGML_FP32_ಗಮ್ಯ_FP16(x[i]);
    }
}

ಶೂನ್ಯ ggml_bf16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು(ಸ್ಥಿರಾಂಕ ggml_bf16_t * x, ದಶಮಾಂಶ * y, int64_t n) {
    ಪೂರ್ಣಾಂಕ i = 0;
    ಗಾಗಿ (; i < n; ++i) {
        y[i] = GGML_BF16_ಗಮ್ಯ_FP32(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_ಗಮ್ಯ_bf16_ಅಡ್ಡಸಾಲು_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ggml_bf16_t * y, int64_t n) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_ಗಮ್ಯ_bf16(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_ಗಮ್ಯ_bf16_ಅಡ್ಡಸಾಲು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ggml_bf16_t * y, int64_t n) {
  ಪೂರ್ಣಾಂಕ i = 0;
#ಆದರೆ defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  ಗಾಗಿ (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i < n; i++) {
        y[i] = GGML_FP32_ಗಮ್ಯ_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ಹಿಂತಿರುಗಿ memcmp(guid_a, guid_b, ಗಾತ್ರ(ggml_guid)) == 0;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_ಆವೃತ್ತಿ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ GGML_ಆವೃತ್ತಿ;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_commit(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ GGML_COMMIT;
}

//
// timing
//

#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
ಸ್ಥಿರ int64_t timer_freq, timer_ಆರಂಭ;
ಸ್ಥಿರ BOOL CALLBACK ggml_ಸಮಯ_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    ಬಳಸದ(once);
    ಬಳಸದ(param);
    ಬಳಸದ(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_ಆರಂಭ = t.QuadPart;

    ಹಿಂತಿರುಗಿ TRUE;
}
ಶೂನ್ಯ ggml_ಸಮಯ_init(ಶೂನ್ಯ) {
    ಸ್ಥಿರ INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_ಸಮಯ_init_once, ಖಾಲಿ, ಖಾಲಿ);
}
int64_t ggml_ಸಮಯ_ms(ಶೂನ್ಯ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ಹಿಂತಿರುಗಿ ((t.QuadPart-timer_ಆರಂಭ) * 1000) / timer_freq;
}
int64_t ggml_ಸಮಯ_us(ಶೂನ್ಯ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ಹಿಂತಿರುಗಿ ((t.QuadPart-timer_ಆರಂಭ) * 1000000) / timer_freq;
}
#ಇಲ್ಲದಿದ್ದರೆ
ಶೂನ್ಯ ggml_ಸಮಯ_init(ಶೂನ್ಯ) {}
int64_t ggml_ಸಮಯ_ms(ಶೂನ್ಯ) {
    ರಚನೆ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ಹಿಂತಿರುಗಿ (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_ಸಮಯ_us(ಶೂನ್ಯ) {
    ರಚನೆ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ಹಿಂತಿರುಗಿ (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#ಆದರೆ_ಅಂತ್ಯ

int64_t ggml_ಚಕ್ರಗಳು(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ clock();
}

int64_t ggml_ಚಕ್ರಗಳು_ಪ್ರತಿ_ms(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
ಸ್ಥಿರ wchar_t * ggml_mbstowcs(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * mbs) {
    ಪೂರ್ಣಾಂಕ wlen = Multi_Byte_ಗಮ್ಯ_Wide_Char(CP_UTF8, 0, mbs, -1, ಖಾಲಿ, 0);
    ಆದರೆ (!wlen) {
        errno = EINVAL;
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * ಗಾತ್ರ(wchar_t));
    wlen = Multi_Byte_ಗಮ್ಯ_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    ಆದರೆ (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ wbuf;
}
#ಆದರೆ_ಅಂತ್ಯ

FILE * ggml_fopen(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fname, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಬಹುಲಕ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    FILE * file = ಖಾಲಿ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    ಆದರೆ (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(ಬಹುಲಕ);
        ಆದರೆ (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    ಹಿಂತಿರುಗಿ file;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ fopen(fname, ಬಹುಲಕ);
#ಆದರೆ_ಅಂತ್ಯ

}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits type_traits[GGML_TYPE_ಎಣಿಕೆ] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ದ್ವಿಗುಣ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ದಶಮಾಂಶ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ggml_fp16_t),
        .is_quantized             = false,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) ggml_fp16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_ಗಮ್ಯ_fp16_ಅಡ್ಡಸಾಲು,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_ಗಾತ್ರ                = QK1_0,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q1_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_ಗಾತ್ರ                = QK2_0,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q2_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_ಗಾತ್ರ                = QK4_0,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q4_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_ಗಾತ್ರ                = QK4_1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q4_1),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_ಗಾತ್ರ                = QK5_0,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q5_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_ಗಾತ್ರ                = QK5_1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q5_1),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_ಗಾತ್ರ                = QK8_0,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q8_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_ಗಾತ್ರ                = QK8_1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_ಗಾತ್ರ                = QK_MXFP4,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_mxfp4),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_ಗಾತ್ರ                = QK_NVFP4,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_nvfp4),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q2_K),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q3_K),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q4_K),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q5_K),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q6_K),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq2_xxs),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq2_xxs,
        .from_float_ref           = ಖಾಲಿ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq2_xs),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq2_xs,
        .from_float_ref           = ಖಾಲಿ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq3_xxs),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq3_s),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq2_s),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq1_s),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq1_s,
        .from_float_ref           = ಖಾಲಿ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq1_m),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq1_m,
        .from_float_ref           = ಖಾಲಿ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_ಗಾತ್ರ                = QK4_NL,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq4_nl),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_iq4_xs),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_ಅಡ್ಡಸಾಲು_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_ಗಾತ್ರ                = 1,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ggml_bf16_t),
        .is_quantized             = false,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) ggml_bf16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_ಗಮ್ಯ_bf16_ಅಡ್ಡಸಾಲು_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_tq1_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_ಗಾತ್ರ                = QK_K,
        .type_ಗಾತ್ರ                = ಗಾತ್ರ(ಖಂಡ_tq2_0),
        .is_quantized             = true,
        .ಗಮ್ಯ_float                 = (ggml_ಗಮ್ಯ_float_t) dequantize_ಅಡ್ಡಸಾಲು_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_ಅಡ್ಡಸಾಲು_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_ಗಾತ್ರ                = 0,
        .type_ಗಾತ್ರ                = 0,
        .is_quantized             = false,
    },
};

ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * ggml_get_type_traits(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ &type_traits[type];
}

//
// ggml object
//

ರಚನೆ ggml_object {
    size_t offs;
    size_t ಗಾತ್ರ_2;

    ರಚನೆ ggml_object * ಮುಂದಿನ;

    ಎಣಿಕೆ ggml_object_type type;

    ಅಕ್ಷರ padding[4];
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t GGML_OBJECT_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ ggml_object);

//
// ggml context
//

ರಚನೆ ggml_context {
    size_t mem_ಗಾತ್ರ;
    ಶೂನ್ಯ * mem_ಬಫರ್;
    bool   mem_ಬಫರ್_owned;
    bool   no_alloc;

    ಪೂರ್ಣಾಂಕ    n_objects;

    ರಚನೆ ggml_object * objects_begin;
    ರಚನೆ ggml_object * objects_ಕೊನೆ;
};

//
// data types
//

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_OP_NAME[GGML_OP_ಎಣಿಕೆ] = {
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

static_assert(GGML_OP_ಎಣಿಕೆ == 102, "GGML_OP_COUNT != 102");

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_OP_ಚಿಹ್ನೆ[GGML_OP_ಎಣಿಕೆ] = {
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

static_assert(GGML_OP_ಎಣಿಕೆ == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_ಎಣಿಕೆ == 2, "GGML_OP_POOL_COUNT != 2");

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_UNARY_OP_NAME[GGML_UNARY_OP_ಎಣಿಕೆ] = {
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

static_assert(GGML_UNARY_OP_ಎಣಿಕೆ == 22, "GGML_UNARY_OP_COUNT != 22");

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_GLU_OP_NAME[GGML_GLU_OP_ಎಣಿಕೆ] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_ಎಣಿಕೆ == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(ಗಾತ್ರ(ರಚನೆ ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(ಗಾತ್ರ(ರಚನೆ ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_print_object(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->ಗಾತ್ರ_2, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) obj->ಮುಂದಿನ);
}

ಶೂನ್ಯ ggml_print_objects(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    ರಚನೆ ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ctx);

    ಆಗ (obj != ಖಾಲಿ) {
        ggml_print_object(obj);
        obj = obj->ಮುಂದಿನ;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಆದರೆ (tensor->ne[i] <= 0) {
            ಹಿಂತಿರುಗಿ 0;
        }
    }

    size_t nbytes;
    ಸ್ಥಿರಾಂಕ size_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(tensor->type);
    ಆದರೆ (blck_ಗಾತ್ರ == 1) {
        nbytes = ggml_type_ಗಾತ್ರ(tensor->type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_ಗಾತ್ರ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ಹಿಂತಿರುಗಿ nbytes;
}

size_t ggml_nbytes_pad(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_ಗಾತ್ರ(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ type_traits[type].blck_ಗಾತ್ರ;
}

size_t ggml_type_ಗಾತ್ರ(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ type_traits[type].type_ಗಾತ್ರ;
}

size_t ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ಎಣಿಕೆ ggml_type type, int64_t ne) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತಿಜ್ಞಾ(ne % ggml_blck_ಗಾತ್ರ(type) == 0);
    ಹಿಂತಿರುಗಿ ggml_type_ಗಾತ್ರ(type)*ne/ggml_blck_ಗಾತ್ರ(type);
}

ದ್ವಿಗುಣ ggml_type_sizef(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ ((ದ್ವಿಗುಣ)(type_traits[type].type_ಗಾತ್ರ))/type_traits[type].blck_ಗಾತ್ರ;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_type_name(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ type_traits[type].type_name;
}

bool ggml_is_quantized(ಎಣಿಕೆ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ type_traits[type].is_quantized;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_op_name(ಎಣಿಕೆ ggml_op op) {
    ಹಿಂತಿರುಗಿ GGML_OP_NAME[op];
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_op_ಚಿಹ್ನೆ(ಎಣಿಕೆ ggml_op op) {
    ಹಿಂತಿರುಗಿ GGML_OP_ಚಿಹ್ನೆ[op];
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_unary_op_name(ಎಣಿಕೆ ggml_unary_op op) {
    ಹಿಂತಿರುಗಿ GGML_UNARY_OP_NAME[op];
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_glu_op_name(ಎಣಿಕೆ ggml_glu_op op) {
    ಹಿಂತಿರುಗಿ GGML_GLU_OP_NAME[op];
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_op_desc(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (t->op == GGML_OP_UNARY) {
        ಎಣಿಕೆ ggml_unary_op uop = ggml_get_unary_op(t);
        ಹಿಂತಿರುಗಿ ggml_unary_op_name(uop);
    }
    ಆದರೆ (t->op == GGML_OP_GLU) {
        ಎಣಿಕೆ ggml_glu_op gop = ggml_get_glu_op(t);
        ಹಿಂತಿರುಗಿ ggml_glu_op_name(gop);
    }
    ಹಿಂತಿರುಗಿ ggml_op_name(t->op);
}

size_t ggml_element_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_type_ಗಾತ್ರ(tensor->type);
}

bool ggml_is_scalar(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_ಸದಿಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_ಮಾತೃಕೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor->ne[3] == 1;
}

ಪೂರ್ಣಾಂಕ ggml_n_dims(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        ಆದರೆ (tensor->ne[i] > 1) {
            ಹಿಂತಿರುಗಿ i + 1;
        }
    }
    ಹಿಂತಿರುಗಿ 1;
}

ಎಣಿಕೆ ggml_type ggml_ftype_ಗಮ್ಯ_ggml_type(ಎಣಿಕೆ ggml_ftype ftype) {
    ಎಣಿಕೆ ggml_type wtype = GGML_TYPE_ಎಣಿಕೆ;

    ಆಯ್ಕೆ (ftype) {
        ಸ್ಥಿತಿ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_ಎಣಿಕೆ; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_ಎಣಿಕೆ; ನಿಲ್ಲಿಸು;
    }

    GGML_ASSERT(wtype != GGML_TYPE_ಎಣಿಕೆ);

    ಹಿಂತಿರುಗಿ wtype;
}

size_t ggml_tensor_overhead(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ GGML_OBJECT_ಗಾತ್ರ + GGML_TENSOR_ಗಾತ್ರ;
}

bool ggml_is_transposed(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor->nb[0] > tensor->nb[1];
}

ಸ್ಥಿರ bool ggml_is_contiguous_m_n(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n) {
    size_t ಮುಂದಿನ_nb = ggml_type_ಗಾತ್ರ(tensor->type);
    ಆದರೆ (tensor->ne[0] != ggml_blck_ಗಾತ್ರ(tensor->type) && tensor->nb[0] != ಮುಂದಿನ_nb) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಮುಂದಿನ_nb *= tensor->ne[0]/ggml_blck_ಗಾತ್ರ(tensor->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < n; i++) {
        ಆದರೆ (i > m) {
            ಆದರೆ (tensor->ne[i] != 1 && tensor->nb[i] != ಮುಂದಿನ_nb) {
                ಹಿಂತಿರುಗಿ false;
            }
            ಮುಂದಿನ_nb *= tensor->ne[i];
        } ಇಲ್ಲದಿದ್ದರೆ {
            // this dimension does not need to be contiguous
            ಮುಂದಿನ_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ಹಿಂತಿರುಗಿ true;
}

bool ggml_is_contiguous(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_ಗಮ್ಯ_1(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_ಗಮ್ಯ_2(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_ಗಮ್ಯ_3(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_ಗಾತ್ರ(tensor->type)/ggml_blck_ಗಾತ್ರ(tensor->type);
}

bool ggml_is_permuted(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_ಗಾತ್ರ(tensor->type);
}

bool ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ
        tensor->ne[0] == ggml_blck_ಗಾತ್ರ(tensor->type) ||
        tensor->nb[0] == ggml_type_ಗಾತ್ರ(tensor->type);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_is_padded_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ
        tensor->nb[0] == ggml_type_ಗಾತ್ರ(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_ಖಾಲಿಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಆದರೆ (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ಹಿಂತಿರುಗಿ true;
        }
    }
    ಹಿಂತಿರುಗಿ false;
}

bool ggml_are_same_shape(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ ggml_is_ಖಾಲಿಗಳು(t0) ? ggml_is_ಖಾಲಿಗಳು(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_repeat_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ರಚನೆ ggml_context * ggml_init(ರಚನೆ ggml_init_params params) {
    ಸ್ಥಿರ bool is_ಮೊದಲ_call = true;

    ggml_critical_section_ಆರಂಭ();

    ಆದರೆ (is_ಮೊದಲ_call) {
        // initialize time system (required on Windows)
        ggml_ಸಮಯ_init();

        is_ಮೊದಲ_call = false;
    }

    ggml_critical_section_ಕೊನೆ();

    ರಚನೆ ggml_context * ctx = GGML_MALLOC(ಗಾತ್ರ(ರಚನೆ ggml_context));

    // allow to call ggml_init with 0 size
    ಆದರೆ (params.mem_ಗಾತ್ರ == 0) {
        params.mem_ಗಾತ್ರ = GGML_MEM_ALIGN;
    }

    ಸ್ಥಿರಾಂಕ size_t mem_ಗಾತ್ರ = params.mem_ಬಫರ್ ? params.mem_ಗಾತ್ರ : GGML_PAD(params.mem_ಗಾತ್ರ, GGML_MEM_ALIGN);

    *ctx = (ರಚನೆ ggml_context) {
        /*.mem_size           =*/ mem_ಗಾತ್ರ,
        /*.mem_buffer         =*/ params.mem_ಬಫರ್ ? params.mem_ಬಫರ್ : ggml_aligned_malloc(mem_ಗಾತ್ರ),
        /*.mem_buffer_owned   =*/ params.mem_ಬಫರ್ ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ಖಾಲಿ,
        /*.objects_end        =*/ ಖಾಲಿ,
    };

    GGML_ASSERT(ctx->mem_ಬಫರ್ != ಖಾಲಿ);

    GGML_ASSERT_ALIGNED(ctx->mem_ಬಫರ್);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ಹಿಂತಿರುಗಿ ctx;
}

ಶೂನ್ಯ ggml_reset(ರಚನೆ ggml_context * ctx) {
    ಆದರೆ (ctx == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ಖಾಲಿ;
    ctx->objects_ಕೊನೆ   = ಖಾಲಿ;
}

ಶೂನ್ಯ ggml_free(ರಚನೆ ggml_context * ctx) {
    ಆದರೆ (ctx == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ctx->mem_ಬಫರ್_owned) {
        ggml_aligned_free(ctx->mem_ಬಫರ್, ctx->mem_ಗಾತ್ರ);
    }

    GGML_FREE(ctx);
}

size_t ggml_ಬಳಸಿದ_mem(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->objects_ಕೊನೆ == ಖಾಲಿ ? 0 : ctx->objects_ಕೊನೆ->offs + ctx->objects_ಕೊನೆ->ಗಾತ್ರ_2;
}

bool ggml_get_no_alloc(ರಚನೆ ggml_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->no_alloc;
}

ಶೂನ್ಯ ggml_ಗಣ_no_alloc(ರಚನೆ ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

ಶೂನ್ಯ * ggml_get_mem_ಬಫರ್(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->mem_ಬಫರ್;
}

size_t ggml_get_mem_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->mem_ಗಾತ್ರ;
}

size_t ggml_get_max_tensor_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    size_t max_ಗಾತ್ರ = 0;

    ಗಾಗಿ (ರಚನೆ ggml_tensor * tensor = ggml_get_ಮೊದಲ_tensor(ctx); tensor != ಖಾಲಿ; tensor = ggml_get_ಮುಂದಿನ_tensor(ctx, tensor)) {
        size_t ಬೈಟುಗಳು = ggml_nbytes(tensor);
        max_ಗಾತ್ರ = MAX(max_ಗಾತ್ರ, ಬೈಟುಗಳು);
    }

    ಹಿಂತಿರುಗಿ max_ಗಾತ್ರ;
}

////////////////////////////////////////////////////////////////////////////////

ಸ್ಥಿರ ರಚನೆ ggml_object * ggml_new_object(ರಚನೆ ggml_context * ctx, ಎಣಿಕೆ ggml_object_type type, size_t ಗಾತ್ರ_2) {
    // always insert objects at the end of the context's memory pool
    ರಚನೆ ggml_object * obj_cur = ctx->objects_ಕೊನೆ;

    ಸ್ಥಿರಾಂಕ size_t cur_offs = obj_cur == ಖಾಲಿ ? 0 : obj_cur->offs;
    ಸ್ಥಿರಾಂಕ size_t cur_ಗಾತ್ರ = obj_cur == ಖಾಲಿ ? 0 : obj_cur->ಗಾತ್ರ_2;
    ಸ್ಥಿರಾಂಕ size_t cur_ಕೊನೆ  = cur_offs + cur_ಗಾತ್ರ;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(ಗಾತ್ರ_2 <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t ಗಾತ್ರ_needed = GGML_PAD(ಗಾತ್ರ_2, GGML_MEM_ALIGN);

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ mem_ಬಫರ್ = ctx->mem_ಬಫರ್;
    ರಚನೆ ggml_object * ಸ್ಥಿರಾಂಕ obj_new = (ರಚನೆ ggml_object *)(mem_ಬಫರ್ + cur_ಕೊನೆ);

    // integer overflow checks
    ಆದರೆ (cur_ಕೊನೆ > SIZE_MAX - ಗಾತ್ರ_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_ಕೊನೆ, ಗಾತ್ರ_needed);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಆದರೆ (cur_ಕೊನೆ + ಗಾತ್ರ_needed > SIZE_MAX - GGML_OBJECT_ಗಾತ್ರ) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_ಕೊನೆ, ಗಾತ್ರ_needed, (size_t) GGML_OBJECT_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (cur_ಕೊನೆ + ಗಾತ್ರ_needed + GGML_OBJECT_ಗಾತ್ರ > ctx->mem_ಗಾತ್ರ) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_ಕೊನೆ + ಗಾತ್ರ_needed + GGML_OBJECT_ಗಾತ್ರ, ctx->mem_ಗಾತ್ರ);
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    *obj_new = (ರಚನೆ ggml_object) {
        .offs = cur_ಕೊನೆ + GGML_OBJECT_ಗಾತ್ರ,
        .ಗಾತ್ರ_2 = ಗಾತ್ರ_needed,
        .ಮುಂದಿನ = ಖಾಲಿ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_ಬಫರ್ + obj_new->offs);

    ಆದರೆ (obj_cur != ಖಾಲಿ) {
        obj_cur->ಮುಂದಿನ = obj_new;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_ಕೊನೆ = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ಹಿಂತಿರುಗಿ obj_new;
}

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_new_tensor_impl(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type      type,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಸ್ಥಿರಾಂಕ int64_t       * ne,
        ರಚನೆ ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_ಎಣಿಕೆ);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    ಆದರೆ (view_src != ಖಾಲಿ && view_src->view_src != ಖಾಲಿ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < n_dims; i++) {
        ಆದರೆ (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t ದತ್ತಾಂಶ_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(type, ne[0]);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < n_dims; i++) {
        ದತ್ತಾಂಶ_ಗಾತ್ರ *= ne[i];
    }

    GGML_ASSERT(view_src == ಖಾಲಿ || ದತ್ತಾಂಶ_ಗಾತ್ರ == 0 || ದತ್ತಾಂಶ_ಗಾತ್ರ + view_offs <= ggml_nbytes(view_src));

    ಶೂನ್ಯ * ದತ್ತಾಂಶ = view_src != ಖಾಲಿ ? view_src->ದತ್ತಾಂಶ : ಖಾಲಿ;
    ಆದರೆ (ದತ್ತಾಂಶ != ಖಾಲಿ) {
        ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + view_offs;
    }

    size_t obj_alloc_ಗಾತ್ರ = 0;

    ಆದರೆ (view_src == ಖಾಲಿ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_ಗಾತ್ರ = ದತ್ತಾಂಶ_ಗಾತ್ರ;
    }

    GGML_ASSERT(GGML_TENSOR_ಗಾತ್ರ <= SIZE_MAX - obj_alloc_ಗಾತ್ರ);

    ರಚನೆ ggml_object * ಸ್ಥಿರಾಂಕ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_ಗಾತ್ರ + obj_alloc_ಗಾತ್ರ);
    GGML_ASSERT(obj_new);

    ರಚನೆ ggml_tensor * ಸ್ಥಿರಾಂಕ ಫಲಿತಾಂಶ = (ರಚನೆ ggml_tensor *)((ಅಕ್ಷರ *)ctx->mem_ಬಫರ್ + obj_new->offs);

    *ಫಲಿತಾಂಶ = (ರಚನೆ ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ಖಾಲಿ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ಖಾಲಿ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_ಗಾತ್ರ > 0 ? (ಶೂನ್ಯ *)(ಫಲಿತಾಂಶ + 1) : ದತ್ತಾಂಶ,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ಖಾಲಿ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_dims; i++) {
        ಫಲಿತಾಂಶ->ne[i] = ne[i];
    }

    ಫಲಿತಾಂಶ->nb[0] = ggml_type_ಗಾತ್ರ(type);
    ಫಲಿತಾಂಶ->nb[1] = ಫಲಿತಾಂಶ->nb[0]*(ಫಲಿತಾಂಶ->ne[0]/ggml_blck_ಗಾತ್ರ(type));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ->nb[i] = ಫಲಿತಾಂಶ->nb[i - 1]*ಫಲಿತಾಂಶ->ne[i - 1];
    }

    ctx->n_objects++;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_new_tensor(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type      type,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಸ್ಥಿರಾಂಕ int64_t       * ne) {
    ಹಿಂತಿರುಗಿ ggml_new_tensor_impl(ctx, type, n_dims, ne, ಖಾಲಿ, 0);
}

ರಚನೆ ggml_tensor * ggml_new_tensor_1d(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type      type,
        int64_t ne0) {
    ಹಿಂತಿರುಗಿ ggml_new_tensor(ctx, type, 1, &ne0);
}

ರಚನೆ ggml_tensor * ggml_new_tensor_2d(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ಸ್ಥಿರಾಂಕ int64_t ne[2] = { ne0, ne1 };
    ಹಿಂತಿರುಗಿ ggml_new_tensor(ctx, type, 2, ne);
}

ರಚನೆ ggml_tensor * ggml_new_tensor_3d(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ಸ್ಥಿರಾಂಕ int64_t ne[3] = { ne0, ne1, ne2 };
    ಹಿಂತಿರುಗಿ ggml_new_tensor(ctx, type, 3, ne);
}

ರಚನೆ ggml_tensor * ggml_new_tensor_4d(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ಹಿಂತಿರುಗಿ ggml_new_tensor(ctx, type, 4, ne);
}

ಶೂನ್ಯ * ggml_new_ಬಫರ್(ರಚನೆ ggml_context * ctx, size_t nbytes) {
    ರಚನೆ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_ಬಫರ್, nbytes);

    ಹಿಂತಿರುಗಿ (uint8_t *)ctx->mem_ಬಫರ್ + obj->offs;
}

ರಚನೆ ggml_tensor * ggml_dup_tensor(ರಚನೆ ggml_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src) {
    ಹಿಂತಿರುಗಿ ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

ಶೂನ್ಯ ggml_unravel_ಸೂಚ್ಯಂಕ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ಸ್ಥಿರಾಂಕ int64_t ne2 = tensor->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = tensor->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t i3_ = (i/(ne2*ne1*ne0));
    ಸ್ಥಿರಾಂಕ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ಸ್ಥಿರಾಂಕ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ಸ್ಥಿರಾಂಕ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    ಆದರೆ (i0) {
        * i0 = i0_;
    }
    ಆದರೆ (i1) {
        * i1 = i1_;
    }
    ಆದರೆ (i2) {
        * i2 = i2_;
    }
    ಆದರೆ (i3) {
        * i3 = i3_;
    }
}

ಶೂನ್ಯ * ggml_get_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor->ದತ್ತಾಂಶ;
}

ದಶಮಾಂಶ * ggml_get_ದತ್ತಾಂಶ_f32(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಪ್ರತಿಜ್ಞಾ(tensor->type == GGML_TYPE_F32);
    ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ *)(tensor->ದತ್ತಾಂಶ);
}

ಎಣಿಕೆ ggml_unary_op ggml_get_unary_op(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ಹಿಂತಿರುಗಿ (ಎಣಿಕೆ ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

ಎಣಿಕೆ ggml_glu_op ggml_get_glu_op(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ಹಿಂತಿರುಗಿ (ಎಣಿಕೆ ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_get_name(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor->name;
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_name(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    size_t i;
    ಗಾಗಿ (i = 0; i < ಗಾತ್ರ(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ಹಿಂತಿರುಗಿ tensor;
}

ರಚನೆ ggml_tensor * ggml_format_name(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fmt, ...) {
    va_list ವಾದಗಳು_2;
    va_start(ವಾದಗಳು_2, fmt);
    vsnprintf(tensor->name, ಗಾತ್ರ(tensor->name), fmt, ವಾದಗಳು_2);
    va_end(ವಾದಗಳು_2);
    ಹಿಂತಿರುಗಿ tensor;
}

ರಚನೆ ggml_tensor * ggml_view_tensor(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * src) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (view)", src->name);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ->nb[i] = src->nb[i];
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_get_ಮೊದಲ_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx) {
    ರಚನೆ ggml_object * obj = ctx->objects_begin;

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ mem_ಬಫರ್ = ctx->mem_ಬಫರ್;

    ಆಗ (obj != ಖಾಲಿ) {
        ಆದರೆ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ಹಿಂತಿರುಗಿ (ರಚನೆ ggml_tensor *)(mem_ಬಫರ್ + obj->offs);
        }

        obj = obj->ಮುಂದಿನ;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ರಚನೆ ggml_tensor * ggml_get_ಮುಂದಿನ_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_context * ctx, ರಚನೆ ggml_tensor * tensor) {
    ರಚನೆ ggml_object * obj = (ರಚನೆ ggml_object *) ((ಅಕ್ಷರ *)tensor - GGML_OBJECT_ಗಾತ್ರ);
    obj = obj->ಮುಂದಿನ;

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ mem_ಬಫರ್ = ctx->mem_ಬಫರ್;

    ಆಗ (obj != ಖಾಲಿ) {
        ಆದರೆ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ಹಿಂತಿರುಗಿ (ರಚನೆ ggml_tensor *)(mem_ಬಫರ್ + obj->offs);
        }

        obj = obj->ಮುಂದಿನ;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ರಚನೆ ggml_tensor * ggml_get_tensor(ರಚನೆ ggml_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ರಚನೆ ggml_object * obj = ctx->objects_begin;

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ mem_ಬಫರ್ = ctx->mem_ಬಫರ್;

    ಆಗ (obj != ಖಾಲಿ) {
        ಆದರೆ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ರಚನೆ ggml_tensor * cur = (ರಚನೆ ggml_tensor *)(mem_ಬಫರ್ + obj->offs);
            ಆದರೆ (strcmp(cur->name, name) == 0) {
                ಹಿಂತಿರುಗಿ cur;
            }
        }

        obj = obj->ಮುಂದಿನ;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_dup_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DUP;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_dup(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_dup_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_dup_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_dup_impl(ctx, a, true);
}

// ggml_add

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಕೂಡಿಸು;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_ಕೂಡಿಸು_impl(ctx, a, b, false);
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_ಕೂಡಿಸು_impl(ctx, a, b, true);
}

// ggml_add_cast

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_cast_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಎಣಿಕೆ   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_ಅಡ್ಡಸಾಲುಗಳು(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಕೂಡಿಸು;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_cast(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಎಣಿಕೆ   ggml_type      type) {
    ಹಿಂತಿರುಗಿ ggml_ಕೂಡಿಸು_cast_impl(ctx, a, b, type);
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_id(
            ರಚನೆ ggml_context * ctx,
            ರಚನೆ ggml_tensor  * a,
            ರಚನೆ ggml_tensor  * b,
            ರಚನೆ ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಕೂಡಿಸು_ID;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಫಲಿತಾಂಶ->src[2] = ids;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_add1

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_add1_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ADD1;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_add1(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_add1_impl(ctx, a, b, false);
}

ರಚನೆ ggml_tensor * ggml_add1_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_acc_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(ಸ್ಥಾನಾಂತರ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, inplace ? 1 : 0 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_ACC;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_acc(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, false);
}

ರಚನೆ ggml_tensor * ggml_acc_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, true);
}

// ggml_sub

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_sub_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SUB;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_sub(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_sub_impl(ctx, a, b, false);
}

ರಚನೆ ggml_tensor * ggml_sub_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_mul_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_MUL;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_mul(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_mul_impl(ctx, a, b, false);
}

ರಚನೆ ggml_tensor * ggml_mul_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_div_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DIV;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_div(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_div_impl(ctx, a, b, false);
}

ರಚನೆ ggml_tensor * ggml_div_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_sqr_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SQR;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_sqr(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sqr_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_sqr_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_sqrt_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SQRT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_sqrt(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sqrt_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_sqrt_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_log_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_LOG;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_log(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_log_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_log_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_log_impl(ctx, a, true);
}

ರಚನೆ ggml_tensor * ggml_expm1(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ರಚನೆ ggml_tensor * ggml_expm1_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ರಚನೆ ggml_tensor * ggml_softplus(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ರಚನೆ ggml_tensor * ggml_softplus_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_sin_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SIN;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_sin(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sin_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_sin_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_cos_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_COS;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_cos(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_cos_impl(ctx, a, false);
}

ರಚನೆ ggml_tensor * ggml_cos_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ರಚನೆ ggml_tensor * ggml_ಮೊತ್ತ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, a->type, 1);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಮೊತ್ತ;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_sum_rows

ರಚನೆ ggml_tensor * ggml_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_cumsum

ರಚನೆ ggml_tensor * ggml_cumsum(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CUMSUM;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_mean

ರಚನೆ ggml_tensor * ggml_ಸರಾಸರಿ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಸರಾಸರಿ;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_argmax

ರಚನೆ ggml_tensor * ggml_argmax(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ARGMAX;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_count_equal

ರಚನೆ ggml_tensor * ggml_ಎಣಿಕೆ_equal(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಎಣಿಕೆ_EQUAL;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_repeat

ರಚನೆ ggml_tensor * ggml_repeat(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_REPEAT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_repeat_4d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ಸ್ಥಿರಾಂಕ bool can_repeat = ggml_is_ಖಾಲಿಗಳು(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ಫಲಿತಾಂಶ->op     = GGML_OP_REPEAT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_repeat_back

ರಚನೆ ggml_tensor * ggml_repeat_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_REPEAT_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_concat

ರಚನೆ ggml_tensor * ggml_concat(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ರಚನೆ ggml_tensor  * b,
    ಪೂರ್ಣಾಂಕ                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < GGML_MAX_DIMS; ++d) {
        ಆದರೆ (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ಮುಂದುವರಿ;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, dim);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONCAT;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_abs

ರಚನೆ ggml_tensor * ggml_abs(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ರಚನೆ ggml_tensor * ggml_abs_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ರಚನೆ ggml_tensor * ggml_sgn(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ರಚನೆ ggml_tensor * ggml_sgn_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ರಚನೆ ggml_tensor * ggml_neg(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ರಚನೆ ggml_tensor * ggml_neg_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ರಚನೆ ggml_tensor * ggml_ಹೆಜ್ಜೆ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_ಹೆಜ್ಜೆ);
}

ರಚನೆ ggml_tensor * ggml_ಹೆಜ್ಜೆ_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ಹೆಜ್ಜೆ);
}

// ggml_tanh

ರಚನೆ ggml_tensor * ggml_tanh(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ರಚನೆ ggml_tensor * ggml_tanh_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ರಚನೆ ggml_tensor * ggml_elu(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ರಚನೆ ggml_tensor * ggml_elu_inplace(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ರಚನೆ ggml_tensor * ggml_relu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ರಚನೆ ggml_tensor * ggml_relu_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ರಚನೆ ggml_tensor * ggml_leaky_relu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 negative_ಇಳಿಜಾರು,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &negative_ಇಳಿಜಾರು, ಗಾತ್ರ(negative_ಇಳಿಜಾರು));

    ಫಲಿತಾಂಶ->op     = GGML_OP_LEAKY_RELU;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_sigmoid

ರಚನೆ ggml_tensor * ggml_sigmoid(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ರಚನೆ ggml_tensor * ggml_sigmoid_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ರಚನೆ ggml_tensor * ggml_gelu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ರಚನೆ ggml_tensor * ggml_gelu_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ರಚನೆ ggml_tensor * ggml_gelu_erf(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ರಚನೆ ggml_tensor * ggml_gelu_erf_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ರಚನೆ ggml_tensor * ggml_gelu_ಶೀಘ್ರ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ಶೀಘ್ರ);
}

ರಚನೆ ggml_tensor * ggml_gelu_ಶೀಘ್ರ_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ಶೀಘ್ರ);
}

// ggml_silu

ರಚನೆ ggml_tensor * ggml_silu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ರಚನೆ ggml_tensor * ggml_silu_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ರಚನೆ ggml_tensor * ggml_xielu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ alpha_n,
        ದಶಮಾಂಶ alpha_p,
        ದಶಮಾಂಶ beta,
        ದಶಮಾಂಶ eps) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 3, beta);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 4, eps);

    ಫಲಿತಾಂಶ->op     = GGML_OP_UNARY;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_silu_back

ರಚನೆ ggml_tensor * ggml_silu_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SILU_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml hardswish

ರಚನೆ ggml_tensor * ggml_hardswish(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ರಚನೆ ggml_tensor * ggml_hardsigmoid(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ರಚನೆ ggml_tensor * ggml_exp(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ರಚನೆ ggml_tensor * ggml_exp_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_glu_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಎಣಿಕೆ ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    ಆದರೆ (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ಖಾಲಿ, 0);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) op);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, (int32_t) swapped);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GLU;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_floor

ರಚನೆ ggml_tensor * ggml_floor(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ರಚನೆ ggml_tensor * ggml_floor_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ರಚನೆ ggml_tensor * ggml_ceil(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ರಚನೆ ggml_tensor * ggml_ceil_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ರಚನೆ ggml_tensor * ggml_ಸುತ್ತು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_ಸುತ್ತು);
}

ರಚನೆ ggml_tensor * ggml_ಸುತ್ತು_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ಸುತ್ತು);
}

//ggml_trunc

ರಚನೆ ggml_tensor * ggml_trunc(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ರಚನೆ ggml_tensor * ggml_trunc_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ರಚನೆ ggml_tensor * ggml_glu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_glu_op      op,
        bool                  swapped) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, op, swapped);
}

ರಚನೆ ggml_tensor * ggml_glu_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಎಣಿಕೆ ggml_glu_op      op) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ರಚನೆ ggml_tensor * ggml_reglu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_REGLU, false);
}

ರಚನೆ ggml_tensor * ggml_reglu_swapped(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_REGLU, true);
}

ರಚನೆ ggml_tensor * ggml_reglu_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ರಚನೆ ggml_tensor * ggml_geglu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU, false);
}

ರಚನೆ ggml_tensor * ggml_geglu_swapped(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU, true);
}

ರಚನೆ ggml_tensor * ggml_geglu_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ರಚನೆ ggml_tensor * ggml_swiglu(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_SWIGLU, false);
}

ರಚನೆ ggml_tensor * ggml_swiglu_swapped(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_SWIGLU, true);
}

ರಚನೆ ggml_tensor * ggml_swiglu_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ರಚನೆ ggml_tensor * ggml_geglu_erf(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU_ERF, false);
}

ರಚನೆ ggml_tensor * ggml_geglu_erf_swapped(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU_ERF, true);
}

ರಚನೆ ggml_tensor * ggml_geglu_erf_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ರಚನೆ ggml_tensor * ggml_geglu_ಶೀಘ್ರ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU_ಶೀಘ್ರ, false);
}

ರಚನೆ ggml_tensor * ggml_geglu_ಶೀಘ್ರ_swapped(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, ಖಾಲಿ, GGML_GLU_OP_GEGLU_ಶೀಘ್ರ, true);
}

ರಚನೆ ggml_tensor * ggml_geglu_ಶೀಘ್ರ_ವಿಭಜಿಸು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ಶೀಘ್ರ, false);
}

ರಚನೆ ggml_tensor * ggml_swiglu_oai(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 alpha,
        ದಶಮಾಂಶ                 ಎಲ್ಲೆ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 2, alpha);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 3, ಎಲ್ಲೆ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_swiglu_clamp(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 ಎಲ್ಲೆ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 3, ಎಲ್ಲೆ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_norm

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_norm_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &eps, ಗಾತ್ರ(eps));

    ಫಲಿತಾಂಶ->op     = GGML_OP_NORM;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_norm(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_norm_impl(ctx, a, eps, false);
}

ರಚನೆ ggml_tensor * ggml_norm_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_rms_norm_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &eps, ಗಾತ್ರ(eps));

    ಫಲಿತಾಂಶ->op     = GGML_OP_RMS_NORM;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_rms_norm(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_rms_norm_impl(ctx, a, eps, false);
}

ರಚನೆ ggml_tensor * ggml_rms_norm_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ರಚನೆ ggml_tensor * ggml_rms_norm_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 eps) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &eps, ಗಾತ್ರ(eps));

    ಫಲಿತಾಂಶ->op     = GGML_OP_RMS_NORM_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_group_norm

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ಗುಂಪು_norm_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_groups,
        ದಶಮಾಂಶ                 eps,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, n_groups);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 1, eps);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಗುಂಪು_NORM;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಗುಂಪು_norm(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_groups,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_ಗುಂಪು_norm_impl(ctx, a, n_groups, eps, false);
}

ರಚನೆ ggml_tensor * ggml_ಗುಂಪು_norm_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_groups,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_ಗುಂಪು_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_l2_norm_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, eps);

    ಫಲಿತಾಂಶ->op     = GGML_OP_L2_NORM;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_l2_norm(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_l2_norm_impl(ctx, a, eps, false);
}

ರಚನೆ ggml_tensor * ggml_l2_norm_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 eps) {
    ಹಿಂತಿರುಗಿ ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_ಗಣ_acc(
        ರಚನೆ ggml_tensor * a,
        ಎಣಿಕೆ ggml_prec       prec) {
    ಆಯ್ಕೆ (a->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ಸ್ಥಿರಾಂಕ int32_t prec_i32 = (int32_t) prec;
                ggml_ಗಣ_op_params_i32(a, 0, prec_i32);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                ಸ್ಥಿರಾಂಕ int32_t prec_i32 = (int32_t) prec;
                ggml_ಗಣ_op_params_i32(a, 3, prec_i32);
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ false;
    };

    ಹಿಂತಿರುಗಿ true;
}

bool ggml_prec_ಗಣ_src(
        ರಚನೆ ggml_tensor * a,
        ಎಣಿಕೆ ggml_prec       prec,
        ಪೂರ್ಣಾಂಕ                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ಆಯ್ಕೆ (a->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ಆದರೆ (idx != 1) {
                    ಹಿಂತಿರುಗಿ false;
                }

                ಸ್ಥಿರಾಂಕ int32_t prec_i32 = (int32_t) prec;
                ggml_ಗಣ_op_params_i32(a, 2 + idx, prec_i32);
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ false;
    };

    ಹಿಂತಿರುಗಿ true;
}

// ggml_mul_mat

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_mul_mat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ರಚನೆ ggml_tensor * ggml_mul_mat(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_MUL_MAT;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_mul_mat_ಗಣ_prec(
        ರಚನೆ ggml_tensor * a,
        ಎಣಿಕೆ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ಸ್ಥಿರಾಂಕ int32_t prec_i32 = (int32_t) prec;

    ggml_ಗಣ_op_params_i32(a, 0, prec_i32);
}

ಶೂನ್ಯ ggml_mul_mat_ಗಣ_hint(
        ರಚನೆ ggml_tensor * a,
        ಎಣಿಕೆ ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ಸ್ಥಿರಾಂಕ int32_t hint_i32 = (int32_t) hint;

    ggml_ಗಣ_op_params_i32(a, 1, hint_i32);
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
ರಚನೆ ggml_tensor * ggml_mul_mat_id(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * as,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_MUL_MAT_ID;
    ಫಲಿತಾಂಶ->src[0] = as;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಫಲಿತಾಂಶ->src[2] = ids;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_out_prod

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_ಹೊರಹೋಗು_prod(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ರಚನೆ ggml_tensor * ggml_ಹೊರಹೋಗು_prod(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_ಹೊರಹೋಗು_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಹೊರಹೋಗು_PROD;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_scale

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_scale_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 s,
        ದಶಮಾಂಶ                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ದಶಮಾಂಶ params[2] = { s, b };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_SCALE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_scale(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 s) {
    ಹಿಂತಿರುಗಿ ggml_scale_impl(ctx, a, s, 0.0, false);
}

ರಚನೆ ggml_tensor * ggml_scale_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 s) {
    ಹಿಂತಿರುಗಿ ggml_scale_impl(ctx, a, s, 0.0, true);
}

ರಚನೆ ggml_tensor * ggml_scale_bias(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 s,
        ದಶಮಾಂಶ                 b) {
    ಹಿಂತಿರುಗಿ ggml_scale_impl(ctx, a, s, b, false);
}

ರಚನೆ ggml_tensor * ggml_scale_bias_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 s,
        ದಶಮಾಂಶ                 b) {
    ಹಿಂತಿರುಗಿ ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ಗಣ_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(ಸ್ಥಾನಾಂತರ < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, inplace ? 1 : 0 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಗಣ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಗಣ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, false);
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, true);
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], ಸ್ಥಾನಾಂತರ, false);
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_1d_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], ಸ್ಥಾನಾಂತರ, true);
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], ಸ್ಥಾನಾಂತರ, false);
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_2d_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        size_t                nb1,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ ggml_ಗಣ_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], ಸ್ಥಾನಾಂತರ, true);
}

// ggml_cpy

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_cpy_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, b);
    ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(b->name) > 0) {
        ggml_format_name(ಫಲಿತಾಂಶ, "%s (copy of %s)", b->name, a->name);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_format_name(ಫಲಿತಾಂಶ, "%s (copy)", a->name);
    }

    ಫಲಿತಾಂಶ->op     = GGML_OP_CPY;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_cpy(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor * a,
        ರಚನೆ ggml_tensor * b) {
    ಹಿಂತಿರುಗಿ ggml_cpy_impl(ctx, a, b);
}

ರಚನೆ ggml_tensor * ggml_cast(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ   ggml_type      type) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (copy)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CPY;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = ಫಲಿತಾಂಶ; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_cont

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_cont_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (cont)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_cont(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor * a) {
    ಹಿಂತಿರುಗಿ ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ರಚನೆ ggml_tensor * ggml_cont_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0) {
    ಹಿಂತಿರುಗಿ ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ರಚನೆ ggml_tensor * ggml_cont_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ಹಿಂತಿರುಗಿ ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ರಚನೆ ggml_tensor * ggml_cont_3d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ಹಿಂತಿರುಗಿ ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ರಚನೆ ggml_tensor * ggml_cont_4d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (cont)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_reshape

ರಚನೆ ggml_tensor * ggml_reshape(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor * a,
        ರಚನೆ ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (reshaped)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RESHAPE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_reshape_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ಸ್ಥಿರಾಂಕ int64_t ne[1] = { ne0 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (reshaped)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RESHAPE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_reshape_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ಸ್ಥಿರಾಂಕ int64_t ne[2] = { ne0, ne1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (reshaped)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RESHAPE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_reshape_3d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ಸ್ಥಿರಾಂಕ int64_t ne[3] = { ne0, ne1, ne2 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (reshaped)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RESHAPE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_reshape_4d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (reshaped)", a->name);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RESHAPE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_view_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಸ್ಥಿರಾಂಕ int64_t       * ne,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, ಸ್ಥಾನಾಂತರ);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (view)", a->name);

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &ಸ್ಥಾನಾಂತರ, ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ));

    ಫಲಿತಾಂಶ->op     = GGML_OP_VIEW;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_view_1d

ರಚನೆ ggml_tensor * ggml_view_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_impl(ctx, a, 1, &ne0, ಸ್ಥಾನಾಂತರ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_view_2d

ರಚನೆ ggml_tensor * ggml_view_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಸ್ಥಿರಾಂಕ int64_t ne[2] = { ne0, ne1 };

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_impl(ctx, a, 2, ne, ಸ್ಥಾನಾಂತರ);

    ಫಲಿತಾಂಶ->nb[1] = nb1;
    ಫಲಿತಾಂಶ->nb[2] = ಫಲಿತಾಂಶ->nb[1]*ne1;
    ಫಲಿತಾಂಶ->nb[3] = ಫಲಿತಾಂಶ->nb[2];

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_view_3d

ರಚನೆ ggml_tensor * ggml_view_3d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಸ್ಥಿರಾಂಕ int64_t ne[3] = { ne0, ne1, ne2 };

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_impl(ctx, a, 3, ne, ಸ್ಥಾನಾಂತರ);

    ಫಲಿತಾಂಶ->nb[1] = nb1;
    ಫಲಿತಾಂಶ->nb[2] = nb2;
    ಫಲಿತಾಂಶ->nb[3] = ಫಲಿತಾಂಶ->nb[2]*ne2;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_view_4d

ರಚನೆ ggml_tensor * ggml_view_4d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                ಸ್ಥಾನಾಂತರ) {
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_impl(ctx, a, 4, ne, ಸ್ಥಾನಾಂತರ);

    ಫಲಿತಾಂಶ->nb[1] = nb1;
    ಫಲಿತಾಂಶ->nb[2] = nb2;
    ಫಲಿತಾಂಶ->nb[3] = nb3;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_permute

ರಚನೆ ggml_tensor * ggml_ಕ್ರಮಬದಲಾವಣೆ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   axis0,
        ಪೂರ್ಣಾಂಕ                   axis1,
        ಪೂರ್ಣಾಂಕ                   axis2,
        ಪೂರ್ಣಾಂಕ                   axis3) {
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

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (permuted)", a->name);

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

    ಫಲಿತಾಂಶ->ne[0] = ne[0];
    ಫಲಿತಾಂಶ->ne[1] = ne[1];
    ಫಲಿತಾಂಶ->ne[2] = ne[2];
    ಫಲಿತಾಂಶ->ne[3] = ne[3];

    ಫಲಿತಾಂಶ->nb[0] = nb[0];
    ಫಲಿತಾಂಶ->nb[1] = nb[1];
    ಫಲಿತಾಂಶ->nb[2] = nb[2];
    ಫಲಿತಾಂಶ->nb[3] = nb[3];

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಕ್ರಮಬದಲಾವಣೆ;
    ಫಲಿತಾಂಶ->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_transpose

ರಚನೆ ggml_tensor * ggml_transpose(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);
    ggml_format_name(ಫಲಿತಾಂಶ, "%s (transposed)", a->name);

    ಫಲಿತಾಂಶ->ne[0] = a->ne[1];
    ಫಲಿತಾಂಶ->ne[1] = a->ne[0];

    ಫಲಿತಾಂಶ->nb[0] = a->nb[1];
    ಫಲಿತಾಂಶ->nb[1] = a->nb[0];

    ಫಲಿತಾಂಶ->op     = GGML_OP_TRANSPOSE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_get_rows

ರಚನೆ ggml_tensor * ggml_get_ಅಡ್ಡಸಾಲುಗಳು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    ಎಣಿಕೆ ggml_type type = GGML_TYPE_F32;
    ಆದರೆ (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_get_rows_back

ರಚನೆ ggml_tensor * ggml_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(a) && ggml_is_ಸದಿಶ(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_set_rows

ರಚನೆ ggml_tensor * ggml_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(a));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(b));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
    ಫಲಿತಾಂಶ->src[0] = b;
    ಫಲಿತಾಂಶ->src[1] = c;
    ಫಲಿತಾಂಶ->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_diag

ರಚನೆ ggml_tensor * ggml_diag(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DIAG;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_diag_mask_inf

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_inf_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_DIAG_ಮುಖವಾಡ_INF;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_inf(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past) {
    ಹಿಂತಿರುಗಿ ggml_diag_ಮುಖವಾಡ_inf_impl(ctx, a, n_past, false);
}

ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_inf_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past) {
    ಹಿಂತಿರುಗಿ ggml_diag_ಮುಖವಾಡ_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_zero_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_DIAG_ಮುಖವಾಡ_ZERO;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_zero(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past) {
    ಹಿಂತಿರುಗಿ ggml_diag_ಮುಖವಾಡ_zero_impl(ctx, a, n_past, false);
}

ರಚನೆ ggml_tensor * ggml_diag_ಮುಖವಾಡ_zero_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_past) {
    ಹಿಂತಿರುಗಿ ggml_diag_ಮುಖವಾಡ_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_clamp_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ದಶಮಾಂಶ                 min,
        ದಶಮಾಂಶ                 max,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ದಶಮಾಂಶ params[] = { min, max };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_CLAMP;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_clamp(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ದಶಮಾಂಶ                 min,
    ದಶಮಾಂಶ                 max) {
    ಹಿಂತಿರುಗಿ ggml_clamp_impl(ctx, a, min, max, false);
}

ರಚನೆ ggml_tensor * ggml_clamp_inplace(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ದಶಮಾಂಶ                 min,
    ದಶಮಾಂಶ                 max) {
    ಹಿಂತಿರುಗಿ ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_soft_max_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    ಆದರೆ (ಮುಖವಾಡ) {
        GGML_ASSERT(ಮುಖವಾಡ->type == GGML_TYPE_F16 || ಮುಖವಾಡ->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(ಮುಖವಾಡ));
        GGML_ASSERT(ಮುಖವಾಡ->ne[0] == a->ne[0]);
        GGML_ASSERT(ಮುಖವಾಡ->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%ಮುಖವಾಡ->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%ಮುಖವಾಡ->ne[3] == 0);
    }

    ಆದರೆ (max_bias > 0.0f) {
        GGML_ASSERT(ಮುಖವಾಡ);
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ದಶಮಾಂಶ params[] = { scale, max_bias };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_SOFT_MAX;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = ಮುಖವಾಡ;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_soft_max(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_impl(ctx, a, ಖಾಲಿ, 1.0f, 0.0f, false);
}

ರಚನೆ ggml_tensor * ggml_soft_max_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_impl(ctx, a, ಖಾಲಿ, 1.0f, 0.0f, true);
}

ರಚನೆ ggml_tensor * ggml_soft_max_ext(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_impl(ctx, a, ಮುಖವಾಡ, scale, max_bias, false);
}

ರಚನೆ ggml_tensor * ggml_soft_max_ext_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_impl(ctx, a, ಮುಖವಾಡ, scale, max_bias, true);
}

ಶೂನ್ಯ ggml_soft_max_ಕೂಡಿಸು_sinks(
        ರಚನೆ ggml_tensor * a,
        ರಚನೆ ggml_tensor * sinks) {
    ಆದರೆ (!sinks) {
        a->src[2] = ಖಾಲಿ;
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ಖಾಲಿ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_soft_max_ext_ಹಿಂಭಾಗ_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias,
        bool                  inplace) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಸ್ಮರಣೆ_ನಕಲು((ದಶಮಾಂಶ *) ಫಲಿತಾಂಶ->op_params + 0, &scale,    ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು((ದಶಮಾಂಶ *) ಫಲಿತಾಂಶ->op_params + 1, &max_bias, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_soft_max_ext_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_ext_ಹಿಂಭಾಗ_impl(ctx, a, b, scale, max_bias, false);
}

ರಚನೆ ggml_tensor * ggml_soft_max_ext_ಹಿಂಭಾಗ_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias) {
    ಹಿಂತಿರುಗಿ ggml_soft_max_ext_ಹಿಂಭಾಗ_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_rope_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((ಬಹುಲಕ & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_ಸದಿಶ(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_ಬಳಸಿದ = ಬಹುಲಕ & GGML_ROPE_TYPE_MROPE;
    ಆದರೆ (mrope_ಬಳಸಿದ) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    ಆದರೆ (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, ಬಹುಲಕ, /*n_ctx*/ 0, n_ctx_orig };
    ಸ್ಮರಣೆ_ನಕಲು(params +  5, &freq_ತಳ,    ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(params +  6, &freq_scale,   ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(params +  7, &ext_ಗುಣಕ,   ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(params +  8, &attn_ಗುಣಕ,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(params +  9, &beta_fast,    ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(params + 10, &beta_slow,    ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (mrope_ಬಳಸಿದ && sections) {
        ಸ್ಮರಣೆ_ನಕಲು(params + 11, sections,  ಗಾತ್ರ(int32_t) * GGML_MROPE_SECTIONS);
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(params + 11, 0,         ಗಾತ್ರ(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_ROPE;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಫಲಿತಾಂಶ->src[2] = c;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_rope(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, ಖಾಲಿ, n_dims, ಖಾಲಿ, ಬಹುಲಕ, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ರಚನೆ ggml_tensor * ggml_rope_multi(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, false
    );
}

ರಚನೆ ggml_tensor * ggml_rope_multi_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, true
    );
}

ರಚನೆ ggml_tensor * ggml_rope_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, ಖಾಲಿ, n_dims, ಖಾಲಿ, ಬಹುಲಕ, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ರಚನೆ ggml_tensor * ggml_rope_ext(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, c, n_dims, ಖಾಲಿ, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, false
    );
}

ರಚನೆ ggml_tensor * ggml_rope_ext_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, c, n_dims, ಖಾಲಿ, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, true
    );
}

ರಚನೆ ggml_tensor * ggml_rope_custom(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, ಖಾಲಿ, n_dims, ಖಾಲಿ, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, false
    );
}

ರಚನೆ ggml_tensor * ggml_rope_custom_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ಹಿಂತಿರುಗಿ ggml_rope_impl(
        ctx, a, b, ಖಾಲಿ, n_dims, ಖಾಲಿ, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale,
        ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ಸ್ಥಿರ ದಶಮಾಂಶ ggml_rope_yarn_corr_dim(ಪೂರ್ಣಾಂಕ n_dims, ಪೂರ್ಣಾಂಕ n_ctx_orig, ದಶಮಾಂಶ n_rot, ದಶಮಾಂಶ ತಳ) {
    ಹಿಂತಿರುಗಿ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ದಶಮಾಂಶ)M_PI)) / (2 * logf(ತಳ));
}

ಶೂನ್ಯ ggml_rope_yarn_corr_dims(
    ಪೂರ್ಣಾಂಕ n_dims, ಪೂರ್ಣಾಂಕ n_ctx_orig, ದಶಮಾಂಶ freq_ತಳ, ದಶಮಾಂಶ beta_fast, ದಶಮಾಂಶ beta_slow, ದಶಮಾಂಶ dims[2]
) {
    // start and end correction dims
    ದಶಮಾಂಶ ಆರಂಭ = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ತಳ));
    ದಶಮಾಂಶ ಕೊನೆ   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ತಳ));
    dims[0] = MAX(0, ಆರಂಭ);
    dims[1] = MIN(n_dims - 1, ಕೊನೆ);
}

// ggml_rope_back

ರಚನೆ ggml_tensor * ggml_rope_ext_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_rope_ext(
        ctx, a, b, c, n_dims, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
    ಫಲಿತಾಂಶ->op = GGML_OP_ROPE_ಹಿಂಭಾಗ;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_rope_multi_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c,
        ಪೂರ್ಣಾಂಕ                   n_dims,
        ಪೂರ್ಣಾಂಕ                   sections[4],
        ಪೂರ್ಣಾಂಕ                   ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ                   n_ctx_orig,
        ದಶಮಾಂಶ                 freq_ತಳ,
        ದಶಮಾಂಶ                 freq_scale,
        ದಶಮಾಂಶ                 ext_ಗುಣಕ,
        ದಶಮಾಂಶ                 attn_ಗುಣಕ,
        ದಶಮಾಂಶ                 beta_fast,
        ದಶಮಾಂಶ                 beta_slow) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
    ಫಲಿತಾಂಶ->op = GGML_OP_ROPE_ಹಿಂಭಾಗ;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_rope_ಗಣ_ಸ್ಥಾನಾಂತರ(
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_ಹಿಂಭಾಗ);
    GGML_ASSERT(n_offs >= 0);

    ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(ಬಹುಲಕ != GGML_ROPE_TYPE_VISION);

    ggml_ಗಣ_op_params_i32(a, 15, n_offs);
    ಹಿಂತಿರುಗಿ a;
}

ಸ್ಥಿರ int64_t ggml_calc_conv_output_ಗಾತ್ರ(int64_t ins, int64_t ks, ಪೂರ್ಣಾಂಕ s, ಪೂರ್ಣಾಂಕ p, ಪೂರ್ಣಾಂಕ d) {
    ಹಿಂತಿರುಗಿ (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ರಚನೆ ggml_tensor * ggml_im2col(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   d0,
        ಪೂರ್ಣಾಂಕ                   d1,
        bool                  is_2D,
        ಎಣಿಕೆ ggml_type        dst_type) {
    ಆದರೆ (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? ggml_calc_conv_output_ಗಾತ್ರ(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ಸ್ಥಿರಾಂಕ int64_t OW =         ggml_calc_conv_output_ಗಾತ್ರ(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_IM2_ಲಂಬಸಾಲು;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_im2col_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        int64_t             * ne,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   d0,
        ಪೂರ್ಣಾಂಕ                   d1,
        bool                  is_2D) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_1d

ರಚನೆ ggml_tensor * ggml_conv_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   d0) {
    ರಚನೆ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    ಫಲಿತಾಂಶ = ggml_reshape_3d(ctx, ಫಲಿತಾಂಶ, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    ಫಲಿತಾಂಶ = ggml_cont(ctx, ggml_ಕ್ರಮಬದಲಾವಣೆ(ctx, ಫಲಿತಾಂಶ, 0, 2, 1, 3));                // [N, OC, OL]

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_1d_ph

ರಚನೆ ggml_tensor* ggml_conv_1d_ph(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s,
        ಪೂರ್ಣಾಂಕ                   d) {
    ಹಿಂತಿರುಗಿ ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ರಚನೆ ggml_tensor * ggml_conv_1d_dw(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   d0) {
    ರಚನೆ ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ರಚನೆ ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_mul_mat(ctx, im2col, a);

    ಫಲಿತಾಂಶ = ggml_reshape_3d(ctx, ಫಲಿತಾಂಶ, ಫಲಿತಾಂಶ->ne[0], ಫಲಿತಾಂಶ->ne[2], 1);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_1d_dw_ph

ರಚನೆ ggml_tensor * ggml_conv_1d_dw_ph(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   d0) {
    ಹಿಂತಿರುಗಿ ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ರಚನೆ ggml_tensor * ggml_col2im_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   oc,
        ಪೂರ್ಣಾಂಕ                   p0) {
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ಸ್ಥಿರಾಂಕ int64_t K_OC = a->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t T_ಒಳ = a->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t K = K_OC / oc;
    ಸ್ಥಿರಾಂಕ int64_t T_ಹೊರಹೋಗು = (T_ಒಳ - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_ಹೊರಹೋಗು > 0);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { T_ಹೊರಹೋಗು, oc, 1, 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_COL2IM_1D;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_transpose_1d

ಸ್ಥಿರ int64_t ggml_calc_conv_transpose_1d_output_ಗಾತ್ರ(int64_t ins, int64_t ks, ಪೂರ್ಣಾಂಕ s, ಪೂರ್ಣಾಂಕ p, ಪೂರ್ಣಾಂಕ d) {
    ಹಿಂತಿರುಗಿ (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ರಚನೆ ggml_tensor * ggml_conv_transpose_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   d0) {
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_ಗಾತ್ರ(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONV_TRANSPOSE_1D;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ರಚನೆ ggml_tensor * ggml_conv_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   d0,
        ಪೂರ್ಣಾಂಕ                   d1) {
    ರಚನೆ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    ಫಲಿತಾಂಶ = ggml_reshape_4d(ctx, ಫಲಿತಾಂಶ, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    ಫಲಿತಾಂಶ = ggml_cont(ctx, ggml_ಕ್ರಮಬದಲಾವಣೆ(ctx, ಫಲಿತಾಂಶ, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ರಚನೆ ggml_tensor * ggml_im2col_3d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        int64_t               IC,
        ಪೂರ್ಣಾಂಕ                   s0, // stride width
        ಪೂರ್ಣಾಂಕ                   s1, // stride height
        ಪೂರ್ಣಾಂಕ                   s2, // stride depth
        ಪೂರ್ಣಾಂಕ                   p0, // padding width
        ಪೂರ್ಣಾಂಕ                   p1, // padding height
        ಪೂರ್ಣಾಂಕ                   p2, // padding depth
        ಪೂರ್ಣಾಂಕ                   d0, // dilation width
        ಪೂರ್ಣಾಂಕ                   d1, // dilation height
        ಪೂರ್ಣಾಂಕ                   d2, // dilation depth
        ಎಣಿಕೆ ggml_type        dst_type) {
    ಸ್ಥಿರಾಂಕ int64_t N = b->ne[3] / IC;
    ಸ್ಥಿರಾಂಕ int64_t ID = b->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t IH = b->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t IW = b->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t OC = a->ne[3] / IC;
    ಬಳಸದ(OC);
    ಸ್ಥಿರಾಂಕ int64_t KD = a->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t KH = a->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t KW = a->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t OD = ggml_calc_conv_output_ಗಾತ್ರ(ID, KD, s2, p2, d2);
    ಸ್ಥಿರಾಂಕ int64_t OH = ggml_calc_conv_output_ಗಾತ್ರ(IH, KH, s1, p1, d1);
    ಸ್ಥಿರಾಂಕ int64_t OW = ggml_calc_conv_output_ಗಾತ್ರ(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_IM2_ಲಂಬಸಾಲು_3_D;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ರಚನೆ ggml_tensor * ggml_conv_3d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        int64_t               IC,
        ಪೂರ್ಣಾಂಕ                   s0, // stride width
        ಪೂರ್ಣಾಂಕ                   s1, // stride height
        ಪೂರ್ಣಾಂಕ                   s2, // stride depth
        ಪೂರ್ಣಾಂಕ                   p0, // padding width
        ಪೂರ್ಣಾಂಕ                   p1, // padding height
        ಪೂರ್ಣಾಂಕ                   p2, // padding depth
        ಪೂರ್ಣಾಂಕ                   d0, // dilation width
        ಪೂರ್ಣಾಂಕ                   d1, // dilation height
        ಪೂರ್ಣಾಂಕ                   d2  // dilation depth
        ) {
    ರಚನೆ ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    ಫಲಿತಾಂಶ = ggml_reshape_4d(ctx, ಫಲಿತಾಂಶ, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    ಫಲಿತಾಂಶ = ggml_cont(ctx, ggml_ಕ್ರಮಬದಲಾವಣೆ(ctx, ಫಲಿತಾಂಶ, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    ಫಲಿತಾಂಶ = ggml_reshape_4d(ctx, ಫಲಿತಾಂಶ, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_2d_sk_p0

ರಚನೆ ggml_tensor * ggml_conv_2d_sk_p0(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ರಚನೆ ggml_tensor * ggml_conv_2d_s1_ph(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    ಹಿಂತಿರುಗಿ ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ರಚನೆ ggml_tensor * ggml_conv_2d_dw(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   d0,
        ಪೂರ್ಣಾಂಕ                   d1) {
    ರಚನೆ ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ರಚನೆ ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ರಚನೆ ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_mul_mat(ctx, new_a, new_b);
    ಫಲಿತಾಂಶ = ggml_reshape_4d(ctx, ಫಲಿತಾಂಶ, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_2d_dw_direct

ರಚನೆ ggml_tensor * ggml_conv_2d_dw_ನೇರ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   stride0,
        ಪೂರ್ಣಾಂಕ                   stride1,
        ಪೂರ್ಣಾಂಕ                   pad0,
        ಪೂರ್ಣಾಂಕ                   pad1,
        ಪೂರ್ಣಾಂಕ                   dilation0,
        ಪೂರ್ಣಾಂಕ                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, b->type, 4, ne);

    ಆದರೆ (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ಸ್ಥಿರಾಂಕ int64_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(ಫಲಿತಾಂಶ->type);
        GGML_ASSERT(ggml_blck_ಗಾತ್ರ(ಫಲಿತಾಂಶ->type) == 1);
        ಫಲಿತಾಂಶ->nb[0] = ಫಲಿತಾಂಶ->ne[2] * type_ಗಾತ್ರ;
        ಫಲಿತಾಂಶ->nb[1] = ಫಲಿತಾಂಶ->ne[0] * ಫಲಿತಾಂಶ->nb[0];
        ಫಲಿತಾಂಶ->nb[2] = type_ಗಾತ್ರ;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONV_2D_DW;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_2d_direct

ರಚನೆ ggml_tensor * ggml_conv_2d_ನೇರ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ರಚನೆ ggml_tensor  * b,   // input data [W, H, C, N]
        ಪೂರ್ಣಾಂಕ                   s0,  // stride dimension 0
        ಪೂರ್ಣಾಂಕ                   s1,  // stride dimension 1
        ಪೂರ್ಣಾಂಕ                   p0,  // padding dimension 0
        ಪೂರ್ಣಾಂಕ                   p1,  // padding dimension 1
        ಪೂರ್ಣಾಂಕ                   d0,  // dilation dimension 0
        ಪೂರ್ಣಾಂಕ                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, s0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, s1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 2, p0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 3, p1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 4, d0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 5, d1);

    ಫಲಿತಾಂಶ->op = GGML_OP_CONV_2D;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_3d_direct

ರಚನೆ ggml_tensor * ggml_conv_3d_ನೇರ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ಪೂರ್ಣಾಂಕ                   s2,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   p2,
        ಪೂರ್ಣಾಂಕ                   d0,
        ಪೂರ್ಣಾಂಕ                   d1,
        ಪೂರ್ಣಾಂಕ                   d2,
        ಪೂರ್ಣಾಂಕ                   c,
        ಪೂರ್ಣಾಂಕ                   n,
        ಪೂರ್ಣಾಂಕ                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_ಗಾತ್ರ(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0,  s0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1,  s1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 2,  s2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 3,  p0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 4,  p1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 5,  p2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 6,  d0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 7,  d1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 8,  d2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 9,  c);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 10, n);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 11, oc);

    ಫಲಿತಾಂಶ->op = GGML_OP_CONV_3D;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_conv_transpose_2d_p0

ಸ್ಥಿರ int64_t ggml_calc_conv_transpose_output_ಗಾತ್ರ(int64_t ins, int64_t ks, ಪೂರ್ಣಾಂಕ s, ಪೂರ್ಣಾಂಕ p) {
    ಹಿಂತಿರುಗಿ (ins - 1) * s - 2 * p + ks;
}

ರಚನೆ ggml_tensor * ggml_conv_transpose_2d_p0(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ಪೂರ್ಣಾಂಕ                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_ಗಾತ್ರ(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_ಗಾತ್ರ(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ರಚನೆ ggml_tensor* ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, stride);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CONV_TRANSPOSE_2D;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_pool_*

ಸ್ಥಿರ int64_t ggml_calc_pool_output_ಗಾತ್ರ(int64_t ins, ಪೂರ್ಣಾಂಕ ks, ಪೂರ್ಣಾಂಕ s, ದಶಮಾಂಶ p) {
    ಹಿಂತಿರುಗಿ (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ರಚನೆ ggml_tensor * ggml_pool_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_op_pool     op,
        ಪೂರ್ಣಾಂಕ                   k0,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   p0) {
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {
        ggml_calc_pool_output_ಗಾತ್ರ(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_POOL_1D;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_pool_2d

ರಚನೆ ggml_tensor * ggml_pool_2d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_op_pool     op,
        ಪೂರ್ಣಾಂಕ                   k0,
        ಪೂರ್ಣಾಂಕ                   k1,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ದಶಮಾಂಶ                 p0,
        ದಶಮಾಂಶ                 p1) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {
        ggml_calc_pool_output_ಗಾತ್ರ(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_ಗಾತ್ರ(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_POOL_2D;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_pool_2d_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * af,
        ಎಣಿಕೆ ggml_op_pool     op,
        ಪೂರ್ಣಾಂಕ                   k0,
        ಪೂರ್ಣಾಂಕ                   k1,
        ಪೂರ್ಣಾಂಕ                   s0,
        ಪೂರ್ಣಾಂಕ                   s1,
        ದಶಮಾಂಶ                 p0,
        ದಶಮಾಂಶ                 p1) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ;
    ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_POOL_2_D_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = af;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_upscale / ggml_interpolate

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_interpolate_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              ಬಹುಲಕ) {
    GGML_ASSERT((ಬಹುಲಕ & 0xFF) < GGML_SCALE_ಬಹುಲಕ_ಎಣಿಕೆ);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(ಬಹುಲಕ & GGML_SCALE_FLAG_ANTIALIAS) || (ಬಹುಲಕ & 0xFF) == GGML_SCALE_ಬಹುಲಕ_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t)ಬಹುಲಕ);

    ಫಲಿತಾಂಶ->op     = GGML_OP_UPSCALE;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_upscale(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   scale_ಗುಣಕ,
        ಎಣಿಕೆ ggml_scale_ಬಹುಲಕ  ಬಹುಲಕ) {
    GGML_ASSERT(scale_ಗುಣಕ > 1);
    ಹಿಂತಿರುಗಿ ggml_interpolate_impl(ctx, a, a->ne[0] * scale_ಗುಣಕ, a->ne[1] * scale_ಗುಣಕ, a->ne[2], a->ne[3], ಬಹುಲಕ);
}

ರಚನೆ ggml_tensor * ggml_upscale_ext(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   ne0,
        ಪೂರ್ಣಾಂಕ                   ne1,
        ಪೂರ್ಣಾಂಕ                   ne2,
        ಪೂರ್ಣಾಂಕ                   ne3,
        ಎಣಿಕೆ ggml_scale_ಬಹುಲಕ  ಬಹುಲಕ) {
    ಹಿಂತಿರುಗಿ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, ಬಹುಲಕ);
}

ರಚನೆ ggml_tensor * ggml_interpolate(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              ಬಹುಲಕ) {
    ಹಿಂತಿರುಗಿ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, ಬಹುಲಕ);
}

// ggml_pad

ರಚನೆ ggml_tensor * ggml_pad(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   p2,
        ಪೂರ್ಣಾಂಕ                   p3) {
    ಹಿಂತಿರುಗಿ ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ರಚನೆ ggml_tensor * ggml_pad_circular(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1,
        ಪೂರ್ಣಾಂಕ                   p2,
        ಪೂರ್ಣಾಂಕ                   p3) {
    ಹಿಂತಿರುಗಿ ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ರಚನೆ ggml_tensor * ggml_pad_ext(
            ರಚನೆ ggml_context * ctx,
            ರಚನೆ ggml_tensor  * a,
            ಪೂರ್ಣಾಂಕ                  lp0,
            ಪೂರ್ಣಾಂಕ                  rp0,
            ಪೂರ್ಣಾಂಕ                  lp1,
            ಪೂರ್ಣಾಂಕ                  rp1,
            ಪೂರ್ಣಾಂಕ                  lp2,
            ಪೂರ್ಣಾಂಕ                  rp2,
            ಪೂರ್ಣಾಂಕ                  lp3,
            ಪೂರ್ಣಾಂಕ                  rp3
            ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, lp0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, rp0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 2, lp1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 3, rp1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 4, lp2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 5, rp2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 6, lp3);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 7, rp3);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 8, 0); // not circular by default


    ಫಲಿತಾಂಶ->op     = GGML_OP_PAD;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_pad_ext_circular

ರಚನೆ ggml_tensor * ggml_pad_ext_circular(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                  lp0,
        ಪೂರ್ಣಾಂಕ                  rp0,
        ಪೂರ್ಣಾಂಕ                  lp1,
        ಪೂರ್ಣಾಂಕ                  rp1,
        ಪೂರ್ಣಾಂಕ                  lp2,
        ಪೂರ್ಣಾಂಕ                  rp2,
        ಪೂರ್ಣಾಂಕ                  lp3,
        ಪೂರ್ಣಾಂಕ                  rp3
        ) {
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 8, 1); // circular
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_pad_reflect_1d

ರಚನೆ ggml_tensor * ggml_pad_reflect_1d(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   p0,
        ಪೂರ್ಣಾಂಕ                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_PAD_REFLECT_1D;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_roll

ರಚನೆ ggml_tensor * ggml_roll(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   shift0,
        ಪೂರ್ಣಾಂಕ                   shift1,
        ಪೂರ್ಣಾಂಕ                   shift2,
        ಪೂರ್ಣಾಂಕ                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_ಗಾತ್ರ(a->type));
    GGML_ASSERT(ನಿರಪೇಕ್ಷ(shift0) < a->ne[0]);
    GGML_ASSERT(ನಿರಪೇಕ್ಷ(shift1) < a->ne[1]);
    GGML_ASSERT(ನಿರಪೇಕ್ಷ(shift2) < a->ne[2]);
    GGML_ASSERT(ನಿರಪೇಕ್ಷ(shift3) < a->ne[3]);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, shift0);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, shift1);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 2, shift2);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 3, shift3);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ROLL;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_timestep_embedding

ರಚನೆ ggml_tensor * ggml_timestep_embedding(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * timesteps,
        ಪೂರ್ಣಾಂಕ                   dim,
        ಪೂರ್ಣಾಂಕ                   max_period) {

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, dim);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, max_period);

    ಫಲಿತಾಂಶ->op     = GGML_OP_TIMESTEP_EMBEDDING;
    ಫಲಿತಾಂಶ->src[0] = timesteps;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_tri

ರಚನೆ ggml_tensor * ggml_tri(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ಎಣಿಕೆ ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, type);

    ಫಲಿತಾಂಶ->op = GGML_OP_TRI;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_fill

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_fill_impl(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ದಶಮಾಂಶ                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, c);

    ಫಲಿತಾಂಶ->op = GGML_OP_FILL;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_fill(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ದಶಮಾಂಶ                 c) {
    ಹಿಂತಿರುಗಿ ggml_fill_impl(ctx, a, c, false);
}

ರಚನೆ ggml_tensor * ggml_fill_inplace(
    ರಚನೆ ggml_context * ctx,
    ರಚನೆ ggml_tensor  * a,
    ದಶಮಾಂಶ                 c) {
    ಹಿಂತಿರುಗಿ ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ರಚನೆ ggml_tensor * ggml_argsort(
        ರಚನೆ ggml_context  * ctx,
        ರಚನೆ ggml_tensor   * a,
        ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ   ಕ್ರಮ) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) ಕ್ರಮ);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ARGSORT;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_argsort_top_k

ರಚನೆ ggml_tensor * ggml_argsort_ತುದಿ_k(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_argsort(ctx, a, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC);

    ಫಲಿತಾಂಶ = ggml_view_4d(ctx, ಫಲಿತಾಂಶ,
                k, ಫಲಿತಾಂಶ->ne[1], ಫಲಿತಾಂಶ->ne[2], ಫಲಿತಾಂಶ->ne[3],
                   ಫಲಿತಾಂಶ->nb[1], ಫಲಿತಾಂಶ->nb[2], ಫಲಿತಾಂಶ->nb[3],
                0);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_top_k

ರಚನೆ ggml_tensor * ggml_ತುದಿ_k(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ತುದಿ_K;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_arange

ರಚನೆ ggml_tensor * ggml_arange(
        ರಚನೆ ggml_context * ctx,
        ದಶಮಾಂಶ                 ಆರಂಭ,
        ದಶಮಾಂಶ                 stop,
        ದಶಮಾಂಶ                 ಹೆಜ್ಜೆ) {
    GGML_ASSERT(stop > ಆರಂಭ);

    ಸ್ಥಿರಾಂಕ int64_t ಹೆಜ್ಜೆಗಳು = (int64_t) ceilf((stop - ಆರಂಭ) / ಹೆಜ್ಜೆ);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ಹೆಜ್ಜೆಗಳು);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, ಆರಂಭ);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 1, stop);
    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 2, ಹೆಜ್ಜೆ);

    ಫಲಿತಾಂಶ->op = GGML_OP_ARANGE;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_flash_attn_ext

ರಚನೆ ggml_tensor * ggml_flash_attn_ext(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * q,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ,
        ದಶಮಾಂಶ                 scale,
        ದಶಮಾಂಶ                 max_bias,
        ದಶಮಾಂಶ                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    ಆದರೆ (ಮುಖವಾಡ) {
        GGML_ASSERT(ಮುಖವಾಡ->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(ಮುಖವಾಡ));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % ಮುಖವಾಡ->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % ಮುಖವಾಡ->ne[3] == 0);
    }

    ಆದರೆ (max_bias > 0.0f) {
        GGML_ASSERT(ಮುಖವಾಡ);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ದಶಮಾಂಶ params[] = { scale, max_bias, logit_softcap };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_FLASH_ATTN_EXT;
    ಫಲಿತಾಂಶ->src[0] = q;
    ಫಲಿತಾಂಶ->src[1] = k;
    ಫಲಿತಾಂಶ->src[2] = v;
    ಫಲಿತಾಂಶ->src[3] = ಮುಖವಾಡ;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}


ಶೂನ್ಯ ggml_flash_attn_ext_ಗಣ_prec(
        ರಚನೆ ggml_tensor * a,
        ಎಣಿಕೆ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ಸ್ಥಿರಾಂಕ int32_t prec_i32 = (int32_t) prec;

    ggml_ಗಣ_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

ಎಣಿಕೆ ggml_prec ggml_flash_attn_ext_get_prec(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ಸ್ಥಿರಾಂಕ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ಹಿಂತಿರುಗಿ (ಎಣಿಕೆ ggml_prec) prec_i32;
}

ಶೂನ್ಯ ggml_flash_attn_ext_ಗಣ_n_kv_max(
        ರಚನೆ ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_ಗಣ_op_params_i32(a, 4, n_kv_max);
}

ಶೂನ್ಯ ggml_flash_attn_ext_ಕೂಡಿಸು_sinks(
        ರಚನೆ ggml_tensor * a,
        ರಚನೆ ggml_tensor * sinks) {
    ಆದರೆ (!sinks) {
        a->src[4] = ಖಾಲಿ;
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ಖಾಲಿ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ರಚನೆ ggml_tensor * ggml_flash_attn_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * q,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ಸ್ಥಿರಾಂಕ int64_t     D = q->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t     N = q->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t     M = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t   ne2 = q->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t   ne3 = q->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t kvne2 = k->ne[2];

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
    ಸ್ಥಿರಾಂಕ int64_t elem_q = ggml_nelements(q);
    ಸ್ಥಿರಾಂಕ int64_t elem_k = ggml_nelements(k);
    ಸ್ಥಿರಾಂಕ int64_t elem_v = ggml_nelements(v);

    ಎಣಿಕೆ ggml_type ಫಲಿತಾಂಶ_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_ಗಾತ್ರ(ಫಲಿತಾಂಶ_type) == 1);
    ಸ್ಥಿರಾಂಕ size_t tsize = ggml_type_ಗಾತ್ರ(ಫಲಿತಾಂಶ_type);

    ಸ್ಥಿರಾಂಕ size_t offs_q = 0;
    ಸ್ಥಿರಾಂಕ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ಸ್ಥಿರಾಂಕ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ಸ್ಥಿರಾಂಕ size_t ಕೊನೆ    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ಸ್ಥಿರಾಂಕ size_t nelements = (ಕೊನೆ + tsize - 1)/tsize;

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &masked_i, ಗಾತ್ರ(masked_i));

    ಫಲಿತಾಂಶ->op     = GGML_OP_FLASH_ATTN_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = q;
    ಫಲಿತಾಂಶ->src[1] = k;
    ಫಲಿತಾಂಶ->src[2] = v;
    ಫಲಿತಾಂಶ->src[3] = d;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_ssm_conv

ರಚನೆ ggml_tensor * ggml_ssm_conv(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * sx,
        ರಚನೆ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(c));

    ಸ್ಥಿರಾಂಕ int64_t d_conv  = c->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t d_inner = c->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ಸ್ಥಿರಾಂಕ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SSM_CONV;
    ಫಲಿತಾಂಶ->src[0] = sx;
    ಫಲಿತಾಂಶ->src[1] = c;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_ssm_scan

ರಚನೆ ggml_tensor * ggml_ssm_ಪರಿಶೀಲನೆ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * s,
        ರಚನೆ ggml_tensor  * x,
        ರಚನೆ ggml_tensor  * dt,
        ರಚನೆ ggml_tensor  * A,
        ರಚನೆ ggml_tensor  * B,
        ರಚನೆ ggml_tensor  * C,
        ರಚನೆ ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_ಗಾತ್ರ(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_ಗಾತ್ರ(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_ಗಾತ್ರ(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ಸ್ಥಿರಾಂಕ int64_t d_state      = s->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_dim     = x->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ       = x->ne[1];
        ಸ್ಥಿರಾಂಕ int64_t n_seq_ಟೋಕನುಗಳು = x->ne[2];
        ಸ್ಥಿರಾಂಕ int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_ತಲೆ);
        GGML_ASSERT(dt->ne[1] == n_seq_ಟೋಕನುಗಳು);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == ತಲೆ_dim);
        GGML_ASSERT(s->ne[2] == n_ತಲೆ);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_ಟೋಕನುಗಳು);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_ಸದಿಶ(ids));
        GGML_ASSERT(A->ne[1] == n_ತಲೆ);
        GGML_ASSERT(ggml_is_ಮಾತೃಕೆ(A));

        ಆದರೆ (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    ಫಲಿತಾಂಶ->op   = GGML_OP_SSM_ಪರಿಶೀಲನೆ;
    ಫಲಿತಾಂಶ->src[0] = s;
    ಫಲಿತಾಂಶ->src[1] = x;
    ಫಲಿತಾಂಶ->src[2] = dt;
    ಫಲಿತಾಂಶ->src[3] = A;
    ಫಲಿತಾಂಶ->src[4] = B;
    ಫಲಿತಾಂಶ->src[5] = C;
    ಫಲಿತಾಂಶ->src[6] = ids;

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) K);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_win_part

ರಚನೆ ggml_tensor * ggml_win_part(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ px = (w - a->ne[1]%w)%w;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ py = (w - a->ne[2]%w)%w;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ npx = (px + a->ne[1])/w;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ npy = (py + a->ne[2])/w;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np  = npx*npy;

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[0], w, w, np, };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_WIN_PART;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_win_unpart

ರಚನೆ ggml_tensor * ggml_win_unpart(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   w0,
        ಪೂರ್ಣಾಂಕ                   h0,
        ಪೂರ್ಣಾಂಕ                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_WIN_UNPART;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_get_rel_pos

ರಚನೆ ggml_tensor * ggml_get_rel_pos(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಪೂರ್ಣಾಂಕ                   qh,
        ಪೂರ್ಣಾಂಕ                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GET_REL_POS;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_add_rel_pos

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_rel_pos_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * pw,
        ರಚನೆ ggml_tensor  * ph,
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

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, inplace ? 1 : 0);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ಕೂಡಿಸು_REL_POS;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = pw;
    ಫಲಿತಾಂಶ->src[2] = ph;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_rel_pos(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * pw,
        ರಚನೆ ggml_tensor  * ph) {
    ಹಿಂತಿರುಗಿ ggml_ಕೂಡಿಸು_rel_pos_impl(ctx, a, pw, ph, false);
}

ರಚನೆ ggml_tensor * ggml_ಕೂಡಿಸು_rel_pos_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * pw,
        ರಚನೆ ggml_tensor  * ph) {
    ಹಿಂತಿರುಗಿ ggml_ಕೂಡಿಸು_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ರಚನೆ ggml_tensor * ggml_rwkv_wkv6(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * r,
        ರಚನೆ ggml_tensor  * tf,
        ರಚನೆ ggml_tensor  * td,
        ರಚನೆ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಸ್ಥಿರಾಂಕ int64_t S = k->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = k->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { S * H, n_ಟೋಕನುಗಳು + S * n_seqs, 1, 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RWKV_WKV6;
    ಫಲಿತಾಂಶ->src[0] = k;
    ಫಲಿತಾಂಶ->src[1] = v;
    ಫಲಿತಾಂಶ->src[2] = r;
    ಫಲಿತಾಂಶ->src[3] = tf;
    ಫಲಿತಾಂಶ->src[4] = td;
    ಫಲಿತಾಂಶ->src[5] = state;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_gated_linear_attn

ರಚನೆ ggml_tensor * ggml_gated_ರೇಖೀಯ_attn(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * q,
        ರಚನೆ ggml_tensor  * g,
        ರಚನೆ ggml_tensor  * state,
        ದಶಮಾಂಶ scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಸ್ಥಿರಾಂಕ int64_t S = k->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = k->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { S * H, n_ಟೋಕನುಗಳು + S * n_seqs, 1, 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, scale);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GATED_ರೇಖೀಯ_ATTN;
    ಫಲಿತಾಂಶ->src[0] = k;
    ಫಲಿತಾಂಶ->src[1] = v;
    ಫಲಿತಾಂಶ->src[2] = q;
    ಫಲಿತಾಂಶ->src[3] = g;
    ಫಲಿತಾಂಶ->src[4] = state;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_rwkv_wkv7

ರಚನೆ ggml_tensor * ggml_rwkv_wkv7(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * r,
        ರಚನೆ ggml_tensor  * w,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಸ್ಥಿರಾಂಕ int64_t S = k->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = k->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { S * H, n_ಟೋಕನುಗಳು + S * n_seqs, 1, 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op     = GGML_OP_RWKV_WKV7;
    ಫಲಿತಾಂಶ->src[0] = r;
    ಫಲಿತಾಂಶ->src[1] = w;
    ಫಲಿತಾಂಶ->src[2] = k;
    ಫಲಿತಾಂಶ->src[3] = v;
    ಫಲಿತಾಂಶ->src[4] = a;
    ಫಲಿತಾಂಶ->src[5] = b;
    ಫಲಿತಾಂಶ->src[6] = state;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_unary

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_unary_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(a));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) op);

    ಫಲಿತಾಂಶ->op     = GGML_OP_UNARY;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_unary(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_unary_op    op) {
    ಹಿಂತಿರುಗಿ ggml_unary_impl(ctx, a, op, false);
}

ರಚನೆ ggml_tensor * ggml_unary_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ಎಣಿಕೆ ggml_unary_op    op) {
    ಹಿಂತಿರುಗಿ ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_map_custom1_impl(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ಸ್ಥಿರಾಂಕ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ರಚನೆ ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_MAP_CUSTOM1;
    ಫಲಿತಾಂಶ->src[0] = a;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_map_custom1(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ಸ್ಥಿರಾಂಕ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ರಚನೆ ggml_tensor * ggml_map_custom1_inplace(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ಸ್ಥಿರಾಂಕ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_map_custom2_impl(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ಸ್ಥಿರಾಂಕ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ರಚನೆ ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_MAP_CUSTOM2;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_map_custom2(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ಸ್ಥಿರಾಂಕ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ರಚನೆ ggml_tensor * ggml_map_custom2_inplace(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ಸ್ಥಿರಾಂಕ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_map_custom3_impl(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ರಚನೆ ggml_tensor       * c,
        ಸ್ಥಿರಾಂಕ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ರಚನೆ ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op     = GGML_OP_MAP_CUSTOM3;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಫಲಿತಾಂಶ->src[2] = c;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_map_custom3(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ರಚನೆ ggml_tensor       * c,
        ಸ್ಥಿರಾಂಕ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ರಚನೆ ggml_tensor * ggml_map_custom3_inplace(
        ರಚನೆ ggml_context      * ctx,
        ರಚನೆ ggml_tensor       * a,
        ರಚನೆ ggml_tensor       * b,
        ರಚನೆ ggml_tensor       * c,
        ಸ್ಥಿರಾಂಕ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಂಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಹಿಂತಿರುಗಿ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ರಚನೆ ggml_tensor * ggml_custom_4d(
        ರಚನೆ ggml_context * ctx,
        ಎಣಿಕೆ ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ರಚನೆ ggml_tensor ** ವಾದಗಳು_2,
        ಪೂರ್ಣಾಂಕ                   n_ವಾದಗಳು,
        ggml_custom_op_t      fun,
        ಪೂರ್ಣಾಂಕ                   n_tasks,
        ಶೂನ್ಯ                * userdata) {

    GGML_ASSERT(n_ವಾದಗಳು < GGML_MAX_SRC);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ರಚನೆ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op = GGML_OP_CUSTOM;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ವಾದಗಳು; i++) {
        ಫಲಿತಾಂಶ->src[i] = ವಾದಗಳು_2[i];
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_custom_inplace(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor ** ವಾದಗಳು_2,
        ಪೂರ್ಣಾಂಕ                   n_ವಾದಗಳು,
        ggml_custom_op_t      fun,
        ಪೂರ್ಣಾಂಕ                   n_tasks,
        ಶೂನ್ಯ                * userdata) {

    GGML_ASSERT(n_ವಾದಗಳು < GGML_MAX_SRC - 1);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);

    ರಚನೆ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_ಗಣ_op_params(ಫಲಿತಾಂಶ, &params, ಗಾತ್ರ(params));

    ಫಲಿತಾಂಶ->op = GGML_OP_CUSTOM;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ವಾದಗಳು; i++) {
        ಫಲಿತಾಂಶ->src[i + 1] = ವಾದಗಳು_2[i];
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}
// ggml_cross_entropy_loss

ರಚನೆ ggml_tensor * ggml_cross_entropy_loss(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, a->type, 1);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_cross_entropy_loss_back

ರಚನೆ ggml_tensor * ggml_cross_entropy_loss_ಹಿಂಭಾಗ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        ರಚನೆ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_dup_tensor(ctx, b);

    ಫಲಿತಾಂಶ->op     = GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;
    ಫಲಿತಾಂಶ->src[2] = c;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// opt_step_adamw

ರಚನೆ ggml_tensor * ggml_opt_ಹೆಜ್ಜೆ_adamw(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * grad,
        ರಚನೆ ggml_tensor  * m,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = grad;
    ಫಲಿತಾಂಶ->src[2] = m;
    ಫಲಿತಾಂಶ->src[3] = v;
    ಫಲಿತಾಂಶ->src[4] = adamw_params;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// opt_step_sgd

ರಚನೆ ggml_tensor * ggml_opt_ಹೆಜ್ಜೆ_sgd(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * grad,
        ರಚನೆ ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_view_tensor(ctx, a);

    ಫಲಿತಾಂಶ->op     = GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = grad;
    ಫಲಿತಾಂಶ->src[2] = params;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// solve_tri

ರಚನೆ ggml_tensor * ggml_solve_tri(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * a,
        ರಚನೆ ggml_tensor  * b,
        bool                  ಎಡ_2,
        bool                  ಅಧೋ,
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

    GGML_ASSERT(ಅಧೋ && ಎಡ_2 && !uni); // TODO: support other variants

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_SOLVE_TRI;
    ಫಲಿತಾಂಶ->src[0] = a;
    ಫಲಿತಾಂಶ->src[1] = b;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_gated_delta_net

ರಚನೆ ggml_tensor * ggml_gated_delta_net(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * q,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * v,
        ರಚನೆ ggml_tensor  * g,
        ರಚನೆ ggml_tensor  * beta,
        ರಚನೆ ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(q));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(k));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t S_v      = v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H        = v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ಸ್ಥಿರಾಂಕ int64_t state_ಅಡ್ಡಸಾಲುಗಳು = K * S_v * n_seqs;
    ಸ್ಥಿರಾಂಕ int64_t ne[4] = { S_v * H, n_ಟೋಕನುಗಳು * n_seqs + state_ಅಡ್ಡಸಾಲುಗಳು, 1, 1 };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 0, (int32_t) K);

    ಫಲಿತಾಂಶ->op     = GGML_OP_GATED_DELTA_NET;
    ಫಲಿತಾಂಶ->src[0] = q;
    ಫಲಿತಾಂಶ->src[1] = k;
    ಫಲಿತಾಂಶ->src[2] = v;
    ಫಲಿತಾಂಶ->src[3] = g;
    ಫಲಿತಾಂಶ->src[4] = beta;
    ಫಲಿತಾಂಶ->src[5] = state;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_lightning_indexer

ರಚನೆ ggml_tensor * ggml_lightning_indexer(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * q,
        ರಚನೆ ggml_tensor  * k,
        ರಚನೆ ggml_tensor  * weights,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    ಮುಖವಾಡ->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   ಮುಖವಾಡ->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   ಮುಖವಾಡ->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   ಮುಖವಾಡ->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % ಮುಖವಾಡ->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಫಲಿತಾಂಶ->op   = GGML_OP_LIGHTNING_INDEXER;
    ಫಲಿತಾಂಶ->src[0] = q;
    ಫಲಿತಾಂಶ->src[1] = k;
    ಫಲಿತಾಂಶ->src[2] = weights;
    ಫಲಿತಾಂಶ->src[3] = ಮುಖವಾಡ;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_dsv4_hc_comb

ರಚನೆ ggml_tensor * ggml_dsv4_hc_comb(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * mixes,
        ರಚನೆ ggml_tensor  * scale,
        ರಚನೆ ggml_tensor  * ತಳ,
        ದಶಮಾಂಶ                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ತಳ->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ಸ್ಥಿರಾಂಕ int64_t hc_mix_dim = mixes->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು   = mixes->ne[1];

    int64_t hc = 0;
    ಗಾಗಿ (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        ಆದರೆ ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ನಿಲ್ಲಿಸು;
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
    GGML_ASSERT(ತಳ->ne[0] == hc_mix_dim);
    GGML_ASSERT(ತಳ->ne[1] == 1);
    GGML_ASSERT(ತಳ->ne[2] == 1);
    GGML_ASSERT(ತಳ->ne[3] == 1);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_ಟೋಕನುಗಳು);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, eps);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, n_iter);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DSV4_HC_COMB;
    ಫಲಿತಾಂಶ->src[0] = mixes;
    ಫಲಿತಾಂಶ->src[1] = scale;
    ಫಲಿತಾಂಶ->src[2] = ತಳ;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_dsv4_hc_pre

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_dsv4_hc_pre_impl(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * x,
        ರಚನೆ ggml_tensor  * weights,
        ದಶಮಾಂಶ                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t hc       = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    ಆದರೆ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ಟೋಕನುಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ಟೋಕನುಗಳು);

    ggml_ಗಣ_op_params_f32(ಫಲಿತಾಂಶ, 0, scale);
    ggml_ಗಣ_op_params_i32(ಫಲಿತಾಂಶ, 1, gated ? 1 : 0);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DSV4_HC_PRE;
    ಫಲಿತಾಂಶ->src[0] = x;
    ಫಲಿತಾಂಶ->src[1] = weights;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_dsv4_hc_pre(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * x,
        ರಚನೆ ggml_tensor  * weights) {
    ಹಿಂತಿರುಗಿ ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ರಚನೆ ggml_tensor * ggml_dsv4_hc_pre_gated(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * x,
        ರಚನೆ ggml_tensor  * gate,
        ದಶಮಾಂಶ                 scale) {
    ಹಿಂತಿರುಗಿ ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ರಚನೆ ggml_tensor * ggml_dsv4_hc_post(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * x,
        ರಚನೆ ggml_tensor  * residual,
        ರಚನೆ ggml_tensor  * post,
        ರಚನೆ ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ಟೋಕನುಗಳು);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ಟೋಕನುಗಳು);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    ಆದರೆ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ಟೋಕನುಗಳು);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_ಟೋಕನುಗಳು);

    ಫಲಿತಾಂಶ->op     = GGML_OP_DSV4_HC_POST;
    ಫಲಿತಾಂಶ->src[0] = x;
    ಫಲಿತಾಂಶ->src[1] = residual;
    ಫಲಿತಾಂಶ->src[2] = post;
    ಫಲಿತಾಂಶ->src[3] = comb;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// ggml_top_k_mask

ರಚನೆ ggml_tensor * ggml_ತುದಿ_k_ಮುಖವಾಡ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_tensor  * ಮುಖವಾಡ,
        ರಚನೆ ggml_tensor  * ತುದಿ_k,
        ರಚನೆ ggml_tensor  * score,
        ರಚನೆ ggml_tensor  * ಖಂಡಗಳು,
        ರಚನೆ ggml_tensor  * extra) {
    GGML_ASSERT(ಮುಖವಾಡ->type == GGML_TYPE_F16 || ಮುಖವಾಡ->type == GGML_TYPE_F32);
    GGML_ASSERT(ಮುಖವಾಡ->nb[0] == ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type));
    GGML_ASSERT(ತುದಿ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(ತುದಿ_k->ne[1] % ಮುಖವಾಡ->ne[1] == 0 && ತುದಿ_k->ne[2] % ಮುಖವಾಡ->ne[2] == 0 && ತುದಿ_k->ne[3] % ಮುಖವಾಡ->ne[3] == 0);

    ಆದರೆ (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == ತುದಿ_k->ne[1] && score->ne[2] == ತುದಿ_k->ne[2] && score->ne[3] == ತುದಿ_k->ne[3]);
    }

    ಆದರೆ (ಖಂಡಗಳು) {
        GGML_ASSERT(ಖಂಡಗಳು->type == GGML_TYPE_I32);
        GGML_ASSERT(ಖಂಡಗಳು->ne[2] == 1 && ಖಂಡಗಳು->ne[3] == 1);
        GGML_ASSERT(!score || ಖಂಡಗಳು->ne[1] == score->ne[0]);
    }

    ಆದರೆ (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == ತುದಿ_k->ne[1] && extra->ne[2] == ತುದಿ_k->ne[2] && extra->ne[3] == ತುದಿ_k->ne[3]);
    }

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, ಮುಖವಾಡ->type, ಮುಖವಾಡ->ne[0], ತುದಿ_k->ne[1], ತುದಿ_k->ne[2], ತುದಿ_k->ne[3]);

    ಫಲಿತಾಂಶ->op     = GGML_OP_ತುದಿ_K_ಮುಖವಾಡ;
    ಫಲಿತಾಂಶ->src[0] = ಮುಖವಾಡ;
    ಫಲಿತಾಂಶ->src[1] = ತುದಿ_k;
    ಫಲಿತಾಂಶ->src[2] = score;
    ಫಲಿತಾಂಶ->src[3] = ಖಂಡಗಳು;
    ಫಲಿತಾಂಶ->src[4] = extra;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

////////////////////////////////////////////////////////////////////////////////

ರಚನೆ ggml_hash_ಗಣ ggml_hash_ಗಣ_new(size_t ಗಾತ್ರ_2) {
    ಗಾತ್ರ_2 = ggml_hash_ಗಾತ್ರ(ಗಾತ್ರ_2);
    ರಚನೆ ggml_hash_ಗಣ ಫಲಿತಾಂಶ;
    ಫಲಿತಾಂಶ.ಗಾತ್ರ_2 = ಗಾತ್ರ_2;
    ಫಲಿತಾಂಶ.ಕೀಲಿಗಳು = GGML_MALLOC(ಗಾತ್ರ(ರಚನೆ ggml_tensor *) * ಗಾತ್ರ_2);
    ಫಲಿತಾಂಶ.ಬಳಸಿದ = GGML_CALLOC(ggml_bitset_ಗಾತ್ರ(ಗಾತ್ರ_2), ಗಾತ್ರ(ggml_bitset_t));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_hash_ಗಣ_reset(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ) {
    memset(hash_ಗಣ->ಬಳಸಿದ, 0, ಗಾತ್ರ(ggml_bitset_t) * ggml_bitset_ಗಾತ್ರ(hash_ಗಣ->ಗಾತ್ರ_2));
}

ಶೂನ್ಯ ggml_hash_ಗಣ_free(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ) {
    GGML_FREE(hash_ಗಣ->ಬಳಸಿದ);
    GGML_FREE(hash_ಗಣ->ಕೀಲಿಗಳು);
}

size_t ggml_hash_ಗಾತ್ರ(size_t min_sz) {
    // next primes after powers of two
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t ಅವಿಭಾಜ್ಯಗಳು[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t n_ಅವಿಭಾಜ್ಯಗಳು = ಗಾತ್ರ(ಅವಿಭಾಜ್ಯಗಳು)/ಗಾತ್ರ(ಅವಿಭಾಜ್ಯಗಳು[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_ಅವಿಭಾಜ್ಯಗಳು;
    ಆಗ (l < r) {
        size_t m = (l + r)/2;
        ಆದರೆ (ಅವಿಭಾಜ್ಯಗಳು[m] < min_sz) {
            l = m + 1;
        } ಇಲ್ಲದಿದ್ದರೆ {
            r = m;
        }
    }
    size_t sz = l < n_ಅವಿಭಾಜ್ಯಗಳು ? ಅವಿಭಾಜ್ಯಗಳು[l] : min_sz | 1;
    ಹಿಂತಿರುಗಿ sz;
}

ರಚನೆ hash_map {
    ರಚನೆ ggml_hash_ಗಣ ಗಣ;
    ರಚನೆ ggml_tensor ** vals;
};

ಸ್ಥಿರ ರಚನೆ hash_map * ggml_new_hash_map(size_t ಗಾತ್ರ_2) {
    ರಚನೆ hash_map * ಫಲಿತಾಂಶ = GGML_MALLOC(ಗಾತ್ರ(ರಚನೆ hash_map));
    ಫಲಿತಾಂಶ->ಗಣ = ggml_hash_ಗಣ_new(ಗಾತ್ರ_2);
    ಫಲಿತಾಂಶ->vals = GGML_CALLOC(ಫಲಿತಾಂಶ->ಗಣ.ಗಾತ್ರ_2, ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hash_map_free(ರಚನೆ hash_map * map) {
    ggml_hash_ಗಣ_free(&map->ಗಣ);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

ಸ್ಥಿರ ಶೂನ್ಯ ggml_ಕೂಡಿಸು_or_ಗಣ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_cgraph  * cgraph,
        size_t                isrc,
        ರಚನೆ ggml_tensor  * tensor) {
    ರಚನೆ ggml_tensor * src = cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಆದರೆ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_ಕೂಡಿಸು_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_acc_or_ಗಣ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_cgraph  * cgraph,
        size_t                isrc,
        ರಚನೆ ggml_tensor  * tensor,
        ಸ್ಥಿರಾಂಕ  size_t         nb1,
        ಸ್ಥಿರಾಂಕ  size_t         nb2,
        ಸ್ಥಿರಾಂಕ  size_t         nb3,
        ಸ್ಥಿರಾಂಕ  size_t         ಸ್ಥಾನಾಂತರ) {
    ರಚನೆ ggml_tensor * src = cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಆದರೆ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, cgraph->grad_accs[isrc]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_add1_or_ಗಣ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_cgraph  * cgraph,
        size_t                isrc,
        ರಚನೆ ggml_tensor  * tensor) {
    ರಚನೆ ggml_tensor * src = cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಆದರೆ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sub_or_ಗಣ(
        ರಚನೆ ggml_context * ctx,
        ರಚನೆ ggml_cgraph  * cgraph,
        size_t                isrc,
        ರಚನೆ ggml_tensor  * tensor) {
    ರಚನೆ ggml_tensor * src = cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಆದರೆ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_backward(
        ರಚನೆ ggml_context * ctx, ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i, ಸ್ಥಿರಾಂಕ bool * grads_needed) {
    ರಚನೆ ggml_tensor * tensor = cgraph->nodes[i];
    ರಚನೆ ggml_tensor * grad   = ggml_ನಕ್ಷೆ_get_grad(cgraph, tensor);

    ಆದರೆ (!grad) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ ggml_tensor * src0 = tensor->src[0];
    ರಚನೆ ggml_tensor * src1 = tensor->src[1];
    ರಚನೆ ggml_tensor * src2 = tensor->src[2];
    ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ = &cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ;
    ಸ್ಥಿರಾಂಕ size_t isrc0 = src0 ? ggml_hash_ಕಂಡುಹಿಡಿ(hash_ಗಣ, src0) : (size_t) -1;
    ಸ್ಥಿರಾಂಕ size_t isrc1 = src1 ? ggml_hash_ಕಂಡುಹಿಡಿ(hash_ಗಣ, src1) : (size_t) -1;
    ಸ್ಥಿರಾಂಕ size_t isrc2 = src2 ? ggml_hash_ಕಂಡುಹಿಡಿ(hash_ಗಣ, src2) : (size_t) -1;
    ಸ್ಥಿರಾಂಕ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, isrc0) && grads_needed[isrc0];
    ಸ್ಥಿರಾಂಕ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, isrc1) && grads_needed[isrc1];
    ಸ್ಥಿರಾಂಕ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, isrc2) && grads_needed[isrc2];

    ಆಯ್ಕೆ (tensor->op) {
        ಸ್ಥಿತಿ GGML_OP_DUP: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
            ಆದರೆ (src1_needs_grads) {
                ರಚನೆ ggml_tensor * tmp = grad;
                ಆದರೆ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_ಹಿಂಭಾಗ(ctx, tmp, src1);
                }
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, tmp);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ADD1: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
            ಆದರೆ (src1_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, ggml_ಸರಾಸರಿ(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ACC: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
            ಆದರೆ (src1_needs_grads) {
                ಸ್ಥಿರಾಂಕ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ಸ್ಥಿರಾಂಕ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ಸ್ಥಿರಾಂಕ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = ((int32_t *) tensor->op_params)[3];

                ರಚನೆ ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ);

                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
            ಆದರೆ (src1_needs_grads) {
                ggml_sub_or_ಗಣ(ctx, cgraph, isrc1, grad);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            ಆದರೆ (src1_needs_grads) {
                ರಚನೆ ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                ಆದರೆ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_ಹಿಂಭಾಗ(ctx, tmp, src1);
                }
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, tmp);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            ಆದರೆ (src1_needs_grads) {
                ggml_sub_or_ಗಣ(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS: {
            ಆದರೆ (src0_needs_grads) {
                ggml_sub_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ: {
            ಆದರೆ (src0_needs_grads) {
                ggml_add1_or_ಗಣ(ctx, cgraph, isrc0, grad);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ: {
            ಆದರೆ (src0_needs_grads) {
                ggml_add1_or_ಗಣ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_repeat_ಹಿಂಭಾಗ(ctx, grad, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM: {
            ಆದರೆ (src0_needs_grads) {
                ದಶಮಾಂಶ eps;
                ಸ್ಮರಣೆ_ನಕಲು(&eps, tensor->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_rms_norm_ಹಿಂಭಾಗ(ctx, grad, src0, eps));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT: {
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

            ಆದರೆ (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ರಚನೆ ggml_tensor * tmp =
                    ggml_ಹೊರಹೋಗು_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                ಆದರೆ (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ಸ್ಥಿರಾಂಕ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ಸ್ಥಿರಾಂಕ size_t nb2 = tmp->nb[2] * nr2;
                    ಸ್ಥಿರಾಂಕ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_ಹಿಂಭಾಗ(ctx, tmp, src0);
                }
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, tmp);
            }
            ಆದರೆ (src1_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_ಹೊರಹೋಗು_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE: {
            ಆದರೆ (src0_needs_grads) {
                ದಶಮಾಂಶ s;
                ಸ್ಮರಣೆ_ನಕಲು(&s, tensor->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ: {
            ಸ್ಥಿರಾಂಕ size_t nb1    = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[0];
            ಸ್ಥಿರಾಂಕ size_t nb2    = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[1];
            ಸ್ಥಿರಾಂಕ size_t nb3    = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[2];
            ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[3];

            ರಚನೆ ggml_tensor * tensor_grad_view = ಖಾಲಿ;

            ಆದರೆ (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ);
            }

            ಆದರೆ (src0_needs_grads) {
                ರಚನೆ ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ, false));
            }

            ಆದರೆ (src1_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            ಆದರೆ (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            ಆದರೆ (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT: {
            // same as cpy
            ಆದರೆ (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RESHAPE: {
            ಆದರೆ (src0_needs_grads) {
                ರಚನೆ ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_VIEW: {
            ಆದರೆ (src0_needs_grads) {
                size_t ಸ್ಥಾನಾಂತರ;

                ಸ್ಮರಣೆ_ನಕಲು(&ಸ್ಥಾನಾಂತರ, tensor->op_params, ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                ಆದರೆ (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_ಗಾತ್ರ(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_ಗಾತ್ರ(src0);
                    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    ಸ್ಥಾನಾಂತರ = (ಸ್ಥಾನಾಂತರ / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_ಗಣ(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, ಸ್ಥಾನಾಂತರ);
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ: {
            ಆದರೆ (src0_needs_grads) {
                ಸ್ಥಿರಾಂಕ int32_t * axes = (ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis0 = axes[0] & 0x3;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis1 = axes[1] & 0x3;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis2 = axes[2] & 0x3;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis3 = axes[3] & 0x3;
                ಪೂರ್ಣಾಂಕ axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_ಕ್ರಮಬದಲಾವಣೆ(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ctx, grad, src1, src0));
            }
            ಆದರೆ (src1_needs_grads) {
                // noop
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF: {
            ಆದರೆ (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[0];
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_diag_ಮುಖವಾಡ_zero_impl(ctx, grad, n_past, false));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_ZERO: {
            ಆದರೆ (src0_needs_grads) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[0];
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_diag_ಮುಖವಾಡ_zero_impl(ctx, grad, n_past, false));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX: {
            ಆದರೆ (src0_needs_grads) {
                ದಶಮಾಂಶ scale    = 1.0f;
                ದಶಮಾಂಶ max_bias = 0.0f;

                ಸ್ಮರಣೆ_ನಕಲು(&scale,    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_soft_max_ext_ಹಿಂಭಾಗ(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE: {
            ಆದರೆ (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims     = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[1];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಹುಲಕ       = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((ಸ್ಥಿರಾಂಕ int32_t *) tensor->op_params)[4];
                ದಶಮಾಂಶ freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow;
                ಪೂರ್ಣಾಂಕ sections[4] = {0, 0, 0, 0};

                ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ,   (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params +  5, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&freq_scale,  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params +  6, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ,  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params +  7, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params +  8, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&beta_fast,   (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params +  9, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&beta_slow,   (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) tensor->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(&sections,                    tensor->op_params + 11, ಗಾತ್ರ(sections));

                ರಚನೆ ggml_tensor * rope_ಹಿಂಭಾಗ = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_ಹಿಂಭಾಗ(ctx, grad, src1, src2, n_dims,
                        ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow) :
                    ggml_rope_multi_ಹಿಂಭಾಗ(ctx, grad, src1, src2, n_dims, sections,
                        ಬಹುಲಕ, n_ctx_orig, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, rope_ಹಿಂಭಾಗ);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು: {
            ಆದರೆ (src1_needs_grads) {
                ಸ್ಥಿರಾಂಕ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ಸ್ಥಿರಾಂಕ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ಸ್ಥಿರಾಂಕ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ಸ್ಥಿರಾಂಕ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ಸ್ಥಿರಾಂಕ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ಸ್ಥಿರಾಂಕ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ಸ್ಥಿರಾಂಕ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, ggml_im2col_ಹಿಂಭಾಗ(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D: {
            ಆದರೆ (src0_needs_grads) {
                ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ಸ್ಥಿರಾಂಕ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ಸ್ಥಿರಾಂಕ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ಸ್ಥಿರಾಂಕ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ಸ್ಥಿರಾಂಕ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ಸ್ಥಿರಾಂಕ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ಸ್ಥಿರಾಂಕ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_pool_2d_ಹಿಂಭಾಗ(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_WIN_PART:
        ಸ್ಥಿತಿ GGML_OP_WIN_UNPART:
        ಸ್ಥಿತಿ GGML_OP_UNARY: {
            ಆಯ್ಕೆ (ggml_get_unary_op(tensor)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN: {
                    // noop
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_sub_or_ಗಣ(ctx, cgraph, isrc0, grad);
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ: {
                    // noop
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_ಹೆಜ್ಜೆ(ctx, src0), grad));
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_silu_ಹಿಂಭಾಗ(ctx, grad, src0));
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS: {
                    ಆದರೆ (src0_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS: {
            ಆದರೆ (src0_needs_grads) {
                ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_cross_entropy_loss_ಹಿಂಭಾಗ(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU: {
            ಆಯ್ಕೆ (ggml_get_glu_op(tensor)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU: {
                    ಆದರೆ (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc0, ggml_silu_ಹಿಂಭಾಗ(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    ಆದರೆ (src1_needs_grads) {
                        ggml_ಕೂಡಿಸು_or_ಗಣ(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE: {
            // noop
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ:
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ಸ್ಥಿರ size_t ggml_ಭೇಟಿ_parents_ನಕ್ಷೆ(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_tensor * ಗಂಟು, bool compute) {
    ಆದರೆ (ಗಂಟು->op != GGML_OP_NONE && compute) {
        ಗಂಟು->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ಸ್ಥಿರಾಂಕ size_t ಗಂಟು_hash_pos = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು);
    GGML_ASSERT(ಗಂಟು_hash_pos != GGML_HASHSET_ತುಂಬಿದ);

    ಆದರೆ (ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, ಗಂಟು_hash_pos)) {
        // already visited

        ಆದರೆ (compute) {
            // update the compute flag regardless
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; ++i) {
                ರಚನೆ ggml_tensor * src = ಗಂಟು->src[i];
                ಆದರೆ (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_ಭೇಟಿ_parents_ನಕ್ಷೆ(cgraph, src, true);
                }
            }
        }

        ಹಿಂತಿರುಗಿ ಗಂಟು_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[ಗಂಟು_hash_pos] = ಗಂಟು;
    ggml_bitset_ಗಣ(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, ಗಂಟು_hash_pos);
    cgraph->use_counts[ಗಂಟು_hash_pos] = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k =
            (cgraph->ಕ್ರಮ == GGML_CGRAPH_EVAL_ಕ್ರಮ_ಎಡ_ಗಮ್ಯ_ಬಲ) ? i :
            (cgraph->ಕ್ರಮ == GGML_CGRAPH_EVAL_ಕ್ರಮ_ಬಲ_ಗಮ್ಯ_ಎಡ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ರಚನೆ ggml_tensor * src = ಗಂಟು->src[k];
        ಆದರೆ (src) {
            ಸ್ಥಿರಾಂಕ size_t src_hash_pos = ggml_ಭೇಟಿ_parents_ನಕ್ಷೆ(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    ಆದರೆ (ಗಂಟು->op == GGML_OP_NONE && !(ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->ಗಾತ್ರ_2);

        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಗಂಟು->name) == 0) {
            ggml_format_name(ಗಂಟು, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ಗಂಟು;
        cgraph->n_leafs++;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(cgraph->n_nodes < cgraph->ಗಾತ್ರ_2);

        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಗಂಟು->name) == 0) {
            ggml_format_name(ಗಂಟು, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ಗಂಟು;
        cgraph->n_nodes++;
    }

    ಹಿಂತಿರುಗಿ ಗಂಟು_hash_pos;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_build_forward_impl(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_tensor * tensor, bool expand, bool compute) {
    ಆದರೆ (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_ನಕ್ಷೆ_clear(cgraph);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_old = cgraph->n_nodes;

    ggml_ಭೇಟಿ_parents_ನಕ್ಷೆ(cgraph, tensor, compute);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    ಆದರೆ (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ರಚನೆ ggml_tensor * ggml_build_forward_select(
        ರಚನೆ ggml_cgraph  * cgraph,
        ರಚನೆ ggml_tensor ** tensors,
        ಪೂರ್ಣಾಂಕ                   n_tensors,
        ಪೂರ್ಣಾಂಕ                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ಹಿಂತಿರುಗಿ tensors[idx];
}

ಶೂನ್ಯ ggml_build_forward_expand(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

ಶೂನ್ಯ ggml_build_forward_ಕ್ರಮ(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

ಶೂನ್ಯ ggml_build_backward_expand(
        ರಚನೆ ggml_context *  ctx,
        ರಚನೆ ggml_cgraph  *  cgraph,
        ರಚನೆ ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    bool * grads_needed = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2, ಗಾತ್ರ(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_nodes_f; ++i) {
            ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
            any_params = any_params || (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_nodes_f; ++i) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಆದರೆ (ಗಂಟು->type == GGML_TYPE_I32) {
            ಮುಂದುವರಿ;
        }

        bool ಗಂಟು_needs_grad = (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) || (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ಆಯ್ಕೆ (ಗಂಟು->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:      // only used for its shape
            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ: // same as IM2COL
                ignore_src[0] = true;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_UNARY: {
                ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_unary_op uop = ggml_get_unary_op(ಗಂಟು);
                // SGN and STEP unary ops are piecewise constant
                ಆದರೆ (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_ಹೆಜ್ಜೆ) {
                    ignore_src[0] = true;
                }
            } ನಿಲ್ಲಿಸು;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            ಸ್ಥಿತಿ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:      // row indices not differentiable
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ: // same as for GET_ROWS
            ಸ್ಥಿತಿ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ನಿಲ್ಲಿಸು;

            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; ++j) {
            ಆದರೆ (!ಗಂಟು->src[j] || ignore_src[j] || !grads_needed[ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು->src[j])]) {
                ಮುಂದುವರಿ;
            }
            GGML_ASSERT(ಗಂಟು->src[j]->type == GGML_TYPE_F32 || ಗಂಟು->src[j]->type == GGML_TYPE_F16);
            ಗಂಟು_needs_grad = true;
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (!ಗಂಟು_needs_grad) {
            ಮುಂದುವರಿ;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ಗಂಟು->view_src || ಗಂಟು->op == GGML_OP_CPY || ಗಂಟು->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು || ಗಂಟು->op == GGML_OP_VIEW ||
            ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || ಗಂಟು->op == GGML_OP_TRANSPOSE);

        ಸ್ಥಿರಾಂಕ size_t ihash = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು);
        GGML_ASSERT(ihash != GGML_HASHSET_ತುಂಬಿದ);
        GGML_ASSERT(ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, ihash));
        ಆದರೆ (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ಗಂಟು->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    ಬಿಡುಗಡೆ(grads_needed);
}

ಸ್ಥಿರ ಶೂನ್ಯ * incr_ptr_aligned(ಶೂನ್ಯ ** p, size_t ಗಾತ್ರ_2, size_t align) {
    ಶೂನ್ಯ * ptr = *p;
    ptr = (ಶೂನ್ಯ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) ptr + ಗಾತ್ರ_2);
    ಹಿಂತಿರುಗಿ ptr;
}

ಸ್ಥಿರ size_t ggml_ನಕ್ಷೆ_nbytes(size_t ಗಾತ್ರ_2, bool grads) {
    size_t hash_ಗಾತ್ರ = ggml_hash_ಗಾತ್ರ(ಗಾತ್ರ_2 * 2);
    ಶೂನ್ಯ * p = (ಅಕ್ಷರ *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, ಗಾತ್ರ(ರಚನೆ ggml_cgraph), 1);
    incr_ptr_aligned(&p, ಗಾತ್ರ_2 * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, ಗಾತ್ರ_2 * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(int32_t), ಗಾತ್ರ(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)); // hash keys
    ಆದರೆ (grads) {
        incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_ಗಾತ್ರ(hash_ಗಾತ್ರ) * ಗಾತ್ರ(ggml_bitset_t), ಗಾತ್ರ(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ಹಿಂತಿರುಗಿ nbytes - 1024;
}

size_t ggml_ನಕ್ಷೆ_overhead_custom(size_t ಗಾತ್ರ_2, bool grads) {
    ಹಿಂತಿರುಗಿ GGML_OBJECT_ಗಾತ್ರ + GGML_PAD(ggml_ನಕ್ಷೆ_nbytes(ಗಾತ್ರ_2, grads), GGML_MEM_ALIGN);
}

size_t ggml_ನಕ್ಷೆ_overhead(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ ggml_ನಕ್ಷೆ_overhead_custom(GGML_DEFAULT_ನಕ್ಷೆ_ಗಾತ್ರ, false);
}

ರಚನೆ ggml_cgraph * ggml_new_ನಕ್ಷೆ_custom(ರಚನೆ ggml_context * ctx, size_t ಗಾತ್ರ_2, bool grads) {
    ಸ್ಥಿರಾಂಕ size_t obj_ಗಾತ್ರ = ggml_ನಕ್ಷೆ_nbytes(ಗಾತ್ರ_2, grads);
    ರಚನೆ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_ನಕ್ಷೆ, obj_ಗಾತ್ರ);
    ರಚನೆ ggml_cgraph * cgraph = (ರಚನೆ ggml_cgraph *) ((ಅಕ್ಷರ *) ctx->mem_ಬಫರ್ + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_ಗಾತ್ರ = ggml_hash_ಗಾತ್ರ(ಗಾತ್ರ_2 * 2);

    ಶೂನ್ಯ * p = cgraph + 1;

    ರಚನೆ ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, ಗಾತ್ರ_2      * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ರಚನೆ ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, ಗಾತ್ರ_2      * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(int32_t), ಗಾತ್ರ(int32_t));
    ರಚನೆ ggml_tensor ** hash_ಕೀಲಿಗಳು_ptr  =         incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ರಚನೆ ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)) : ಖಾಲಿ;
    ರಚನೆ ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *), ಗಾತ್ರ(ರಚನೆ ggml_tensor *)) : ಖಾಲಿ;

    ggml_bitset_t * hash_ಬಳಸಿದ = incr_ptr_aligned(&p, ggml_bitset_ಗಾತ್ರ(hash_ಗಾತ್ರ) * ಗಾತ್ರ(ggml_bitset_t), ಗಾತ್ರ(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ಪ್ರತಿಜ್ಞಾ(obj_ಗಾತ್ರ == (size_t)((ಅಕ್ಷರ *)p - (ಅಕ್ಷರ *)cgraph));

    *cgraph = (ರಚನೆ ggml_cgraph) {
        /*.size         =*/ ಗಾತ್ರ_2,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_ಗಾತ್ರ, hash_ಬಳಸಿದ, hash_ಕೀಲಿಗಳು_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ಕ್ರಮ_ಎಡ_ಗಮ್ಯ_ಬಲ,
        /*.uid          =*/ 0,
    };

    ggml_hash_ಗಣ_reset(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ);
    ಆದರೆ (grads) {
        memset(cgraph->grads,     0, hash_ಗಾತ್ರ*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_ಗಾತ್ರ*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    }

    ಹಿಂತಿರುಗಿ cgraph;
}

ರಚನೆ ggml_cgraph * ggml_new_ನಕ್ಷೆ(ರಚನೆ ggml_context * ctx) {
    ಹಿಂತಿರುಗಿ ggml_new_ನಕ್ಷೆ_custom(ctx, GGML_DEFAULT_ನಕ್ಷೆ_ಗಾತ್ರ, false);
}

ರಚನೆ ggml_cgraph ggml_ನಕ್ಷೆ_view(ರಚನೆ ggml_cgraph * cgraph0, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1) {
    ರಚನೆ ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ಖಾಲಿ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ಖಾಲಿ,
        /*.leafs            =*/ ಖಾಲಿ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->ಭೇಟಿಯಾದ_hash_ಗಣ,
        /*.order            =*/ cgraph0->ಕ್ರಮ,
        /*.uid              =*/ 0
    };

    ಹಿಂತಿರುಗಿ cgraph;
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_cpy(ರಚನೆ ggml_cgraph * src, ರಚನೆ ggml_cgraph * dst) {
    GGML_ASSERT(dst->ಗಾತ್ರ_2 >= src->n_leafs);
    GGML_ASSERT(dst->ಗಾತ್ರ_2 >= src->n_nodes);
    GGML_ASSERT(dst->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2 >= src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->ಕ್ರಮ   = src->ಕ್ರಮ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    ಗಾಗಿ (size_t i = 0; i < src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2; ++i) {
        // copy all hashset keys (tensors) that are in use
        ಆದರೆ (ggml_bitset_get(src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, i)) {
            size_t new_hash_pos = ggml_hash_ಸೇರಿಸು(&dst->ಭೇಟಿಯಾದ_hash_ಗಣ, src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಕೀಲಿಗಳು[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    ಆದರೆ (dst->grads) {
        memset(dst->grads,     0, dst->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        memset(dst->grad_accs, 0, dst->ಭೇಟಿಯಾದ_hash_ಗಣ.ಗಾತ್ರ_2*ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    }
    ಆದರೆ (src->grads) {
        GGML_ASSERT(dst->grads     != ಖಾಲಿ);
        GGML_ASSERT(dst->grad_accs != ಖಾಲಿ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src->n_nodes; ++i) {
            ಸ್ಥಿರಾಂಕ size_t igrad_src = ggml_hash_ಕಂಡುಹಿಡಿ(&src->ಭೇಟಿಯಾದ_hash_ಗಣ, src->nodes[i]);
            ಸ್ಥಿರಾಂಕ size_t igrad_dst = ggml_hash_ಕಂಡುಹಿಡಿ(&dst->ಭೇಟಿಯಾದ_hash_ಗಣ, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_ತುಂಬಿದ);
            GGML_ASSERT(ggml_bitset_get(src->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_ತುಂಬಿದ);
            GGML_ASSERT(ggml_bitset_get(dst->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ರಚನೆ ggml_cgraph * ggml_ನಕ್ಷೆ_dup(ರಚನೆ ggml_context * ctx, ರಚನೆ ggml_cgraph * cgraph, bool force_grads) {
    ರಚನೆ ggml_cgraph * ಫಲಿತಾಂಶ = ggml_new_ನಕ್ಷೆ_custom(ctx, cgraph->ಗಾತ್ರ_2, cgraph->grads || force_grads);
    ggml_ನಕ್ಷೆ_cpy(cgraph, ಫಲಿತಾಂಶ);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_zero(ರಚನೆ ggml_tensor * tensor) {
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(tensor)) {
        ಹಿಂತಿರುಗಿ tensor;
    }
    ಆದರೆ (tensor->ಬಫರ್) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(tensor->ದತ್ತಾಂಶ);
        memset(tensor->ದತ್ತಾಂಶ, 0, ggml_nbytes(tensor));
    }
    ಹಿಂತಿರುಗಿ tensor;
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_reset(ರಚನೆ ggml_cgraph * cgraph) {
    ಆದರೆ (!cgraph) {
        ಹಿಂತಿರುಗಿ;
    }
    GGML_ASSERT(cgraph->grads != ಖಾಲಿ);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು     = cgraph->nodes[i];
        ರಚನೆ ggml_tensor * grad_acc = ggml_ನಕ್ಷೆ_get_grad_acc(cgraph, ಗಂಟು);

        ಆದರೆ (ಗಂಟು->op == GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW) {
            // clear momenta
            ggml_ಗಣ_zero(ಗಂಟು->src[2]);
            ggml_ಗಣ_zero(ಗಂಟು->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        ಆದರೆ (grad_acc) {
            ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ onef = 1.0f;
                ಆದರೆ (grad_acc->ಬಫರ್) {
                    ggml_backend_tensor_ಗಣ(grad_acc, &onef, 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ASSERT(grad_acc->ದತ್ತಾಂಶ);
                    *((ದಶಮಾಂಶ *) grad_acc->ದತ್ತಾಂಶ) = onef;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_ಗಣ_zero(grad_acc);
            }
        }
    }
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_clear(ರಚನೆ ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_ಗಣ_reset(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ);
}

ಪೂರ್ಣಾಂಕ ggml_ನಕ್ಷೆ_ಗಾತ್ರ(ರಚನೆ ggml_cgraph * cgraph) {
    ಹಿಂತಿರುಗಿ cgraph->ಗಾತ್ರ_2;
}

ರಚನೆ ggml_tensor * ggml_ನಕ್ಷೆ_ಗಂಟು(ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i) {
    ಆದರೆ (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ಹಿಂತಿರುಗಿ cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ಹಿಂತಿರುಗಿ cgraph->nodes[i];
}

ರಚನೆ ggml_tensor ** ggml_ನಕ್ಷೆ_nodes(ರಚನೆ ggml_cgraph * cgraph) {
    ಹಿಂತಿರುಗಿ cgraph->nodes;
}

ಪೂರ್ಣಾಂಕ ggml_ನಕ್ಷೆ_n_nodes(ರಚನೆ ggml_cgraph * cgraph) {
    ಹಿಂತಿರುಗಿ cgraph->n_nodes;
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_ಕೂಡಿಸು_ಗಂಟು(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->ಗಾತ್ರ_2 > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ರಚನೆ ggml_tensor * ggml_ನಕ್ಷೆ_get_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = cgraph->leafs[i];

        ಆದರೆ (strcmp(ಎಲೆ->name, name) == 0) {
            ಹಿಂತಿರುಗಿ ಎಲೆ;
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಆದರೆ (strcmp(ಗಂಟು->name, name) == 0) {
            ಹಿಂತಿರುಗಿ ಗಂಟು;
        }
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ರಚನೆ ggml_tensor * ggml_ನಕ್ಷೆ_get_grad(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು) {
    ಸ್ಥಿರಾಂಕ size_t igrad = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು);
    ಹಿಂತಿರುಗಿ igrad != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad) && cgraph->grads ? cgraph->grads[igrad] : ಖಾಲಿ;
}

ರಚನೆ ggml_tensor * ggml_ನಕ್ಷೆ_get_grad_acc(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು) {
    ಸ್ಥಿರಾಂಕ size_t igrad = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು);
    ಹಿಂತಿರುಗಿ igrad != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ಖಾಲಿ;
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_print(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ಗಂಟು->ne[0], ಗಂಟು->ne[1], ಗಂಟು->ne[2],
                ggml_op_name(ಗಂಟು->op), (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_ನಕ್ಷೆ_get_grad(cgraph, ಗಂಟು) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ಗಂಟು->ne[0], ಗಂಟು->ne[1],
                ggml_op_name(ಗಂಟು->op),
                ggml_get_name(ಗಂಟು));
    }

    GGML_LOG_INFO("========================================\n");
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_ಗಂಟು_list_ಕಂಡುಹಿಡಿ_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                idxs,
                                      ಪೂರ್ಣಾಂಕ                        ಎಣಿಕೆ_2,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಎಣಿಕೆ_2; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = idxs[i];

        ಆದರೆ (ಗಂಟು_idx >= cgraph->n_nodes) {
            ಹಿಂತಿರುಗಿ -1;
        }
        ಆದರೆ (cgraph->nodes[ಗಂಟು_idx] == tensor) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ -1;
}

ಸ್ಥಿರ bool ggml_is_constant(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor->ಬಫರ್ != ಖಾಲಿ && ggml_backend_ಬಫರ್_get_usage(tensor->ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                ಗಂಟು_idxs,
                                ಪೂರ್ಣಾಂಕ                        ಎಣಿಕೆ_2,
                                ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op *       ops,
                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                outputs,
                                ಪೂರ್ಣಾಂಕ                        ಅಂಕೆ_outputs) {
    GGML_ASSERT(outputs && ಅಂಕೆ_outputs > 0);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಎಣಿಕೆ_2; ++i) {
        ಆದರೆ (ಗಂಟು_idxs[i] >= cgraph->n_nodes) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[i]];

        ಆದರೆ (ಗಂಟು->op != ops[i]) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಆದರೆ (ggml_ಗಂಟು_list_ಕಂಡುಹಿಡಿ_tensor(cgraph, outputs, ಅಂಕೆ_outputs, ಗಂಟು) != -1) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಪೂರ್ಣಾಂಕ subgraph_ಬಳಸುತ್ತದೆ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = i + 1; j < ಎಣಿಕೆ_2; ++j) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * other_ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[j]];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ಆದರೆ (other_ಗಂಟು->src[src_idx] == ಗಂಟು) {
                    subgraph_ಬಳಸುತ್ತದೆ++;
                }
            }
        }

        ಆದರೆ (subgraph_ಬಳಸುತ್ತದೆ != ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idxs[i])) {
            ಹಿಂತಿರುಗಿ false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ರಚನೆ ggml_tensor * view_src = ಗಂಟು->view_src;
        ಆಗ (view_src) {
            ಆದರೆ (ggml_ಗಂಟು_list_ಕಂಡುಹಿಡಿ_tensor(cgraph, ಗಂಟು_idxs, ಎಣಿಕೆ_2, view_src) == -1 && !ggml_is_constant(view_src)) {
                ಹಿಂತಿರುಗಿ false;
            }
            view_src = view_src->view_src;
        }
    }

    ಹಿಂತಿರುಗಿ true;
}

// check if node is part of the graph
ಸ್ಥಿರ bool ggml_ನಕ್ಷೆ_ಕಂಡುಹಿಡಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು) {
    ಆದರೆ (cgraph == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ true;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ಆದರೆ (cgraph->nodes[i] == ಗಂಟು) {
            ಹಿಂತಿರುಗಿ true;
        }
    }

    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_ನಕ್ಷೆ_get_ಪೋಷಕ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಪೋಷಕ = cgraph->nodes[i];
        ರಚನೆ ggml_tensor * grad = ggml_ನಕ್ಷೆ_get_grad(cgraph, ಪೋಷಕ);

        ಆದರೆ (grad == ಗಂಟು) {
            ಹಿಂತಿರುಗಿ ಪೋಷಕ;
        }
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_ನಕ್ಷೆ_dump_dot_ಗಂಟು_ಅಂಚು(FILE * fp, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * gb, ರಚನೆ ggml_tensor * ಗಂಟು, ರಚನೆ ggml_tensor * ಪೋಷಕ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * label)  {
    ರಚನೆ ggml_tensor * gparent = ggml_ನಕ್ಷೆ_get_ಪೋಷಕ(gb, ಗಂಟು);
    ರಚನೆ ggml_tensor * gparent0 = ggml_ನಕ್ಷೆ_get_ಪೋಷಕ(gb, ಪೋಷಕ);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (ಶೂನ್ಯ *) gparent0 : (ಶೂನ್ಯ *) ಪೋಷಕ,
            gparent ? (ಶೂನ್ಯ *) gparent : (ಶೂನ್ಯ *) ಗಂಟು,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_ನಕ್ಷೆ_dump_dot_ಎಲೆ_ಅಂಚು(FILE * fp, ರಚನೆ ggml_tensor * ಗಂಟು, ರಚನೆ ggml_tensor * ಪೋಷಕ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (ಶೂನ್ಯ *) ಪೋಷಕ,
            (ಶೂನ್ಯ *) ಗಂಟು,
            label);
}

ಶೂನ್ಯ ggml_ನಕ್ಷೆ_dump_dot(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * gb, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * filename) {
    ಅಕ್ಷರ color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gb->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = gb->nodes[i];
        ರಚನೆ ggml_tensor * grad = ggml_ನಕ್ಷೆ_get_grad(gb, ಗಂಟು);

        ಆದರೆ (ggml_ನಕ್ಷೆ_get_ಪೋಷಕ(gb, ಗಂಟು) != ಖಾಲಿ) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, ಗಾತ್ರ(color), "yellow");
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (grad) {
            ಆದರೆ (ggml_ನಕ್ಷೆ_ಕಂಡುಹಿಡಿ(cgraph, ಗಂಟು)) {
                snprintf(color, ಗಾತ್ರ(color), "green");
            } ಇಲ್ಲದಿದ್ದರೆ {
                snprintf(color, ಗಾತ್ರ(color), "lightblue");
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            snprintf(color, ಗಾತ್ರ(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (ಶೂನ್ಯ *) ಗಂಟು, color);

        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಗಂಟು->name) > 0) {
            fprintf(fp, "%s (%s)|", ಗಂಟು->name, ggml_type_name(ಗಂಟು->type));
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(fp, "(%s)|", ggml_type_name(ಗಂಟು->type));
        }

        ಆದರೆ (ggml_is_ಮಾತೃಕೆ(ಗಂಟು)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1], ggml_op_ಚಿಹ್ನೆ(ಗಂಟು->op));
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1], ಗಂಟು->ne[2], ggml_op_ಚಿಹ್ನೆ(ಗಂಟು->op));
        }

        ಆದರೆ (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_ಚಿಹ್ನೆ(grad->op));
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(fp, "\"; ]\n");
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gb->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = gb->leafs[i];

        snprintf(color, ಗಾತ್ರ(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (ಶೂನ್ಯ *) ಗಂಟು, color);

        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಗಂಟು->name) > 0) {
            fprintf(fp, "%s (%s)|", ಗಂಟು->name, ggml_type_name(ಗಂಟು->type));
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(fp, "(%s)|", ggml_type_name(ಗಂಟು->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1]);
        ಆದರೆ (ggml_nelements(ಗಂಟು) < 5 && ಗಂಟು->ದತ್ತಾಂಶ != ಖಾಲಿ) {
            fprintf(fp, " | (");
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ggml_nelements(ಗಂಟು); j++) {
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
                ಆದರೆ (j < ggml_nelements(ಗಂಟು) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gb->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = gb->nodes[i];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಆದರೆ (ಗಂಟು->src[j]) {
                ಅಕ್ಷರ label[16];
                snprintf(label, ಗಾತ್ರ(label), "src %d", j);
                ggml_ನಕ್ಷೆ_dump_dot_ಗಂಟು_ಅಂಚು(fp, gb, ಗಂಟು, ಗಂಟು->src[j], label);
            }
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gb->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = gb->leafs[i];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಆದರೆ (ಗಂಟು->src[j]) {
                ಅಕ್ಷರ label[16];
                snprintf(label, ಗಾತ್ರ(label), "src %d", j);
                ggml_ನಕ್ಷೆ_dump_dot_ಎಲೆ_ಅಂಚು(fp, ಗಂಟು, ಗಂಟು->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_ಗಣ_ಒಳಹರಿವು(ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_ಒಳಹರಿವು;
}

ಶೂನ್ಯ ggml_ಗಣ_output(ರಚನೆ ggml_tensor * tensor) {
    ಗಾಗಿ (ರಚನೆ ggml_tensor * cur = tensor; cur != ಖಾಲಿ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

ಶೂನ್ಯ ggml_ಗಣ_param(ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

ಶೂನ್ಯ ggml_ಗಣ_loss(ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_quantize_init(ಎಣಿಕೆ ggml_type type) {
    ggml_critical_section_ಆರಂಭ();

    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // nothing
            ನಿಲ್ಲಿಸು;
    }

    ggml_critical_section_ಕೊನೆ();
}

ಶೂನ್ಯ ggml_quantize_free(ಶೂನ್ಯ) {
    ggml_critical_section_ಆರಂಭ();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_ಕೊನೆ();
}

bool ggml_quantize_requires_imatrix(ಎಣಿಕೆ ggml_type type) {
    ಹಿಂತಿರುಗಿ
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        ಎಣಿಕೆ ggml_type   type,
           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                  ಶೂನ್ಯ * dst,
               int64_t   ಆರಂಭ,
               int64_t   nrows,
               int64_t   n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು,
           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix) {
    ಸ್ಥಿರಾಂಕ int64_t n = nrows * n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಆದರೆ (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ಖಾಲಿ);
    }

    GGML_ASSERT(ಆರಂಭ % type_traits[type].blck_ಗಾತ್ರ == 0);
    GGML_ASSERT(ಆರಂಭ % n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ಸ್ಥಿರಾಂಕ size_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = ಆರಂಭ / n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(type, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);

    size_t ಫಲಿತಾಂಶ = 0;

    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:    ಫಲಿತಾಂಶ = quantize_q1_0   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:    ಫಲಿತಾಂಶ = quantize_q2_0   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:    ಫಲಿತಾಂಶ = quantize_q4_0   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:    ಫಲಿತಾಂಶ = quantize_q4_1   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:    ಫಲಿತಾಂಶ = quantize_q5_0   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:    ಫಲಿತಾಂಶ = quantize_q5_1   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:    ಫಲಿತಾಂಶ = quantize_q8_0   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:   ಫಲಿತಾಂಶ = quantize_mxfp4  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:   ಫಲಿತಾಂಶ = quantize_nvfp4  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:    ಫಲಿತಾಂಶ = quantize_q2_K   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:    ಫಲಿತಾಂಶ = quantize_q3_K   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:    ಫಲಿತಾಂಶ = quantize_q4_K   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:    ಫಲಿತಾಂಶ = quantize_q5_K   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:    ಫಲಿತಾಂಶ = quantize_q6_K   (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:   ಫಲಿತಾಂಶ = quantize_tq1_0  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:   ಫಲಿತಾಂಶ = quantize_tq2_0  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS: ಫಲಿತಾಂಶ = quantize_iq2_xxs(src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:  ಫಲಿತಾಂಶ = quantize_iq2_xs (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS: ಫಲಿತಾಂಶ = quantize_iq3_xxs(src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:   ಫಲಿತಾಂಶ = quantize_iq3_s  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:   ಫಲಿತಾಂಶ = quantize_iq2_s  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:   ಫಲಿತಾಂಶ = quantize_iq1_s  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:   ಫಲಿತಾಂಶ = quantize_iq1_m  (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:  ಫಲಿತಾಂಶ = quantize_iq4_nl (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:  ಫಲಿತಾಂಶ = quantize_iq4_xs (src + ಆರಂಭ, (ಅಕ್ಷರ *) dst + ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows, n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, imatrix); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                size_t elemsize = ಗಾತ್ರ(ggml_fp16_t);
                ggml_fp32_ಗಮ್ಯ_fp16_ಅಡ್ಡಸಾಲು(src + ಆರಂಭ, (ggml_fp16_t *)dst + ಆರಂಭ, n);
                ಫಲಿತಾಂಶ = n * elemsize;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                size_t elemsize = ಗಾತ್ರ(ggml_bf16_t);
                ggml_fp32_ಗಮ್ಯ_bf16_ಅಡ್ಡಸಾಲು_ref(src + ಆರಂಭ, (ggml_bf16_t *)dst + ಆರಂಭ, n);
                ಫಲಿತಾಂಶ = n * elemsize;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                size_t elemsize = ಗಾತ್ರ(ದಶಮಾಂಶ);
                ಫಲಿತಾಂಶ = n * elemsize;
                ಸ್ಮರಣೆ_ನಕಲು((uint8_t *)dst + ಆರಂಭ * elemsize, src + ಆರಂಭ, ಫಲಿತಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಪ್ರತಿಜ್ಞಾ(false);
    }

    GGML_ASSERT(ಫಲಿತಾಂಶ == nrows * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_log_get(ggml_log_callback * log_callback, ಶೂನ್ಯ ** user_ದತ್ತಾಂಶ) {
    *log_callback = g_logger_state.log_callback;
    *user_ದತ್ತಾಂಶ    = g_logger_state.log_callback_user_ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ ggml_log_ಗಣ(ggml_log_callback log_callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_ದತ್ತಾಂಶ = user_ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ ggml_threadpool_params_init(ರಚನೆ ggml_threadpool_params * p, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    p->n_ಎಳೆಗಳು  = n_ಎಳೆಗಳು;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_ಎಳೆಗಳು); // all-zero means use the default affinity (usually inherited)
}

ರಚನೆ ggml_threadpool_params ggml_threadpool_params_default(ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ರಚನೆ ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ p;
}

bool ggml_threadpool_params_ಹೊಂದಿಕೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_threadpool_params * p0, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_threadpool_params * p1) {
    ಆದರೆ (p0->n_ಎಳೆಗಳು  != p1->n_ಎಳೆಗಳು  ) ಹಿಂತಿರುಗಿ false;
    ಆದರೆ (p0->prio       != p1->prio       ) ಹಿಂತಿರುಗಿ false;
    ಆದರೆ (p0->poll       != p1->poll       ) ಹಿಂತಿರುಗಿ false;
    ಆದರೆ (p0->strict_cpu != p1->strict_cpu ) ಹಿಂತಿರುಗಿ false;
    ಹಿಂತಿರುಗಿ memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_ಎಳೆಗಳು) == 0;
}
