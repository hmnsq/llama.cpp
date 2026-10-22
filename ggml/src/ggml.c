#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#ಅನ್ತರ್ಭಾವ "ggml-version.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

// FIXME: required here for quantization functions
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#ಅನ್ತರ್ಭಾವ <hbwmalloc.h>
#endif

#ಯದಿ defined(_MSC_VER) || defined(__MINGW32__)
#ಅನ್ತರ್ಭಾವ <malloc.h> // using malloc.h with MSC/MINGW
#ನೋಚೇದ್ಯದಿ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ಅನ್ತರ್ಭಾವ <alloca.h>
#endif

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
#ಯದಿ defined(__gnu_linux__)
#ಅನ್ತರ್ಭಾವ <syscall.h>
#endif

#ಯದಿ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <mach/mach.h>
#ಅನ್ತರ್ಭಾವ <TargetConditionals.h>
#endif

#ಯದಿ defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#ಅನ್ತರ್ಭಾವ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(ಶೂನ್ಯ) {
#ifdef _MSC_VER
#ಯದಿ defined(_WIN32)
    ಸ್ಥಿರ ಚಞ್ಚಲ LONG counter = 1;
    ಪ್ರತ್ಯಾಗಮ (uint64_t) InterlockedIncrement(&counter) - 1;
#ಅನ್ಯಥಾ
    ಸ್ಥಿರ ಚಞ್ಚಲ long long counter = 1;
    ಪ್ರತ್ಯಾಗಮ (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#ಅನ್ಯಥಾ
    ಸ್ಥಿರ uint64_t counter = 1;
    ಪ್ರತ್ಯಾಗಮ __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#ಯದಿ defined(__AVX512BF16__)
#ಯದಿ defined(_MSC_VER)
#define m512i(p) p
#ಅನ್ಯಥಾ
#ಅನ್ತರ್ಭಾವ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#ಯದಿ defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/wait.h>
#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/prctl.h>
#endif

#ಯದಿ defined(__ANDROID__)
#ಅನ್ತರ್ಭಾವ <unwind.h>
#ಅನ್ತರ್ಭಾವ <dlfcn.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>

ಸಂರಚನಾ backtrace_state {
    ಶೂನ್ಯ ** current;
    ಶೂನ್ಯ ** end;
};

ಸ್ಥಿರ _Unwind_Reason_Code unwind_callback(ಸಂರಚನಾ _Unwind_Context* context, ಶೂನ್ಯ* arg) {
    ಸಂರಚನಾ backtrace_state * state = (ಸಂರಚನಾ backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    ಯದಿ (pc) {
        ಯದಿ (state->current == state->end) {
            ಪ್ರತ್ಯಾಗಮ URC_END_OF_ರಾಶಿ;
        } ಅನ್ಯಥಾ {
            *state->current++ = (ಶೂನ್ಯ*)pc;
        }
    }
    ಪ್ರತ್ಯಾಗಮ _URC_NO_REASON;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max = 100;
    ಶೂನ್ಯ* buffer[max];

    ಸಂರಚನಾ backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    ಪೂರ್ಣಾಙ್ಕ ಎಣಿಕೆ = state.current - buffer;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx = 0; idx < ಎಣಿಕೆ; ++idx) {
        ಧ್ರುವ ಶೂನ್ಯ * addr = buffer[idx];
        ಧ್ರುವ ವರ್ಣ * ಚಿಹ್ನೆ = "";

        Dl_info info;
        ಯದಿ (dladdr(addr, &info) && info.dli_sname) {
            ಚಿಹ್ನೆ = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, ಚಿಹ್ನೆ);
    }
}
#ನೋಚೇದ್ಯದಿ defined(__linux__) && defined(__GLIBC__)
#ಅನ್ತರ್ಭಾವ <execinfo.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಶೂನ್ಯ * trace[100];
    ಪೂರ್ಣಾಙ್ಕ nptrs = backtrace(trace, ಪರಿಮಾಣ(trace)/ಪರಿಮಾಣ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ನೋಚೇದ್ಯದಿ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <execinfo.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    ಶೂನ್ಯ * trace[100];
    ಪೂರ್ಣಾಙ್ಕ nptrs = backtrace(trace, ಪರಿಮಾಣ(trace)/ಪರಿಮಾಣ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ಅನ್ಯಥಾ
ಸ್ಥಿರ ಶೂನ್ಯ ggml_print_backtrace_symbols(ಶೂನ್ಯ) {
    // platform not supported
}
#endif

ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ) {
    ಧ್ರುವ ವರ್ಣ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ಯದಿ (GGML_NO_BACKTRACE) {
        ಪ್ರತ್ಯಾಗಮ;
    }
#ಯದಿ defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ಧ್ರುವ ವರ್ಣ * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    ಯದಿ (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ಪ್ರತ್ಯಾಗಮ;
    }
#endif
#ಯದಿ defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    ವರ್ಣ * ಸಾಲು = ರಿಕ್ತ;
    ssize_t length = 0;
    ಯಾವತ್ ((length = getline(&ಸಾಲು, &size, f)) > 0) {
        ಯದಿ (!strncmp(ಸಾಲು, "TracerPid:", ಪರಿಮಾಣ("TracerPid:") - 1) &&
            (length != ಪರಿಮಾಣ("TracerPid:\t0\n") - 1 || ಸಾಲು[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            ಮುಕ್ತ(ಸಾಲು);
            fclose(f);
            ಪ್ರತ್ಯಾಗಮ;
        }
    }
    ಮುಕ್ತ(ಸಾಲು);
    fclose(f);
    ಪೂರ್ಣಾಙ್ಕ lock[2] = { -1, -1 };
    (ಶೂನ್ಯ) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parent_pid = getpid();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ child_pid = fork();
    ಯದಿ (child_pid < 0) { // error
#ಯದಿ defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        ಪ್ರತ್ಯಾಗಮ;
    } ಅನ್ಯಥಾ ಯದಿ (child_pid == 0) { // child
        ವರ್ಣ attach[32];
        snprintf(attach, ಪರಿಮಾಣ(attach), "attach %d", parent_pid);
#ಯದಿ defined(__linux__)
        close(lock[1]);
        (ಶೂನ್ಯ) !ಪಠ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (ವರ್ಣ *) ರಿಕ್ತ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[ಪರಿಮಾಣ("attach ") - 1],
            (ವರ್ಣ *) ರಿಕ್ತ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } ಅನ್ಯಥಾ { // parent
#ಯದಿ defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, ರಿಕ್ತ, 0);
    }
}
#ಅನ್ಯಥಾ
ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ) {
    // platform not supported
}
#endif

ಸ್ಥಿರ ggml_abort_callback_t g_abort_callback = ರಿಕ್ತ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ಪ್ರತ್ಯಾಗಮ ret_val;
}

ಶೂನ್ಯ ggml_abort(ಧ್ರುವ ವರ್ಣ * file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು, ಧ್ರುವ ವರ್ಣ * fmt, ...) {
    fflush(stdout);

    ವರ್ಣ message[2048];
    ಪೂರ್ಣಾಙ್ಕ offset = snprintf(message, ಪರಿಮಾಣ(message), "%s:%d: ", file, ಸಾಲು);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, ಪರಿಮಾಣ(message) - offset, fmt, args);
    va_end(args);

    ಯದಿ (g_abort_callback) {
        g_abort_callback(message);
    } ಅನ್ಯಥಾ {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", message);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

ಸಂರಚನಾ ggml_logger_state {
    ggml_log_callback log_callback;
    ಶೂನ್ಯ * log_callback_user_data;
};
ಸ್ಥಿರ ಸಂರಚನಾ ggml_logger_state g_logger_state = {ggml_log_callback_default, ರಿಕ್ತ};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_log_internal_v(ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * format, va_list args) {
    ಯದಿ (format == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    ವರ್ಣ buffer[128];
    ಪೂರ್ಣಾಙ್ಕ len = vsnprintf(buffer, 128, format, args);
    ಯದಿ (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } ಅನ್ಯಥಾ {
        ವರ್ಣ * buffer2 = (ವರ್ಣ *) calloc(len + 1, ಪರಿಮಾಣ(ವರ್ಣ));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        ಮುಕ್ತ(buffer2);
    }
    va_end(args_copy);
}

ಶೂನ್ಯ ggml_log_internal(ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

ಶೂನ್ಯ ggml_log_callback_default(ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * text, ಶೂನ್ಯ * user_data) {
    (ಶೂನ್ಯ) level;
    (ಶೂನ್ಯ) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#ifdef GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#endif


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#ಯದಿ defined(__has_feature)
#ಯದಿ __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#ಅನ್ಯಥಾ
#ಯದಿ defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

ಶೂನ್ಯ * ggml_aligned_malloc(size_t size) {
#ಯದಿ defined(__s390x__)
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ alignment = 256;
#ಅನ್ಯಥಾ
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ alignment = 64;
#endif

#ಯದಿ defined(_MSC_VER) || defined(__MINGW32__)
    ಪ್ರತ್ಯಾಗಮ _aligned_malloc(size, alignment);
#ಅನ್ಯಥಾ
    ಯದಿ (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಶೂನ್ಯ * aligned_memory = ರಿಕ್ತ;
  #ifdef GGML_USE_CPU_HBM
    ಪೂರ್ಣಾಙ್ಕ result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #ನೋಚೇದ್ಯದಿ TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    ಪೂರ್ಣಾಙ್ಕ result = EFAULT;
    ವಿಕಲ್ಪ (alloc_status) {
        ಪಕ್ಷ KERN_SUCCESS:
            result = 0;
            ಭಙ್ಗ;
        ಪಕ್ಷ KERN_INVALID_ADDRESS:
            result = EINVAL;
            ಭಙ್ಗ;
        ಪಕ್ಷ KERN_NO_SPACE:
            result = ENOMEM;
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            result = EFAULT;
            ಭಙ್ಗ;
    }
  #ಅನ್ಯಥಾ
    ಪೂರ್ಣಾಙ್ಕ result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    ಯದಿ (result != 0) {
        // Handle allocation failure
        ಧ್ರುವ ವರ್ಣ *error_desc = "unknown allocation error";
        ವಿಕಲ್ಪ (result) {
            ಪಕ್ಷ EINVAL:
                error_desc = "invalid alignment value";
                ಭಙ್ಗ;
            ಪಕ್ಷ ENOMEM:
                error_desc = "insufficient memory";
                ಭಙ್ಗ;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ aligned_memory;
#endif
}

ಶೂನ್ಯ ggml_aligned_free(ಶೂನ್ಯ * ptr, size_t size) {
    GGML_UNUSED(size);
#ಯದಿ defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#ನೋಚೇದ್ಯದಿ GGML_USE_CPU_HBM
    ಯದಿ (ptr != ರಿಕ್ತ) {
        hbw_free(ptr);
    }
#ನೋಚೇದ್ಯದಿ TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    ಯದಿ (ptr != ರಿಕ್ತ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#ಅನ್ಯಥಾ
    ಮುಕ್ತ(ptr);
#endif
}


ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ ಶೂನ್ಯ * ggml_malloc(size_t size) {
    ಯದಿ (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಶೂನ್ಯ * result = malloc(size);
    ಯದಿ (result == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ಪ್ರತ್ಯಾಗಮ result;
}

// calloc
ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ ಶೂನ್ಯ * ggml_calloc(size_t ಅಂಕೆ, size_t size) {
    ಯದಿ (ಅಂಕೆ == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಶೂನ್ಯ * result = calloc(ಅಂಕೆ, size);
    ಯದಿ (result == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ಪ್ರತ್ಯಾಗಮ result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(ಅಂಕೆ, size) ggml_calloc(ಅಂಕೆ, size)

#define GGML_FREE(ptr) ಮುಕ್ತ(ptr)

ಧ್ರುವ ವರ್ಣ * ggml_status_to_string(ಗಣನಾ ggml_status status) {
    ವಿಕಲ್ಪ (status) {
        ಪಕ್ಷ GGML_STATUS_ALLOC_FAILED: ಪ್ರತ್ಯಾಗಮ "GGML status: error (failed to allocate memory)";
        ಪಕ್ಷ GGML_STATUS_FAILED:       ಪ್ರತ್ಯಾಗಮ "GGML status: error (operation failed)";
        ಪಕ್ಷ GGML_STATUS_SUCCESS:      ಪ್ರತ್ಯಾಗಮ "GGML status: success";
        ಪಕ್ಷ GGML_STATUS_ABORTED:      ಪ್ರತ್ಯಾಗಮ "GGML status: warning (operation aborted)";
    }

    ಪ್ರತ್ಯಾಗಮ "GGML status: unknown";
}

ಭಿನ್ನ ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    ಪ್ರತ್ಯಾಗಮ GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(ಭಿನ್ನ x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    ಪ್ರತ್ಯಾಗಮ GGML_FP32_TO_FP16(x);
}

ಭಿನ್ನ ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    ಪ್ರತ್ಯಾಗಮ GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(ಭಿನ್ನ x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    ಪ್ರತ್ಯಾಗಮ GGML_FP32_TO_BF16(x);
}

ಶೂನ್ಯ ggml_fp16_to_fp32_row(ಧ್ರುವ ggml_fp16_t * x, ಭಿನ್ನ * y, int64_t n) {
    ಪ್ರತಿ (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_to_fp16_row(ಧ್ರುವ ಭಿನ್ನ * x, ggml_fp16_t * y, int64_t n) {
    ಪೂರ್ಣಾಙ್ಕ i = 0;
    ಪ್ರತಿ (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

ಶೂನ್ಯ ggml_bf16_to_fp32_row(ಧ್ರುವ ggml_bf16_t * x, ಭಿನ್ನ * y, int64_t n) {
    ಪೂರ್ಣಾಙ್ಕ i = 0;
    ಪ್ರತಿ (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_to_bf16_row_ref(ಧ್ರುವ ಭಿನ್ನ * x, ggml_bf16_t * y, int64_t n) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

ಶೂನ್ಯ ggml_fp32_to_bf16_row(ಧ್ರುವ ಭಿನ್ನ * x, ggml_bf16_t * y, int64_t n) {
  ಪೂರ್ಣಾಙ್ಕ i = 0;
#ಯದಿ defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  ಪ್ರತಿ (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    ಪ್ರತಿ (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ಪ್ರತ್ಯಾಗಮ memcmp(guid_a, guid_b, ಪರಿಮಾಣ(ggml_guid)) == 0;
}

ಧ್ರುವ ವರ್ಣ * ggml_version(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ GGML_VERSION;
}

ಧ್ರುವ ವರ್ಣ * ggml_commit(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ GGML_COMMIT;
}

//
// timing
//

#ಯದಿ defined(_MSC_VER) || defined(__MINGW32__)
ಸ್ಥಿರ int64_t timer_freq, timer_start;
ಸ್ಥಿರ BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    UNUSED(once);
    UNUSED(param);
    UNUSED(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;

    ಪ್ರತ್ಯಾಗಮ TRUE;
}
ಶೂನ್ಯ ggml_time_init(ಶೂನ್ಯ) {
    ಸ್ಥಿರ INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, ರಿಕ್ತ, ರಿಕ್ತ);
}
int64_t ggml_time_ms(ಶೂನ್ಯ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ಪ್ರತ್ಯಾಗಮ ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(ಶೂನ್ಯ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ಪ್ರತ್ಯಾಗಮ ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#ಅನ್ಯಥಾ
ಶೂನ್ಯ ggml_time_init(ಶೂನ್ಯ) {}
int64_t ggml_time_ms(ಶೂನ್ಯ) {
    ಸಂರಚನಾ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ಪ್ರತ್ಯಾಗಮ (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(ಶೂನ್ಯ) {
    ಸಂರಚನಾ timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ಪ್ರತ್ಯಾಗಮ (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ clock();
}

int64_t ggml_cycles_per_ms(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
ಸ್ಥಿರ wchar_t * ggml_mbstowcs(ಧ್ರುವ ವರ್ಣ * mbs) {
    ಪೂರ್ಣಾಙ್ಕ wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, ರಿಕ್ತ, 0);
    ಯದಿ (!wlen) {
        errno = EINVAL;
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * ಪರಿಮಾಣ(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    ಯದಿ (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ wbuf;
}
#endif

FILE * ggml_fopen(ಧ್ರುವ ವರ್ಣ * fname, ಧ್ರುವ ವರ್ಣ * mode) {
#ifdef _WIN32
    FILE * file = ರಿಕ್ತ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    ಯದಿ (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        ಯದಿ (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    ಪ್ರತ್ಯಾಗಮ file;
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ fopen(fname, mode);
#endif

}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits type_traits[GGML_TYPE_ಎಣಿಕೆ] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(ದ್ವಿಗುಣ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(ಭಿನ್ನ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = ಪರಿಮಾಣ(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = ಪರಿಮಾಣ(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = ಪರಿಮಾಣ(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = ಪರಿಮಾಣ(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = ಪರಿಮಾಣ(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = ಪರಿಮಾಣ(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = ಪರಿಮಾಣ(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = ಪರಿಮಾಣ(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = ಪರಿಮಾಣ(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = ಪರಿಮಾಣ(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = ರಿಕ್ತ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = ರಿಕ್ತ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = ರಿಕ್ತ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = ರಿಕ್ತ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = ಪರಿಮಾಣ(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = ಪರಿಮಾಣ(ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_bf16_row_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = ಪರಿಮಾಣ(block_tq2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
};

ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * ggml_get_type_traits(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ &type_traits[type];
}

//
// ggml object
//

ಸಂರಚನಾ ggml_object {
    size_t offs;
    size_t size;

    ಸಂರಚನಾ ggml_object * next;

    ಗಣನಾ ggml_object_type type;

    ವರ್ಣ padding[4];
};

ಸ್ಥಿರ ಧ್ರುವ size_t GGML_OBJECT_SIZE = ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_object);

//
// ggml context
//

ಸಂರಚನಾ ggml_context {
    size_t mem_size;
    ಶೂನ್ಯ * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    ಪೂರ್ಣಾಙ್ಕ    n_objects;

    ಸಂರಚನಾ ggml_object * objects_begin;
    ಸಂರಚನಾ ggml_object * objects_end;
};

//
// data types
//

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * GGML_OP_NAME[GGML_OP_ಎಣಿಕೆ] = {
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

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * GGML_OP_ಚಿಹ್ನೆ[GGML_OP_ಎಣಿಕೆ] = {
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

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * GGML_UNARY_OP_NAME[GGML_UNARY_OP_ಎಣಿಕೆ] = {
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

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * GGML_GLU_OP_NAME[GGML_GLU_OP_ಎಣಿಕೆ] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_ಎಣಿಕೆ == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_print_object(ಧ್ರುವ ಸಂರಚನಾ ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ಧ್ರುವ ಶೂನ್ಯ *) obj->next);
}

ಶೂನ್ಯ ggml_print_objects(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    ಸಂರಚನಾ ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ಧ್ರುವ ಶೂನ್ಯ *) ctx);

    ಯಾವತ್ (obj != ರಿಕ್ತ) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಯದಿ (tensor->ne[i] <= 0) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
    }

    size_t nbytes;
    ಧ್ರುವ size_t blck_size = ggml_blck_size(tensor->type);
    ಯದಿ (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    ಅನ್ಯಥಾ {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ಪ್ರತ್ಯಾಗಮ nbytes;
}

size_t ggml_nbytes_pad(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ type_traits[type].blck_size;
}

size_t ggml_type_size(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ type_traits[type].type_size;
}

size_t ggml_row_size(ಗಣನಾ ggml_type type, int64_t ne) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತಿಜ್ಞಾ(ne % ggml_blck_size(type) == 0);
    ಪ್ರತ್ಯಾಗಮ ggml_type_size(type)*ne/ggml_blck_size(type);
}

ದ್ವಿಗುಣ ggml_type_sizef(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ ((ದ್ವಿಗುಣ)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ಧ್ರುವ ವರ್ಣ * ggml_type_name(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ type_traits[type].type_name;
}

bool ggml_is_quantized(ಗಣನಾ ggml_type type) {
    ಪ್ರತಿಜ್ಞಾ(type >= 0);
    ಪ್ರತಿಜ್ಞಾ(type < GGML_TYPE_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ type_traits[type].is_quantized;
}

ಧ್ರುವ ವರ್ಣ * ggml_op_name(ಗಣನಾ ggml_op op) {
    ಪ್ರತ್ಯಾಗಮ GGML_OP_NAME[op];
}

ಧ್ರುವ ವರ್ಣ * ggml_op_ಚಿಹ್ನೆ(ಗಣನಾ ggml_op op) {
    ಪ್ರತ್ಯಾಗಮ GGML_OP_ಚಿಹ್ನೆ[op];
}

ಧ್ರುವ ವರ್ಣ * ggml_unary_op_name(ಗಣನಾ ggml_unary_op op) {
    ಪ್ರತ್ಯಾಗಮ GGML_UNARY_OP_NAME[op];
}

ಧ್ರುವ ವರ್ಣ * ggml_glu_op_name(ಗಣನಾ ggml_glu_op op) {
    ಪ್ರತ್ಯಾಗಮ GGML_GLU_OP_NAME[op];
}

ಧ್ರುವ ವರ್ಣ * ggml_op_desc(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (t->op == GGML_OP_UNARY) {
        ಗಣನಾ ggml_unary_op uop = ggml_get_unary_op(t);
        ಪ್ರತ್ಯಾಗಮ ggml_unary_op_name(uop);
    }
    ಯದಿ (t->op == GGML_OP_GLU) {
        ಗಣನಾ ggml_glu_op gop = ggml_get_glu_op(t);
        ಪ್ರತ್ಯಾಗಮ ggml_glu_op_name(gop);
    }
    ಪ್ರತ್ಯಾಗಮ ggml_op_name(t->op);
}

size_t ggml_element_size(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor->ne[3] == 1;
}

ಪೂರ್ಣಾಙ್ಕ ggml_n_dims(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        ಯದಿ (tensor->ne[i] > 1) {
            ಪ್ರತ್ಯಾಗಮ i + 1;
        }
    }
    ಪ್ರತ್ಯಾಗಮ 1;
}

ಗಣನಾ ggml_type ggml_ftype_to_ggml_type(ಗಣನಾ ggml_ftype ftype) {
    ಗಣನಾ ggml_type wtype = GGML_TYPE_ಎಣಿಕೆ;

    ವಿಕಲ್ಪ (ftype) {
        ಪಕ್ಷ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_ಎಣಿಕೆ; ಭಙ್ಗ;
        ಪಕ್ಷ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_ಎಣಿಕೆ; ಭಙ್ಗ;
    }

    GGML_ASSERT(wtype != GGML_TYPE_ಎಣಿಕೆ);

    ಪ್ರತ್ಯಾಗಮ wtype;
}

size_t ggml_tensor_overhead(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor->nb[0] > tensor->nb[1];
}

ಸ್ಥಿರ bool ggml_is_contiguous_m_n(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಪೂರ್ಣಾಙ್ಕ m, ಪೂರ್ಣಾಙ್ಕ n) {
    size_t next_nb = ggml_type_size(tensor->type);
    ಯದಿ (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n; i++) {
        ಯದಿ (i > m) {
            ಯದಿ (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                ಪ್ರತ್ಯಾಗಮ false;
            }
            next_nb *= tensor->ne[i];
        } ಅನ್ಯಥಾ {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ಪ್ರತ್ಯಾಗಮ true;
}

bool ggml_is_contiguous(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_is_padded_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಯದಿ (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ಪ್ರತ್ಯಾಗಮ true;
        }
    }
    ಪ್ರತ್ಯಾಗಮ false;
}

bool ggml_are_same_shape(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_repeat_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ಸಂರಚನಾ ggml_context * ggml_init(ಸಂರಚನಾ ggml_init_params params) {
    ಸ್ಥಿರ bool is_first_call = true;

    ggml_critical_section_start();

    ಯದಿ (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    ಸಂರಚನಾ ggml_context * ctx = GGML_MALLOC(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_context));

    // allow to call ggml_init with 0 size
    ಯದಿ (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ಧ್ರುವ size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (ಸಂರಚನಾ ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ರಿಕ್ತ,
        /*.objects_end        =*/ ರಿಕ್ತ,
    };

    GGML_ASSERT(ctx->mem_buffer != ರಿಕ್ತ);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ಪ್ರತ್ಯಾಗಮ ctx;
}

ಶೂನ್ಯ ggml_reset(ಸಂರಚನಾ ggml_context * ctx) {
    ಯದಿ (ctx == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ರಿಕ್ತ;
    ctx->objects_end   = ರಿಕ್ತ;
}

ಶೂನ್ಯ ggml_free(ಸಂರಚನಾ ggml_context * ctx) {
    ಯದಿ (ctx == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->objects_end == ರಿಕ್ತ ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(ಸಂರಚನಾ ggml_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->no_alloc;
}

ಶೂನ್ಯ ggml_set_no_alloc(ಸಂರಚನಾ ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

ಶೂನ್ಯ * ggml_get_mem_buffer(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->mem_buffer;
}

size_t ggml_get_mem_size(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    size_t max_size = 0;

    ಪ್ರತಿ (ಸಂರಚನಾ ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != ರಿಕ್ತ; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    ಪ್ರತ್ಯಾಗಮ max_size;
}

////////////////////////////////////////////////////////////////////////////////

ಸ್ಥಿರ ಸಂರಚನಾ ggml_object * ggml_new_object(ಸಂರಚನಾ ggml_context * ctx, ಗಣನಾ ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    ಸಂರಚನಾ ggml_object * obj_cur = ctx->objects_end;

    ಧ್ರುವ size_t cur_offs = obj_cur == ರಿಕ್ತ ? 0 : obj_cur->offs;
    ಧ್ರುವ size_t cur_size = obj_cur == ರಿಕ್ತ ? 0 : obj_cur->size;
    ಧ್ರುವ size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    ವರ್ಣ * ಧ್ರುವ mem_buffer = ctx->mem_buffer;
    ಸಂರಚನಾ ggml_object * ಧ್ರುವ obj_new = (ಸಂರಚನಾ ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    ಯದಿ (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಯದಿ (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    *obj_new = (ಸಂರಚನಾ ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = ರಿಕ್ತ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    ಯದಿ (obj_cur != ರಿಕ್ತ) {
        obj_cur->next = obj_new;
    } ಅನ್ಯಥಾ {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ಪ್ರತ್ಯಾಗಮ obj_new;
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_new_tensor_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type      type,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಧ್ರುವ int64_t       * ne,
        ಸಂರಚನಾ ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_ಎಣಿಕೆ);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    ಯದಿ (view_src != ರಿಕ್ತ && view_src->view_src != ರಿಕ್ತ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_dims; i++) {
        ಯದಿ (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == ರಿಕ್ತ || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    ಶೂನ್ಯ * data = view_src != ರಿಕ್ತ ? view_src->data : ರಿಕ್ತ;
    ಯದಿ (data != ರಿಕ್ತ) {
        data = (ವರ್ಣ *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    ಯದಿ (view_src == ರಿಕ್ತ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    ಸಂರಚನಾ ggml_object * ಧ್ರುವ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    ಸಂರಚನಾ ggml_tensor * ಧ್ರುವ result = (ಸಂರಚನಾ ggml_tensor *)((ವರ್ಣ *)ctx->mem_buffer + obj_new->offs);

    *result = (ಸಂರಚನಾ ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ರಿಕ್ತ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ರಿಕ್ತ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (ಶೂನ್ಯ *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ರಿಕ್ತ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_new_tensor(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type      type,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಧ್ರುವ int64_t       * ne) {
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor_impl(ctx, type, n_dims, ne, ರಿಕ್ತ, 0);
}

ಸಂರಚನಾ ggml_tensor * ggml_new_tensor_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type      type,
        int64_t ne0) {
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor(ctx, type, 1, &ne0);
}

ಸಂರಚನಾ ggml_tensor * ggml_new_tensor_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ಧ್ರುವ int64_t ne[2] = { ne0, ne1 };
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor(ctx, type, 2, ne);
}

ಸಂರಚನಾ ggml_tensor * ggml_new_tensor_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ಧ್ರುವ int64_t ne[3] = { ne0, ne1, ne2 };
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor(ctx, type, 3, ne);
}

ಸಂರಚನಾ ggml_tensor * ggml_new_tensor_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ಧ್ರುವ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor(ctx, type, 4, ne);
}

ಶೂನ್ಯ * ggml_new_buffer(ಸಂರಚನಾ ggml_context * ctx, size_t nbytes) {
    ಸಂರಚನಾ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    ಪ್ರತ್ಯಾಗಮ (uint8_t *)ctx->mem_buffer + obj->offs;
}

ಸಂರಚನಾ ggml_tensor * ggml_dup_tensor(ಸಂರಚನಾ ggml_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src) {
    ಪ್ರತ್ಯಾಗಮ ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

ಶೂನ್ಯ ggml_unravel_index(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ಧ್ರುವ int64_t ne2 = tensor->ne[2];
    ಧ್ರುವ int64_t ne1 = tensor->ne[1];
    ಧ್ರುವ int64_t ne0 = tensor->ne[0];

    ಧ್ರುವ int64_t i3_ = (i/(ne2*ne1*ne0));
    ಧ್ರುವ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ಧ್ರುವ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ಧ್ರುವ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    ಯದಿ (i0) {
        * i0 = i0_;
    }
    ಯದಿ (i1) {
        * i1 = i1_;
    }
    ಯದಿ (i2) {
        * i2 = i2_;
    }
    ಯದಿ (i3) {
        * i3 = i3_;
    }
}

ಶೂನ್ಯ * ggml_get_data(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor->data;
}

ಭಿನ್ನ * ggml_get_data_f32(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿಜ್ಞಾ(tensor->type == GGML_TYPE_F32);
    ಪ್ರತ್ಯಾಗಮ (ಭಿನ್ನ *)(tensor->data);
}

ಗಣನಾ ggml_unary_op ggml_get_unary_op(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ಪ್ರತ್ಯಾಗಮ (ಗಣನಾ ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

ಗಣನಾ ggml_glu_op ggml_get_glu_op(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ಪ್ರತ್ಯಾಗಮ (ಗಣನಾ ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ಧ್ರುವ ವರ್ಣ * ggml_get_name(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor->name;
}

ಸಂರಚನಾ ggml_tensor * ggml_set_name(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ವರ್ಣ * name) {
    size_t i;
    ಪ್ರತಿ (i = 0; i < ಪರಿಮಾಣ(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ಪ್ರತ್ಯಾಗಮ tensor;
}

ಸಂರಚನಾ ggml_tensor * ggml_format_name(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ವರ್ಣ * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, ಪರಿಮಾಣ(tensor->name), fmt, args);
    va_end(args);
    ಪ್ರತ್ಯಾಗಮ tensor;
}

ಸಂರಚನಾ ggml_tensor * ggml_view_tensor(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * src) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_get_first_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx) {
    ಸಂರಚನಾ ggml_object * obj = ctx->objects_begin;

    ವರ್ಣ * ಧ್ರುವ mem_buffer = ctx->mem_buffer;

    ಯಾವತ್ (obj != ರಿಕ್ತ) {
        ಯದಿ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ಪ್ರತ್ಯಾಗಮ (ಸಂರಚನಾ ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

ಸಂರಚನಾ ggml_tensor * ggml_get_next_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_context * ctx, ಸಂರಚನಾ ggml_tensor * tensor) {
    ಸಂರಚನಾ ggml_object * obj = (ಸಂರಚನಾ ggml_object *) ((ವರ್ಣ *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    ವರ್ಣ * ಧ್ರುವ mem_buffer = ctx->mem_buffer;

    ಯಾವತ್ (obj != ರಿಕ್ತ) {
        ಯದಿ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ಪ್ರತ್ಯಾಗಮ (ಸಂರಚನಾ ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

ಸಂರಚನಾ ggml_tensor * ggml_get_tensor(ಸಂರಚನಾ ggml_context * ctx, ಧ್ರುವ ವರ್ಣ * name) {
    ಸಂರಚನಾ ggml_object * obj = ctx->objects_begin;

    ವರ್ಣ * ಧ್ರುವ mem_buffer = ctx->mem_buffer;

    ಯಾವತ್ (obj != ರಿಕ್ತ) {
        ಯದಿ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ಸಂರಚನಾ ggml_tensor * cur = (ಸಂರಚನಾ ggml_tensor *)(mem_buffer + obj->offs);
            ಯದಿ (strcmp(cur->name, name) == 0) {
                ಪ್ರತ್ಯಾಗಮ cur;
            }
        }

        obj = obj->next;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_dup_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_dup(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_dup_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_dup_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_dup_impl(ctx, a, true);
}

// ggml_add

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_add_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_add(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_add_impl(ctx, a, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_add_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_add_cast_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಗಣನಾ   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_add_cast(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಗಣನಾ   ggml_type      type) {
    ಪ್ರತ್ಯಾಗಮ ggml_add_cast_impl(ctx, a, b, type);
}

ಸಂರಚನಾ ggml_tensor * ggml_add_id(
            ಸಂರಚನಾ ggml_context * ctx,
            ಸಂರಚನಾ ggml_tensor  * a,
            ಸಂರಚನಾ ggml_tensor  * b,
            ಸಂರಚನಾ ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_add1

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_add1_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_add1(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_add1_impl(ctx, a, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_add1_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_acc_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_acc(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_acc_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_sub_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_sub(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_sub_impl(ctx, a, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_sub_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_mul_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_mul(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_mul_impl(ctx, a, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_mul_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_div_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_div(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_div_impl(ctx, a, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_div_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_sqr_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_sqr(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sqr_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_sqr_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_sqrt_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_sqrt(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sqrt_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_sqrt_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_log_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_log(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_log_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_log_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_log_impl(ctx, a, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_expm1(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ಸಂರಚನಾ ggml_tensor * ggml_expm1_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ಸಂರಚನಾ ggml_tensor * ggml_softplus(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ಸಂರಚನಾ ggml_tensor * ggml_softplus_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_sin_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_sin(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sin_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_sin_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_cos_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_cos(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_cos_impl(ctx, a, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_cos_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ಸಂರಚನಾ ggml_tensor * ggml_sum(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_sum_rows

ಸಂರಚನಾ ggml_tensor * ggml_sum_rows(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_cumsum

ಸಂರಚನಾ ggml_tensor * ggml_cumsum(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_mean

ಸಂರಚನಾ ggml_tensor * ggml_mean(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_argmax

ಸಂರಚನಾ ggml_tensor * ggml_argmax(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_count_equal

ಸಂರಚನಾ ggml_tensor * ggml_ಎಣಿಕೆ_equal(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_ಎಣಿಕೆ_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_repeat

ಸಂರಚನಾ ggml_tensor * ggml_repeat(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_repeat_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ಧ್ರುವ bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_repeat_back

ಸಂರಚನಾ ggml_tensor * ggml_repeat_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_concat

ಸಂರಚನಾ ggml_tensor * ggml_concat(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಸಂರಚನಾ ggml_tensor  * b,
    ಪೂರ್ಣಾಙ್ಕ                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < GGML_MAX_DIMS; ++d) {
        ಯದಿ (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ಅನುವರ್ತ;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_abs

ಸಂರಚನಾ ggml_tensor * ggml_abs(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ಸಂರಚನಾ ggml_tensor * ggml_abs_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ಸಂರಚನಾ ggml_tensor * ggml_sgn(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ಸಂರಚನಾ ggml_tensor * ggml_sgn_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ಸಂರಚನಾ ggml_tensor * ggml_neg(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ಸಂರಚನಾ ggml_tensor * ggml_neg_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ಸಂರಚನಾ ggml_tensor * ggml_step(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

ಸಂರಚನಾ ggml_tensor * ggml_step_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

ಸಂರಚನಾ ggml_tensor * ggml_tanh(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ಸಂರಚನಾ ggml_tensor * ggml_tanh_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ಸಂರಚನಾ ggml_tensor * ggml_elu(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ಸಂರಚನಾ ggml_tensor * ggml_elu_inplace(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ಸಂರಚನಾ ggml_tensor * ggml_relu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ಸಂರಚನಾ ggml_tensor * ggml_relu_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ಸಂರಚನಾ ggml_tensor * ggml_leaky_relu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 negative_slope,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, ಪರಿಮಾಣ(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_sigmoid

ಸಂರಚನಾ ggml_tensor * ggml_sigmoid(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ಸಂರಚನಾ ggml_tensor * ggml_sigmoid_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ಸಂರಚನಾ ggml_tensor * ggml_gelu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ಸಂರಚನಾ ggml_tensor * ggml_gelu_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ಸಂರಚನಾ ggml_tensor * ggml_gelu_erf(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ಸಂರಚನಾ ggml_tensor * ggml_gelu_erf_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ಸಂರಚನಾ ggml_tensor * ggml_gelu_quick(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

ಸಂರಚನಾ ggml_tensor * ggml_gelu_quick_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

ಸಂರಚನಾ ggml_tensor * ggml_silu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ಸಂರಚನಾ ggml_tensor * ggml_silu_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ಸಂರಚನಾ ggml_tensor * ggml_xielu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ alpha_n,
        ಭಿನ್ನ alpha_p,
        ಭಿನ್ನ beta,
        ಭಿನ್ನ eps) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_silu_back

ಸಂರಚನಾ ggml_tensor * ggml_silu_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml hardswish

ಸಂರಚನಾ ggml_tensor * ggml_hardswish(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ಸಂರಚನಾ ggml_tensor * ggml_hardsigmoid(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ಸಂರಚನಾ ggml_tensor * ggml_exp(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ಸಂರಚನಾ ggml_tensor * ggml_exp_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_glu_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಗಣನಾ ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    ಯದಿ (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ರಿಕ್ತ, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_floor

ಸಂರಚನಾ ggml_tensor * ggml_floor(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ಸಂರಚನಾ ggml_tensor * ggml_floor_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ಸಂರಚನಾ ggml_tensor * ggml_ceil(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ಸಂರಚನಾ ggml_tensor * ggml_ceil_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ಸಂರಚನಾ ggml_tensor * ggml_round(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

ಸಂರಚನಾ ggml_tensor * ggml_round_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

ಸಂರಚನಾ ggml_tensor * ggml_trunc(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ಸಂರಚನಾ ggml_tensor * ggml_trunc_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ಸಂರಚನಾ ggml_tensor * ggml_glu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_glu_op      op,
        bool                  swapped) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, op, swapped);
}

ಸಂರಚನಾ ggml_tensor * ggml_glu_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಗಣನಾ ggml_glu_op      op) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ಸಂರಚನಾ ggml_tensor * ggml_reglu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_REGLU, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_reglu_swapped(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_REGLU, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_reglu_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ಸಂರಚನಾ ggml_tensor * ggml_geglu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_swapped(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ಸಂರಚನಾ ggml_tensor * ggml_swiglu(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_SWIGLU, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_swiglu_swapped(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_SWIGLU, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_swiglu_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ಸಂರಚನಾ ggml_tensor * ggml_geglu_erf(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU_ERF, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_erf_swapped(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU_ERF, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_erf_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ಸಂರಚನಾ ggml_tensor * ggml_geglu_quick(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU_QUICK, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_quick_swapped(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, ರಿಕ್ತ, GGML_GLU_OP_GEGLU_QUICK, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_geglu_quick_split(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_swiglu_oai(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 alpha,
        ಭಿನ್ನ                 limit) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_swiglu_clamp(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 limit) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_norm

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_norm_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ಪರಿಮಾಣ(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_norm(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_norm_impl(ctx, a, eps, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_norm_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_rms_norm_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ಪರಿಮಾಣ(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_rms_norm(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_rms_norm_impl(ctx, a, eps, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_rms_norm_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ಸಂರಚನಾ ggml_tensor * ggml_rms_norm_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 eps) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ಪರಿಮಾಣ(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_group_norm

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_group_norm_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_groups,
        ಭಿನ್ನ                 eps,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_group_norm(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_groups,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_group_norm_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_groups,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_l2_norm_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_l2_norm(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_l2_norm_impl(ctx, a, eps, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_l2_norm_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 eps) {
    ಪ್ರತ್ಯಾಗಮ ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        ಸಂರಚನಾ ggml_tensor * a,
        ಗಣನಾ ggml_prec       prec) {
    ವಿಕಲ್ಪ (a->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            {
                ಧ್ರುವ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            {
                ಧ್ರುವ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ false;
    };

    ಪ್ರತ್ಯಾಗಮ true;
}

bool ggml_prec_set_src(
        ಸಂರಚನಾ ggml_tensor * a,
        ಗಣನಾ ggml_prec       prec,
        ಪೂರ್ಣಾಙ್ಕ                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ವಿಕಲ್ಪ (a->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            {
                ಯದಿ (idx != 1) {
                    ಪ್ರತ್ಯಾಗಮ false;
                }

                ಧ್ರುವ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ false;
    };

    ಪ್ರತ್ಯಾಗಮ true;
}

// ggml_mul_mat

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_mul_mat(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ಸಂರಚನಾ ggml_tensor * ggml_mul_mat(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ಧ್ರುವ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಶೂನ್ಯ ggml_mul_mat_set_prec(
        ಸಂರಚನಾ ggml_tensor * a,
        ಗಣನಾ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ಧ್ರುವ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

ಶೂನ್ಯ ggml_mul_mat_set_hint(
        ಸಂರಚನಾ ggml_tensor * a,
        ಗಣನಾ ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ಧ್ರುವ int32_t hint_i32 = (int32_t) hint;

    ggml_set_op_params_i32(a, 1, hint_i32);
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
ಸಂರಚನಾ ggml_tensor * ggml_mul_mat_id(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * as,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ಧ್ರುವ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_out_prod

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_out_prod(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಪ್ರತ್ಯಾಗಮ (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ಸಂರಚನಾ ggml_tensor * ggml_out_prod(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ಧ್ರುವ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_scale

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_scale_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 s,
        ಭಿನ್ನ                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಭಿನ್ನ params[2] = { s, b };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_scale(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 s) {
    ಪ್ರತ್ಯಾಗಮ ggml_scale_impl(ctx, a, s, 0.0, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_scale_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 s) {
    ಪ್ರತ್ಯಾಗಮ ggml_scale_impl(ctx, a, s, 0.0, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_scale_bias(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 s,
        ಭಿನ್ನ                 b) {
    ಪ್ರತ್ಯಾಗಮ ggml_scale_impl(ctx, a, s, b, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_scale_bias_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 s,
        ಭಿನ್ನ                 b) {
    ಪ್ರತ್ಯಾಗಮ ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_set_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_set(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_set_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_set_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_set_1d_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_set_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_set_2d_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ಪ್ರತ್ಯಾಗಮ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_cpy_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, b);
    ಯದಿ (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } ಅನ್ಯಥಾ {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_cpy(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor * a,
        ಸಂರಚನಾ ggml_tensor * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_cpy_impl(ctx, a, b);
}

ಸಂರಚನಾ ggml_tensor * ggml_cast(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ   ggml_type      type) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_cont

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_cont_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_cont(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ಸಂರಚನಾ ggml_tensor * ggml_cont_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0) {
    ಪ್ರತ್ಯಾಗಮ ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ಸಂರಚನಾ ggml_tensor * ggml_cont_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ಪ್ರತ್ಯಾಗಮ ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ಸಂರಚನಾ ggml_tensor * ggml_cont_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ಪ್ರತ್ಯಾಗಮ ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ಸಂರಚನಾ ggml_tensor * ggml_cont_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_reshape

ಸಂರಚನಾ ggml_tensor * ggml_reshape(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor * a,
        ಸಂರಚನಾ ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_reshape_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ಧ್ರುವ int64_t ne[1] = { ne0 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_reshape_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ಧ್ರುವ int64_t ne[2] = { ne0, ne1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_reshape_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ಧ್ರುವ int64_t ne[3] = { ne0, ne1, ne2 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_reshape_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ಧ್ರುವ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_view_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಧ್ರುವ int64_t       * ne,
        size_t                offset) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, ಪರಿಮಾಣ(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_view_1d

ಸಂರಚನಾ ggml_tensor * ggml_view_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_view_2d

ಸಂರಚನಾ ggml_tensor * ggml_view_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ಧ್ರುವ int64_t ne[2] = { ne0, ne1 };

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_view_3d

ಸಂರಚನಾ ggml_tensor * ggml_view_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ಧ್ರುವ int64_t ne[3] = { ne0, ne1, ne2 };

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_view_4d

ಸಂರಚನಾ ggml_tensor * ggml_view_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ಧ್ರುವ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_permute

ಸಂರಚನಾ ggml_tensor * ggml_permute(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   axis0,
        ಪೂರ್ಣಾಙ್ಕ                   axis1,
        ಪೂರ್ಣಾಙ್ಕ                   axis2,
        ಪೂರ್ಣಾಙ್ಕ                   axis3) {
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

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (permuted)", a->name);

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

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op     = GGML_OP_PERMUTE;
    result->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_transpose

ಸಂರಚನಾ ggml_tensor * ggml_transpose(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_get_rows

ಸಂರಚನಾ ggml_tensor * ggml_get_rows(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    ಗಣನಾ ggml_type type = GGML_TYPE_F32;
    ಯದಿ (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_get_rows_back

ಸಂರಚನಾ ggml_tensor * ggml_get_rows_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_set_rows

ಸಂರಚನಾ ggml_tensor * ggml_set_rows(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_rows(a));
    GGML_ASSERT(ggml_is_contiguous_rows(b));

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_diag

ಸಂರಚನಾ ggml_tensor * ggml_diag(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ಧ್ರುವ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_diag_mask_inf

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_inf_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_inf(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past) {
    ಪ್ರತ್ಯಾಗಮ ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_inf_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past) {
    ಪ್ರತ್ಯಾಗಮ ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_zero_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_zero(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past) {
    ಪ್ರತ್ಯಾಗಮ ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_diag_mask_zero_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_past) {
    ಪ್ರತ್ಯಾಗಮ ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_clamp_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಭಿನ್ನ                 min,
        ಭಿನ್ನ                 max,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಭಿನ್ನ params[] = { min, max };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_clamp(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಭಿನ್ನ                 min,
    ಭಿನ್ನ                 max) {
    ಪ್ರತ್ಯಾಗಮ ggml_clamp_impl(ctx, a, min, max, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_clamp_inplace(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಭಿನ್ನ                 min,
    ಭಿನ್ನ                 max) {
    ಪ್ರತ್ಯಾಗಮ ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_soft_max_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * mask,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    ಯದಿ (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    ಯದಿ (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಭಿನ್ನ params[] = { scale, max_bias };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_impl(ctx, a, ರಿಕ್ತ, 1.0f, 0.0f, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_impl(ctx, a, ರಿಕ್ತ, 1.0f, 0.0f, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max_ext(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * mask,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max_ext_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * mask,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

ಶೂನ್ಯ ggml_soft_max_add_sinks(
        ಸಂರಚನಾ ggml_tensor * a,
        ಸಂರಚನಾ ggml_tensor * sinks) {
    ಯದಿ (!sinks) {
        a->src[2] = ರಿಕ್ತ;
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ರಿಕ್ತ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_soft_max_ext_back_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias,
        bool                  inplace) {
    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((ಭಿನ್ನ *) result->op_params + 0, &scale,    ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy((ಭಿನ್ನ *) result->op_params + 1, &max_bias, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max_ext_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_soft_max_ext_back_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias) {
    ಪ್ರತ್ಯಾಗಮ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_rope_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    ಯದಿ (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    ಯದಿ (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(params +  6, &freq_scale,   ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(params +  7, &ext_factor,   ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(params +  8, &attn_factor,  ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(params +  9, &beta_fast,    ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(params + 10, &beta_slow,    ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಯದಿ (mrope_used && sections) {
        memcpy(params + 11, sections,  ಪರಿಮಾಣ(int32_t) * GGML_MROPE_SECTIONS);
    } ಅನ್ಯಥಾ {
        memset(params + 11, 0,         ಪರಿಮಾಣ(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_rope(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, ರಿಕ್ತ, n_dims, ರಿಕ್ತ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_multi(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_multi_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   sections[GGML_MROPE_SECTIONS],
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, ರಿಕ್ತ, n_dims, ರಿಕ್ತ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_ext(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, c, n_dims, ರಿಕ್ತ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_ext_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, c, n_dims, ರಿಕ್ತ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_custom(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, ರಿಕ್ತ, n_dims, ರಿಕ್ತ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_custom_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಪ್ರತ್ಯಾಗಮ ggml_rope_impl(
        ctx, a, b, ರಿಕ್ತ, n_dims, ರಿಕ್ತ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ಸ್ಥಿರ ಭಿನ್ನ ggml_rope_yarn_corr_dim(ಪೂರ್ಣಾಙ್ಕ n_dims, ಪೂರ್ಣಾಙ್ಕ n_ctx_orig, ಭಿನ್ನ n_rot, ಭಿನ್ನ base) {
    ಪ್ರತ್ಯಾಗಮ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ಭಿನ್ನ)M_PI)) / (2 * logf(base));
}

ಶೂನ್ಯ ggml_rope_yarn_corr_dims(
    ಪೂರ್ಣಾಙ್ಕ n_dims, ಪೂರ್ಣಾಙ್ಕ n_ctx_orig, ಭಿನ್ನ freq_base, ಭಿನ್ನ beta_fast, ಭಿನ್ನ beta_slow, ಭಿನ್ನ dims[2]
) {
    // start and end correction dims
    ಭಿನ್ನ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ಭಿನ್ನ end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

ಸಂರಚನಾ ggml_tensor * ggml_rope_ext_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_multi_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c,
        ಪೂರ್ಣಾಙ್ಕ                   n_dims,
        ಪೂರ್ಣಾಙ್ಕ                   sections[4],
        ಪೂರ್ಣಾಙ್ಕ                   mode,
        ಪೂರ್ಣಾಙ್ಕ                   n_ctx_orig,
        ಭಿನ್ನ                 freq_base,
        ಭಿನ್ನ                 freq_scale,
        ಭಿನ್ನ                 ext_factor,
        ಭಿನ್ನ                 attn_factor,
        ಭಿನ್ನ                 beta_fast,
        ಭಿನ್ನ                 beta_slow) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_rope_set_offset(
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ಧ್ರುವ int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    ಪ್ರತ್ಯಾಗಮ a;
}

ಸ್ಥಿರ int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, ಪೂರ್ಣಾಙ್ಕ s, ಪೂರ್ಣಾಙ್ಕ p, ಪೂರ್ಣಾಙ್ಕ d) {
    ಪ್ರತ್ಯಾಗಮ (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ಸಂರಚನಾ ggml_tensor * ggml_im2col(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   d0,
        ಪೂರ್ಣಾಙ್ಕ                   d1,
        bool                  is_2D,
        ಗಣನಾ ggml_type        dst_type) {
    ಯದಿ (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } ಅನ್ಯಥಾ {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ಧ್ರುವ int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ಧ್ರುವ int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ಧ್ರುವ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_im2col_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        int64_t             * ne,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   d0,
        ಪೂರ್ಣಾಙ್ಕ                   d1,
        bool                  is_2D) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_1d

ಸಂರಚನಾ ggml_tensor * ggml_conv_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   d0) {
    ಸಂರಚನಾ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ಸಂರಚನಾ ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_1d_ph

ಸಂರಚನಾ ggml_tensor* ggml_conv_1d_ph(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s,
        ಪೂರ್ಣಾಙ್ಕ                   d) {
    ಪ್ರತ್ಯಾಗಮ ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ಸಂರಚನಾ ggml_tensor * ggml_conv_1d_dw(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   d0) {
    ಸಂರಚನಾ ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ಸಂರಚನಾ ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ಸಂರಚನಾ ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_1d_dw_ph

ಸಂರಚನಾ ggml_tensor * ggml_conv_1d_dw_ph(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   d0) {
    ಪ್ರತ್ಯಾಗಮ ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ಸಂರಚನಾ ggml_tensor * ggml_col2im_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   oc,
        ಪೂರ್ಣಾಙ್ಕ                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ಧ್ರುವ int64_t K_OC = a->ne[0];
    ಧ್ರುವ int64_t T_in = a->ne[1];
    ಧ್ರುವ int64_t K = K_OC / oc;
    ಧ್ರುವ int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ಧ್ರುವ int64_t ne[4] = { T_out, oc, 1, 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_transpose_1d

ಸ್ಥಿರ int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, ಪೂರ್ಣಾಙ್ಕ s, ಪೂರ್ಣಾಙ್ಕ p, ಪೂರ್ಣಾಙ್ಕ d) {
    ಪ್ರತ್ಯಾಗಮ (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ಸಂರಚನಾ ggml_tensor * ggml_conv_transpose_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ಧ್ರುವ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ಸಂರಚನಾ ggml_tensor * ggml_conv_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   d0,
        ಪೂರ್ಣಾಙ್ಕ                   d1) {
    ಸಂರಚನಾ ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ಸಂರಚನಾ ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ಪ್ರತ್ಯಾಗಮ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ಸಂರಚನಾ ggml_tensor * ggml_im2col_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        int64_t               IC,
        ಪೂರ್ಣಾಙ್ಕ                   s0, // stride width
        ಪೂರ್ಣಾಙ್ಕ                   s1, // stride height
        ಪೂರ್ಣಾಙ್ಕ                   s2, // stride depth
        ಪೂರ್ಣಾಙ್ಕ                   p0, // padding width
        ಪೂರ್ಣಾಙ್ಕ                   p1, // padding height
        ಪೂರ್ಣಾಙ್ಕ                   p2, // padding depth
        ಪೂರ್ಣಾಙ್ಕ                   d0, // dilation width
        ಪೂರ್ಣಾಙ್ಕ                   d1, // dilation height
        ಪೂರ್ಣಾಙ್ಕ                   d2, // dilation depth
        ಗಣನಾ ggml_type        dst_type) {
    ಧ್ರುವ int64_t N = b->ne[3] / IC;
    ಧ್ರುವ int64_t ID = b->ne[2];
    ಧ್ರುವ int64_t IH = b->ne[1];
    ಧ್ರುವ int64_t IW = b->ne[0];

    ಧ್ರುವ int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    ಧ್ರುವ int64_t KD = a->ne[2];
    ಧ್ರುವ int64_t KH = a->ne[1];
    ಧ್ರುವ int64_t KW = a->ne[0];
    ಧ್ರುವ int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    ಧ್ರುವ int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    ಧ್ರುವ int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ಧ್ರುವ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ಸಂರಚನಾ ggml_tensor * ggml_conv_3d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        int64_t               IC,
        ಪೂರ್ಣಾಙ್ಕ                   s0, // stride width
        ಪೂರ್ಣಾಙ್ಕ                   s1, // stride height
        ಪೂರ್ಣಾಙ್ಕ                   s2, // stride depth
        ಪೂರ್ಣಾಙ್ಕ                   p0, // padding width
        ಪೂರ್ಣಾಙ್ಕ                   p1, // padding height
        ಪೂರ್ಣಾಙ್ಕ                   p2, // padding depth
        ಪೂರ್ಣಾಙ್ಕ                   d0, // dilation width
        ಪೂರ್ಣಾಙ್ಕ                   d1, // dilation height
        ಪೂರ್ಣಾಙ್ಕ                   d2  // dilation depth
        ) {
    ಸಂರಚನಾ ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ಸಂರಚನಾ ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_2d_sk_p0

ಸಂರಚನಾ ggml_tensor * ggml_conv_2d_sk_p0(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ಸಂರಚನಾ ggml_tensor * ggml_conv_2d_s1_ph(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    ಪ್ರತ್ಯಾಗಮ ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ಸಂರಚನಾ ggml_tensor * ggml_conv_2d_dw(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   d0,
        ಪೂರ್ಣಾಙ್ಕ                   d1) {
    ಸಂರಚನಾ ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ಸಂರಚನಾ ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ಸಂರಚನಾ ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ಸಂರಚನಾ ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_2d_dw_direct

ಸಂರಚನಾ ggml_tensor * ggml_conv_2d_dw_direct(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   stride0,
        ಪೂರ್ಣಾಙ್ಕ                   stride1,
        ಪೂರ್ಣಾಙ್ಕ                   pad0,
        ಪೂರ್ಣಾಙ್ಕ                   pad1,
        ಪೂರ್ಣಾಙ್ಕ                   dilation0,
        ಪೂರ್ಣಾಙ್ಕ                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ಯದಿ (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ಧ್ರುವ int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_2d_direct

ಸಂರಚನಾ ggml_tensor * ggml_conv_2d_direct(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ಸಂರಚನಾ ggml_tensor  * b,   // input data [W, H, C, N]
        ಪೂರ್ಣಾಙ್ಕ                   s0,  // stride dimension 0
        ಪೂರ್ಣಾಙ್ಕ                   s1,  // stride dimension 1
        ಪೂರ್ಣಾಙ್ಕ                   p0,  // padding dimension 0
        ಪೂರ್ಣಾಙ್ಕ                   p1,  // padding dimension 1
        ಪೂರ್ಣಾಙ್ಕ                   d0,  // dilation dimension 0
        ಪೂರ್ಣಾಙ್ಕ                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_3d_direct

ಸಂರಚನಾ ggml_tensor * ggml_conv_3d_direct(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಪೂರ್ಣಾಙ್ಕ                   s2,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   p2,
        ಪೂರ್ಣಾಙ್ಕ                   d0,
        ಪೂರ್ಣಾಙ್ಕ                   d1,
        ಪೂರ್ಣಾಙ್ಕ                   d2,
        ಪೂರ್ಣಾಙ್ಕ                   c,
        ಪೂರ್ಣಾಙ್ಕ                   n,
        ಪೂರ್ಣಾಙ್ಕ                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0,  s0);
    ggml_set_op_params_i32(result, 1,  s1);
    ggml_set_op_params_i32(result, 2,  s2);
    ggml_set_op_params_i32(result, 3,  p0);
    ggml_set_op_params_i32(result, 4,  p1);
    ggml_set_op_params_i32(result, 5,  p2);
    ggml_set_op_params_i32(result, 6,  d0);
    ggml_set_op_params_i32(result, 7,  d1);
    ggml_set_op_params_i32(result, 8,  d2);
    ggml_set_op_params_i32(result, 9,  c);
    ggml_set_op_params_i32(result, 10, n);
    ggml_set_op_params_i32(result, 11, oc);

    result->op = GGML_OP_CONV_3D;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_conv_transpose_2d_p0

ಸ್ಥಿರ int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, ಪೂರ್ಣಾಙ್ಕ s, ಪೂರ್ಣಾಙ್ಕ p) {
    ಪ್ರತ್ಯಾಗಮ (ins - 1) * s - 2 * p + ks;
}

ಸಂರಚನಾ ggml_tensor * ggml_conv_transpose_2d_p0(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಪೂರ್ಣಾಙ್ಕ                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ಧ್ರುವ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ಸಂರಚನಾ ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_pool_*

ಸ್ಥಿರ int64_t ggml_calc_pool_output_size(int64_t ins, ಪೂರ್ಣಾಙ್ಕ ks, ಪೂರ್ಣಾಙ್ಕ s, ಭಿನ್ನ p) {
    ಪ್ರತ್ಯಾಗಮ (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ಸಂರಚನಾ ggml_tensor * ggml_pool_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_op_pool     op,
        ಪೂರ್ಣಾಙ್ಕ                   k0,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   p0) {
    ಧ್ರುವ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_pool_2d

ಸಂರಚನಾ ggml_tensor * ggml_pool_2d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_op_pool     op,
        ಪೂರ್ಣಾಙ್ಕ                   k0,
        ಪೂರ್ಣಾಙ್ಕ                   k1,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಭಿನ್ನ                 p0,
        ಭಿನ್ನ                 p1) {
    ಸಂರಚನಾ ggml_tensor * result;
    ಧ್ರುವ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_pool_2d_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * af,
        ಗಣನಾ ggml_op_pool     op,
        ಪೂರ್ಣಾಙ್ಕ                   k0,
        ಪೂರ್ಣಾಙ್ಕ                   k1,
        ಪೂರ್ಣಾಙ್ಕ                   s0,
        ಪೂರ್ಣಾಙ್ಕ                   s1,
        ಭಿನ್ನ                 p0,
        ಭಿನ್ನ                 p1) {
    ಸಂರಚನಾ ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_upscale / ggml_interpolate

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_interpolate_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_ಎಣಿಕೆ);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_upscale(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   scale_factor,
        ಗಣನಾ ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    ಪ್ರತ್ಯಾಗಮ ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

ಸಂರಚನಾ ggml_tensor * ggml_upscale_ext(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   ne0,
        ಪೂರ್ಣಾಙ್ಕ                   ne1,
        ಪೂರ್ಣಾಙ್ಕ                   ne2,
        ಪೂರ್ಣಾಙ್ಕ                   ne3,
        ಗಣನಾ ggml_scale_mode  mode) {
    ಪ್ರತ್ಯಾಗಮ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

ಸಂರಚನಾ ggml_tensor * ggml_interpolate(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    ಪ್ರತ್ಯಾಗಮ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

ಸಂರಚನಾ ggml_tensor * ggml_pad(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   p2,
        ಪೂರ್ಣಾಙ್ಕ                   p3) {
    ಪ್ರತ್ಯಾಗಮ ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ಸಂರಚನಾ ggml_tensor * ggml_pad_circular(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1,
        ಪೂರ್ಣಾಙ್ಕ                   p2,
        ಪೂರ್ಣಾಙ್ಕ                   p3) {
    ಪ್ರತ್ಯಾಗಮ ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ಸಂರಚನಾ ggml_tensor * ggml_pad_ext(
            ಸಂರಚನಾ ggml_context * ctx,
            ಸಂರಚನಾ ggml_tensor  * a,
            ಪೂರ್ಣಾಙ್ಕ                  lp0,
            ಪೂರ್ಣಾಙ್ಕ                  rp0,
            ಪೂರ್ಣಾಙ್ಕ                  lp1,
            ಪೂರ್ಣಾಙ್ಕ                  rp1,
            ಪೂರ್ಣಾಙ್ಕ                  lp2,
            ಪೂರ್ಣಾಙ್ಕ                  rp2,
            ಪೂರ್ಣಾಙ್ಕ                  lp3,
            ಪೂರ್ಣಾಙ್ಕ                  rp3
            ) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_set_op_params_i32(result, 0, lp0);
    ggml_set_op_params_i32(result, 1, rp0);
    ggml_set_op_params_i32(result, 2, lp1);
    ggml_set_op_params_i32(result, 3, rp1);
    ggml_set_op_params_i32(result, 4, lp2);
    ggml_set_op_params_i32(result, 5, rp2);
    ggml_set_op_params_i32(result, 6, lp3);
    ggml_set_op_params_i32(result, 7, rp3);
    ggml_set_op_params_i32(result, 8, 0); // not circular by default


    result->op     = GGML_OP_PAD;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_pad_ext_circular

ಸಂರಚನಾ ggml_tensor * ggml_pad_ext_circular(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                  lp0,
        ಪೂರ್ಣಾಙ್ಕ                  rp0,
        ಪೂರ್ಣಾಙ್ಕ                  lp1,
        ಪೂರ್ಣಾಙ್ಕ                  rp1,
        ಪೂರ್ಣಾಙ್ಕ                  lp2,
        ಪೂರ್ಣಾಙ್ಕ                  rp2,
        ಪೂರ್ಣಾಙ್ಕ                  lp3,
        ಪೂರ್ಣಾಙ್ಕ                  rp3
        ) {
    ಸಂರಚನಾ ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_pad_reflect_1d

ಸಂರಚನಾ ggml_tensor * ggml_pad_reflect_1d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   p0,
        ಪೂರ್ಣಾಙ್ಕ                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_roll

ಸಂರಚನಾ ggml_tensor * ggml_roll(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   shift0,
        ಪೂರ್ಣಾಙ್ಕ                   shift1,
        ಪೂರ್ಣಾಙ್ಕ                   shift2,
        ಪೂರ್ಣಾಙ್ಕ                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_timestep_embedding

ಸಂರಚನಾ ggml_tensor * ggml_timestep_embedding(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * timesteps,
        ಪೂರ್ಣಾಙ್ಕ                   dim,
        ಪೂರ್ಣಾಙ್ಕ                   max_period) {

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_tri

ಸಂರಚನಾ ggml_tensor * ggml_tri(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಗಣನಾ ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_fill

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_fill_impl(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಭಿನ್ನ                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_fill(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಭಿನ್ನ                 c) {
    ಪ್ರತ್ಯಾಗಮ ggml_fill_impl(ctx, a, c, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_fill_inplace(
    ಸಂರಚನಾ ggml_context * ctx,
    ಸಂರಚನಾ ggml_tensor  * a,
    ಭಿನ್ನ                 c) {
    ಪ್ರತ್ಯಾಗಮ ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ಸಂರಚನಾ ggml_tensor * ggml_argsort(
        ಸಂರಚನಾ ggml_context  * ctx,
        ಸಂರಚನಾ ggml_tensor   * a,
        ಗಣನಾ ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_argsort_top_k

ಸಂರಚನಾ ggml_tensor * ggml_argsort_ತುದಿ_k(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ಸಂರಚನಾ ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_top_k

ಸಂರಚನಾ ggml_tensor * ggml_ತುದಿ_k(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_ತುದಿ_K;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_arange

ಸಂರಚನಾ ggml_tensor * ggml_arange(
        ಸಂರಚನಾ ggml_context * ctx,
        ಭಿನ್ನ                 start,
        ಭಿನ್ನ                 stop,
        ಭಿನ್ನ                 step) {
    GGML_ASSERT(stop > start);

    ಧ್ರುವ int64_t steps = (int64_t) ceilf((stop - start) / step);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_flash_attn_ext

ಸಂರಚನಾ ggml_tensor * ggml_flash_attn_ext(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * q,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * mask,
        ಭಿನ್ನ                 scale,
        ಭಿನ್ನ                 max_bias,
        ಭಿನ್ನ                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    ಯದಿ (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    ಯದಿ (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ಭಿನ್ನ params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    ಪ್ರತ್ಯಾಗಮ result;
}


ಶೂನ್ಯ ggml_flash_attn_ext_set_prec(
        ಸಂರಚನಾ ggml_tensor * a,
        ಗಣನಾ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ಧ್ರುವ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

ಗಣನಾ ggml_prec ggml_flash_attn_ext_get_prec(
        ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ಧ್ರುವ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ಪ್ರತ್ಯಾಗಮ (ಗಣನಾ ggml_prec) prec_i32;
}

ಶೂನ್ಯ ggml_flash_attn_ext_set_n_kv_max(
        ಸಂರಚನಾ ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

ಶೂನ್ಯ ggml_flash_attn_ext_add_sinks(
        ಸಂರಚನಾ ggml_tensor * a,
        ಸಂರಚನಾ ggml_tensor * sinks) {
    ಯದಿ (!sinks) {
        a->src[4] = ರಿಕ್ತ;
        ಪ್ರತ್ಯಾಗಮ;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ರಿಕ್ತ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ಸಂರಚನಾ ggml_tensor * ggml_flash_attn_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * q,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ಧ್ರುವ int64_t     D = q->ne[0];
    ಧ್ರುವ int64_t     N = q->ne[1];
    ಧ್ರುವ int64_t     M = k->ne[1];
    ಧ್ರುವ int64_t   ne2 = q->ne[2];
    ಧ್ರುವ int64_t   ne3 = q->ne[3];
    ಧ್ರುವ int64_t kvne2 = k->ne[2];

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
    ಧ್ರುವ int64_t elem_q = ggml_nelements(q);
    ಧ್ರುವ int64_t elem_k = ggml_nelements(k);
    ಧ್ರುವ int64_t elem_v = ggml_nelements(v);

    ಗಣನಾ ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ಧ್ರುವ size_t tsize = ggml_type_size(result_type);

    ಧ್ರುವ size_t offs_q = 0;
    ಧ್ರುವ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ಧ್ರುವ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ಧ್ರುವ size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ಧ್ರುವ size_t nelements = (end + tsize - 1)/tsize;

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, ಪರಿಮಾಣ(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_ssm_conv

ಸಂರಚನಾ ggml_tensor * ggml_ssm_conv(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * sx,
        ಸಂರಚನಾ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ಧ್ರುವ int64_t d_conv  = c->ne[0];
    ಧ್ರುವ int64_t d_inner = c->ne[1];
    ಧ್ರುವ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ಧ್ರುವ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_ssm_scan

ಸಂರಚನಾ ggml_tensor * ggml_ssm_scan(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * s,
        ಸಂರಚನಾ ggml_tensor  * x,
        ಸಂರಚನಾ ggml_tensor  * dt,
        ಸಂರಚನಾ ggml_tensor  * A,
        ಸಂರಚನಾ ggml_tensor  * B,
        ಸಂರಚನಾ ggml_tensor  * C,
        ಸಂರಚನಾ ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_size(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_size(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_size(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ಧ್ರುವ int64_t d_state      = s->ne[0];
        ಧ್ರುವ int64_t head_dim     = x->ne[0];
        ಧ್ರುವ int64_t n_head       = x->ne[1];
        ಧ್ರುವ int64_t n_seq_tokens = x->ne[2];
        ಧ್ರುವ int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_head);
        GGML_ASSERT(dt->ne[1] == n_seq_tokens);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == head_dim);
        GGML_ASSERT(s->ne[2] == n_head);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_tokens);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_vector(ids));
        GGML_ASSERT(A->ne[1] == n_head);
        GGML_ASSERT(ggml_is_matrix(A));

        ಯದಿ (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_win_part

ಸಂರಚನಾ ggml_tensor * ggml_win_part(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ px = (w - a->ne[1]%w)%w;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ py = (w - a->ne[2]%w)%w;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ npx = (px + a->ne[1])/w;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ npy = (py + a->ne[2])/w;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ np  = npx*npy;

    ಧ್ರುವ int64_t ne[4] = { a->ne[0], w, w, np, };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_win_unpart

ಸಂರಚನಾ ggml_tensor * ggml_win_unpart(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   w0,
        ಪೂರ್ಣಾಙ್ಕ                   h0,
        ಪೂರ್ಣಾಙ್ಕ                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_get_rel_pos

ಸಂರಚನಾ ggml_tensor * ggml_get_rel_pos(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಪೂರ್ಣಾಙ್ಕ                   qh,
        ಪೂರ್ಣಾಙ್ಕ                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ಧ್ರುವ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_add_rel_pos

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_add_rel_pos_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * pw,
        ಸಂರಚನಾ ggml_tensor  * ph,
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

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_add_rel_pos(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * pw,
        ಸಂರಚನಾ ggml_tensor  * ph) {
    ಪ್ರತ್ಯಾಗಮ ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_add_rel_pos_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * pw,
        ಸಂರಚನಾ ggml_tensor  * ph) {
    ಪ್ರತ್ಯಾಗಮ ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ಸಂರಚನಾ ggml_tensor * ggml_rwkv_wkv6(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * r,
        ಸಂರಚನಾ ggml_tensor  * tf,
        ಸಂರಚನಾ ggml_tensor  * td,
        ಸಂರಚನಾ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಧ್ರುವ int64_t S = k->ne[0];
    ಧ್ರುವ int64_t H = k->ne[1];
    ಧ್ರುವ int64_t n_tokens = k->ne[2];
    ಧ್ರುವ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಧ್ರುವ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_gated_linear_attn

ಸಂರಚನಾ ggml_tensor * ggml_gated_linear_attn(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * q,
        ಸಂರಚನಾ ggml_tensor  * g,
        ಸಂರಚನಾ ggml_tensor  * state,
        ಭಿನ್ನ scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಧ್ರುವ int64_t S = k->ne[0];
    ಧ್ರುವ int64_t H = k->ne[1];
    ಧ್ರುವ int64_t n_tokens = k->ne[2];
    ಧ್ರುವ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಧ್ರುವ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_rwkv_wkv7

ಸಂರಚನಾ ggml_tensor * ggml_rwkv_wkv7(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * r,
        ಸಂರಚನಾ ggml_tensor  * w,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ಧ್ರುವ int64_t S = k->ne[0];
    ಧ್ರುವ int64_t H = k->ne[1];
    ಧ್ರುವ int64_t n_tokens = k->ne[2];
    ಧ್ರುವ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ಧ್ರುವ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_unary

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_unary_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_unary(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_unary_op    op) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_impl(ctx, a, op, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_unary_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಗಣನಾ ggml_unary_op    op) {
    ಪ್ರತ್ಯಾಗಮ ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_map_custom1_impl(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಧ್ರುವ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಸಂರಚನಾ ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom1(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಧ್ರುವ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom1_inplace(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಧ್ರುವ  ggml_custom1_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_map_custom2_impl(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಧ್ರುವ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಸಂರಚನಾ ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom2(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಧ್ರುವ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom2_inplace(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಧ್ರುವ  ggml_custom2_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_map_custom3_impl(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಸಂರಚನಾ ggml_tensor       * c,
        ಧ್ರುವ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ಸಂರಚನಾ ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ಸಂರಚನಾ ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom3(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಸಂರಚನಾ ggml_tensor       * c,
        ಧ್ರುವ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_map_custom3_inplace(
        ಸಂರಚನಾ ggml_context      * ctx,
        ಸಂರಚನಾ ggml_tensor       * a,
        ಸಂರಚನಾ ggml_tensor       * b,
        ಸಂರಚನಾ ggml_tensor       * c,
        ಧ್ರುವ  ggml_custom3_op_t   fun,
        ಪೂರ್ಣಾಙ್ಕ                        n_tasks,
        ಶೂನ್ಯ                     * userdata) {
    ಪ್ರತ್ಯಾಗಮ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ಸಂರಚನಾ ggml_tensor * ggml_custom_4d(
        ಸಂರಚನಾ ggml_context * ctx,
        ಗಣನಾ ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ಸಂರಚನಾ ggml_tensor ** args,
        ಪೂರ್ಣಾಙ್ಕ                   n_args,
        ggml_custom_op_t      fun,
        ಪೂರ್ಣಾಙ್ಕ                   n_tasks,
        ಶೂನ್ಯ                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ಸಂರಚನಾ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op = GGML_OP_CUSTOM;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_custom_inplace(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor ** args,
        ಪೂರ್ಣಾಙ್ಕ                   n_args,
        ggml_custom_op_t      fun,
        ಪೂರ್ಣಾಙ್ಕ                   n_tasks,
        ಶೂನ್ಯ                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);

    ಸಂರಚನಾ ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ಪರಿಮಾಣ(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    ಪ್ರತ್ಯಾಗಮ result;
}
// ggml_cross_entropy_loss

ಸಂರಚನಾ ggml_tensor * ggml_cross_entropy_loss(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_cross_entropy_loss_back

ಸಂರಚನಾ ggml_tensor * ggml_cross_entropy_loss_back(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        ಸಂರಚನಾ ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ಸಂರಚನಾ ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ಪ್ರತ್ಯಾಗಮ result;
}

// opt_step_adamw

ಸಂರಚನಾ ggml_tensor * ggml_opt_step_adamw(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * grad,
        ಸಂರಚನಾ ggml_tensor  * m,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    ಪ್ರತ್ಯಾಗಮ result;
}

// opt_step_sgd

ಸಂರಚನಾ ggml_tensor * ggml_opt_step_sgd(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * grad,
        ಸಂರಚನಾ ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ಸಂರಚನಾ ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    ಪ್ರತ್ಯಾಗಮ result;
}

// solve_tri

ಸಂರಚನಾ ggml_tensor * ggml_solve_tri(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * a,
        ಸಂರಚನಾ ggml_tensor  * b,
        bool                  ಎಡ,
        bool                  lower,
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

    GGML_ASSERT(lower && ಎಡ && !uni); // TODO: support other variants

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_gated_delta_net

ಸಂರಚನಾ ggml_tensor * ggml_gated_delta_net(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * q,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * v,
        ಸಂರಚನಾ ggml_tensor  * g,
        ಸಂರಚನಾ ggml_tensor  * beta,
        ಸಂರಚನಾ ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_rows(q));
    GGML_ASSERT(ggml_is_contiguous_rows(k));
    GGML_ASSERT(ggml_is_contiguous_rows(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t S_v      = v->ne[0];
    ಧ್ರುವ int64_t H        = v->ne[1];
    ಧ್ರುವ int64_t n_tokens = v->ne[2];
    ಧ್ರುವ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ಧ್ರುವ int64_t state_rows = K * S_v * n_seqs;
    ಧ್ರುವ int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_lightning_indexer

ಸಂರಚನಾ ggml_tensor * ggml_lightning_indexer(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * q,
        ಸಂರಚನಾ ggml_tensor  * k,
        ಸಂರಚನಾ ggml_tensor  * weights,
        ಸಂರಚನಾ ggml_tensor  * mask) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    mask->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   mask->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   mask->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   mask->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % mask->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_dsv4_hc_comb

ಸಂರಚನಾ ggml_tensor * ggml_dsv4_hc_comb(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * mixes,
        ಸಂರಚನಾ ggml_tensor  * scale,
        ಸಂರಚನಾ ggml_tensor  * base,
        ಭಿನ್ನ                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ಧ್ರುವ int64_t hc_mix_dim = mixes->ne[0];
    ಧ್ರುವ int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    ಪ್ರತಿ (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        ಯದಿ ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ಭಙ್ಗ;
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
    GGML_ASSERT(base->ne[0] == hc_mix_dim);
    GGML_ASSERT(base->ne[1] == 1);
    GGML_ASSERT(base->ne[2] == 1);
    GGML_ASSERT(base->ne[3] == 1);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_dsv4_hc_pre

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_dsv4_hc_pre_impl(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * x,
        ಸಂರಚನಾ ggml_tensor  * weights,
        ಭಿನ್ನ                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t n_embd   = x->ne[0];
    ಧ್ರುವ int64_t hc       = x->ne[1];
    ಧ್ರುವ int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    ಯದಿ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_dsv4_hc_pre(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * x,
        ಸಂರಚನಾ ggml_tensor  * weights) {
    ಪ್ರತ್ಯಾಗಮ ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ಸಂರಚನಾ ggml_tensor * ggml_dsv4_hc_pre_gated(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * x,
        ಸಂರಚನಾ ggml_tensor  * gate,
        ಭಿನ್ನ                 scale) {
    ಪ್ರತ್ಯಾಗಮ ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ಸಂರಚನಾ ggml_tensor * ggml_dsv4_hc_post(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * x,
        ಸಂರಚನಾ ggml_tensor  * residual,
        ಸಂರಚನಾ ggml_tensor  * post,
        ಸಂರಚನಾ ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t n_embd   = x->ne[0];
    ಧ್ರುವ int64_t n_tokens = x->ne[1];
    ಧ್ರುವ int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    ಯದಿ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    ಪ್ರತ್ಯಾಗಮ result;
}

// ggml_top_k_mask

ಸಂರಚನಾ ggml_tensor * ggml_ತುದಿ_k_mask(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_tensor  * mask,
        ಸಂರಚನಾ ggml_tensor  * ತುದಿ_k,
        ಸಂರಚನಾ ggml_tensor  * score,
        ಸಂರಚನಾ ggml_tensor  * blocks,
        ಸಂರಚನಾ ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(ತುದಿ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(ತುದಿ_k->ne[1] % mask->ne[1] == 0 && ತುದಿ_k->ne[2] % mask->ne[2] == 0 && ತುದಿ_k->ne[3] % mask->ne[3] == 0);

    ಯದಿ (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == ತುದಿ_k->ne[1] && score->ne[2] == ತುದಿ_k->ne[2] && score->ne[3] == ತುದಿ_k->ne[3]);
    }

    ಯದಿ (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    ಯದಿ (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == ತುದಿ_k->ne[1] && extra->ne[2] == ತುದಿ_k->ne[2] && extra->ne[3] == ತುದಿ_k->ne[3]);
    }

    ಸಂರಚನಾ ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], ತುದಿ_k->ne[1], ತುದಿ_k->ne[2], ತುದಿ_k->ne[3]);

    result->op     = GGML_OP_ತುದಿ_K_MASK;
    result->src[0] = mask;
    result->src[1] = ತುದಿ_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    ಪ್ರತ್ಯಾಗಮ result;
}

////////////////////////////////////////////////////////////////////////////////

ಸಂರಚನಾ ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    ಸಂರಚನಾ ggml_hash_set result;
    result.size = size;
    result.ಕೀಲಿಗಳು = GGML_MALLOC(ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), ಪರಿಮಾಣ(ggml_bitset_t));
    ಪ್ರತ್ಯಾಗಮ result;
}

ಶೂನ್ಯ ggml_hash_set_reset(ಸಂರಚನಾ ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, ಪರಿಮಾಣ(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

ಶೂನ್ಯ ggml_hash_set_free(ಸಂರಚನಾ ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->ಕೀಲಿಗಳು);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    ಸ್ಥಿರ ಧ್ರುವ size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ಸ್ಥಿರ ಧ್ರುವ size_t n_primes = ಪರಿಮಾಣ(primes)/ಪರಿಮಾಣ(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    ಯಾವತ್ (l < r) {
        size_t m = (l + r)/2;
        ಯದಿ (primes[m] < min_sz) {
            l = m + 1;
        } ಅನ್ಯಥಾ {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    ಪ್ರತ್ಯಾಗಮ sz;
}

ಸಂರಚನಾ hash_map {
    ಸಂರಚನಾ ggml_hash_set set;
    ಸಂರಚನಾ ggml_tensor ** vals;
};

ಸ್ಥಿರ ಸಂರಚನಾ hash_map * ggml_new_hash_map(size_t size) {
    ಸಂರಚನಾ hash_map * result = GGML_MALLOC(ಪರಿಮಾಣ(ಸಂರಚನಾ hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hash_map_free(ಸಂರಚನಾ hash_map * map) {
    ggml_hash_set_free(&map->set);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

ಸ್ಥಿರ ಶೂನ್ಯ ggml_add_or_set(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_cgraph  * cgraph,
        size_t                isrc,
        ಸಂರಚನಾ ggml_tensor  * tensor) {
    ಸಂರಚನಾ ggml_tensor * src = cgraph->visited_hash_set.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಯದಿ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } ಅನ್ಯಥಾ {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_acc_or_set(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_cgraph  * cgraph,
        size_t                isrc,
        ಸಂರಚನಾ ggml_tensor  * tensor,
        ಧ್ರುವ  size_t         nb1,
        ಧ್ರುವ  size_t         nb2,
        ಧ್ರುವ  size_t         nb3,
        ಧ್ರುವ  size_t         offset) {
    ಸಂರಚನಾ ggml_tensor * src = cgraph->visited_hash_set.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಯದಿ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.ಕೀಲಿಗಳು[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_add1_or_set(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_cgraph  * cgraph,
        size_t                isrc,
        ಸಂರಚನಾ ggml_tensor  * tensor) {
    ಸಂರಚನಾ ggml_tensor * src = cgraph->visited_hash_set.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಯದಿ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ಅನ್ಯಥಾ {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sub_or_set(
        ಸಂರಚನಾ ggml_context * ctx,
        ಸಂರಚನಾ ggml_cgraph  * cgraph,
        size_t                isrc,
        ಸಂರಚನಾ ggml_tensor  * tensor) {
    ಸಂರಚನಾ ggml_tensor * src = cgraph->visited_hash_set.ಕೀಲಿಗಳು[isrc];
    GGML_ASSERT(src);
    ಯದಿ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ಅನ್ಯಥಾ {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_backward(
        ಸಂರಚನಾ ggml_context * ctx, ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i, ಧ್ರುವ bool * grads_needed) {
    ಸಂರಚನಾ ggml_tensor * tensor = cgraph->nodes[i];
    ಸಂರಚನಾ ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    ಯದಿ (!grad) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = tensor->src[0];
    ಸಂರಚನಾ ggml_tensor * src1 = tensor->src[1];
    ಸಂರಚನಾ ggml_tensor * src2 = tensor->src[2];
    ಸಂರಚನಾ ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ಧ್ರುವ size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ಧ್ರುವ size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ಧ್ರುವ size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ಧ್ರುವ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ಧ್ರುವ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ಧ್ರುವ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    ವಿಕಲ್ಪ (tensor->op) {
        ಪಕ್ಷ GGML_OP_DUP: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ADD: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ಯದಿ (src1_needs_grads) {
                ಸಂರಚನಾ ggml_tensor * tmp = grad;
                ಯದಿ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ADD1: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ಯದಿ (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ACC: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ಯದಿ (src1_needs_grads) {
                ಧ್ರುವ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ಧ್ರುವ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ಧ್ರುವ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ಧ್ರುವ size_t offset = ((int32_t *) tensor->op_params)[3];

                ಸಂರಚನಾ ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUB: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ಯದಿ (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            ಯದಿ (src1_needs_grads) {
                ಸಂರಚನಾ ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                ಯದಿ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DIV: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            ಯದಿ (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SQR: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SQRT: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_LOG: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SIN: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_COS: {
            ಯದಿ (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM: {
            ಯದಿ (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM_ROWS: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MEAN: {
            ಯದಿ (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_REPEAT: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_REPEAT_BACK: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_RMS_NORM: {
            ಯದಿ (src0_needs_grads) {
                ಭಿನ್ನ eps;
                memcpy(&eps, tensor->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT: {
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

            ಯದಿ (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ಸಂರಚನಾ ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                ಯದಿ (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ಧ್ರುವ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ಧ್ರುವ size_t nb2 = tmp->nb[2] * nr2;
                    ಧ್ರುವ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            ಯದಿ (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_out_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SCALE: {
            ಯದಿ (src0_needs_grads) {
                ಭಿನ್ನ s;
                memcpy(&s, tensor->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SET: {
            ಧ್ರುವ size_t nb1    = ((ಧ್ರುವ int32_t *) tensor->op_params)[0];
            ಧ್ರುವ size_t nb2    = ((ಧ್ರುವ int32_t *) tensor->op_params)[1];
            ಧ್ರುವ size_t nb3    = ((ಧ್ರುವ int32_t *) tensor->op_params)[2];
            ಧ್ರುವ size_t offset = ((ಧ್ರುವ int32_t *) tensor->op_params)[3];

            ಸಂರಚನಾ ggml_tensor * tensor_grad_view = ರಿಕ್ತ;

            ಯದಿ (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            ಯದಿ (src0_needs_grads) {
                ಸಂರಚನಾ ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            ಯದಿ (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            ಯದಿ (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            ಯದಿ (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONT: {
            // same as cpy
            ಯದಿ (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_RESHAPE: {
            ಯದಿ (src0_needs_grads) {
                ಸಂರಚನಾ ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_VIEW: {
            ಯದಿ (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, ಪರಿಮಾಣ(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                ಯದಿ (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_size(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_size(src0);
                    GGML_ASSERT(offset % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    offset = (offset / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_set(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, offset);
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_PERMUTE: {
            ಯದಿ (src0_needs_grads) {
                ಧ್ರುವ int32_t * axes = (ಧ್ರುವ int32_t *) tensor->op_params;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis0 = axes[0] & 0x3;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis1 = axes[1] & 0x3;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis2 = axes[2] & 0x3;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis3 = axes[3] & 0x3;
                ಪೂರ್ಣಾಙ್ಕ axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_TRANSPOSE: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GET_ROWS: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            ಯದಿ (src1_needs_grads) {
                // noop
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DIAG_MASK_INF: {
            ಯದಿ (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_past = ((ಧ್ರುವ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_DIAG_MASK_ZERO: {
            ಯದಿ (src0_needs_grads) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_past = ((ಧ್ರುವ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SOFT_MAX: {
            ಯದಿ (src0_needs_grads) {
                ಭಿನ್ನ scale    = 1.0f;
                ಭಿನ್ನ max_bias = 0.0f;

                memcpy(&scale,    (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ROPE: {
            ಯದಿ (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims     = ((ಧ್ರುವ int32_t *) tensor->op_params)[1];
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode       = ((ಧ್ರುವ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig = ((ಧ್ರುವ int32_t *) tensor->op_params)[4];
                ಭಿನ್ನ freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                ಪೂರ್ಣಾಙ್ಕ sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params +  5, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&freq_scale,  (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params +  6, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&ext_factor,  (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params +  7, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&attn_factor, (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params +  8, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&beta_fast,   (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params +  9, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&beta_slow,   (ಧ್ರುವ ಭಿನ್ನ *) tensor->op_params + 10, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&sections,                    tensor->op_params + 11, ಪರಿಮಾಣ(sections));

                ಸಂರಚನಾ ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_IM2COL: {
            ಯದಿ (src1_needs_grads) {
                ಧ್ರುವ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ಧ್ರುವ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ಧ್ರುವ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ಧ್ರುವ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ಧ್ರುವ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ಧ್ರುವ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ಧ್ರುವ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_POOL_2D: {
            ಯದಿ (src0_needs_grads) {
                ಧ್ರುವ ಗಣನಾ ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ಧ್ರುವ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ಧ್ರುವ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ಧ್ರುವ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ಧ್ರುವ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ಧ್ರುವ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ಧ್ರುವ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_WIN_PART:
        ಪಕ್ಷ GGML_OP_WIN_UNPART:
        ಪಕ್ಷ GGML_OP_UNARY: {
            ವಿಕಲ್ಪ (ggml_get_unary_op(tensor)) {
                ಪಕ್ಷ GGML_UNARY_OP_ABS: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_SGN: {
                    // noop
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_NEG: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_STEP: {
                    // noop
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_RELU: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_SILU: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_EXP: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_EXPM1: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ಭಙ್ಗ;
                ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS: {
                    ಯದಿ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ಭಙ್ಗ;
                ಸ್ವಭಾವ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS: {
            ಯದಿ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GLU: {
            ವಿಕಲ್ಪ (ggml_get_glu_op(tensor)) {
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU: {
                    ಯದಿ (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    ಯದಿ (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ಭಙ್ಗ;
                ಸ್ವಭಾವ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_NONE: {
            // noop
        } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ:
        ಸ್ವಭಾವ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ಸ್ಥಿರ size_t ggml_visit_parents_graph(ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, bool compute) {
    ಯದಿ (ಗಂಟು->op != GGML_OP_NONE && compute) {
        ಗಂಟು->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ಧ್ರುವ size_t ಗಂಟು_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು);
    GGML_ASSERT(ಗಂಟು_hash_pos != GGML_HASHSET_FULL);

    ಯದಿ (ggml_bitset_get(cgraph->visited_hash_set.used, ಗಂಟು_hash_pos)) {
        // already visited

        ಯದಿ (compute) {
            // update the compute flag regardless
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; ++i) {
                ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[i];
                ಯದಿ (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಗಂಟು_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.ಕೀಲಿಗಳು[ಗಂಟು_hash_pos] = ಗಂಟು;
    ggml_bitset_set(cgraph->visited_hash_set.used, ಗಂಟು_hash_pos);
    cgraph->use_counts[ಗಂಟು_hash_pos] = 0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ಎಡ_TO_ಬಲ) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ಬಲ_TO_ಎಡ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ಸಂರಚನಾ ggml_tensor * src = ಗಂಟು->src[k];
        ಯದಿ (src) {
            ಧ್ರುವ size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    ಯದಿ (ಗಂಟು->op == GGML_OP_NONE && !(ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        ಯದಿ (strlen(ಗಂಟು->name) == 0) {
            ggml_format_name(ಗಂಟು, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ಗಂಟು;
        cgraph->n_leafs++;
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        ಯದಿ (strlen(ಗಂಟು->name) == 0) {
            ggml_format_name(ಗಂಟು, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ಗಂಟು;
        cgraph->n_nodes++;
    }

    ಪ್ರತ್ಯಾಗಮ ಗಂಟು_hash_pos;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_build_forward_impl(ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_tensor * tensor, bool expand, bool compute) {
    ಯದಿ (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    ಯದಿ (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ಸಂರಚನಾ ggml_tensor * ggml_build_forward_select(
        ಸಂರಚನಾ ggml_cgraph  * cgraph,
        ಸಂರಚನಾ ggml_tensor ** tensors,
        ಪೂರ್ಣಾಙ್ಕ                   n_tensors,
        ಪೂರ್ಣಾಙ್ಕ                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ಪ್ರತ್ಯಾಗಮ tensors[idx];
}

ಶೂನ್ಯ ggml_build_forward_expand(ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

ಶೂನ್ಯ ggml_build_forward_order(ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

ಶೂನ್ಯ ggml_build_backward_expand(
        ಸಂರಚನಾ ggml_context *  ctx,
        ಸಂರಚನಾ ggml_cgraph  *  cgraph,
        ಸಂರಚನಾ ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, ಪರಿಮಾಣ(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_nodes_f; ++i) {
            ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
            any_params = any_params || (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_nodes_f; ++i) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಯದಿ (ಗಂಟು->type == GGML_TYPE_I32) {
            ಅನುವರ್ತ;
        }

        bool ಗಂಟು_needs_grad = (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) || (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ವಿಕಲ್ಪ (ಗಂಟು->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            ಪಕ್ಷ GGML_OP_IM2COL:      // only used for its shape
            ಪಕ್ಷ GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_UNARY: {
                ಧ್ರುವ ಗಣನಾ ggml_unary_op uop = ggml_get_unary_op(ಗಂಟು);
                // SGN and STEP unary ops are piecewise constant
                ಯದಿ (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } ಭಙ್ಗ;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            ಪಕ್ಷ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            ಪಕ್ಷ GGML_OP_GET_ROWS:      // row indices not differentiable
            ಪಕ್ಷ GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            ಪಕ್ಷ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ಭಙ್ಗ;

            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; ++j) {
            ಯದಿ (!ಗಂಟು->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು->src[j])]) {
                ಅನುವರ್ತ;
            }
            GGML_ASSERT(ಗಂಟು->src[j]->type == GGML_TYPE_F32 || ಗಂಟು->src[j]->type == GGML_TYPE_F16);
            ಗಂಟು_needs_grad = true;
            ಭಙ್ಗ;
        }
        ಯದಿ (!ಗಂಟು_needs_grad) {
            ಅನುವರ್ತ;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ಗಂಟು->view_src || ಗಂಟು->op == GGML_OP_CPY || ಗಂಟು->op == GGML_OP_SET_ROWS || ಗಂಟು->op == GGML_OP_VIEW ||
            ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_PERMUTE || ಗಂಟು->op == GGML_OP_TRANSPOSE);

        ಧ್ರುವ size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        ಯದಿ (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ಗಂಟು->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    ಮುಕ್ತ(grads_needed);
}

ಸ್ಥಿರ ಶೂನ್ಯ * incr_ptr_aligned(ಶೂನ್ಯ ** p, size_t size, size_t align) {
    ಶೂನ್ಯ * ptr = *p;
    ptr = (ಶೂನ್ಯ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ಶೂನ್ಯ *) ((ವರ್ಣ *) ptr + size);
    ಪ್ರತ್ಯಾಗಮ ptr;
}

ಸ್ಥಿರ size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    ಶೂನ್ಯ * p = (ವರ್ಣ *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(int32_t), ಪರಿಮಾಣ(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)); // hash keys
    ಯದಿ (grads) {
        incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * ಪರಿಮಾಣ(ggml_bitset_t), ಪರಿಮಾಣ(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ಪ್ರತ್ಯಾಗಮ nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    ಪ್ರತ್ಯಾಗಮ GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

ಸಂರಚನಾ ggml_cgraph * ggml_new_graph_custom(ಸಂರಚನಾ ggml_context * ctx, size_t size, bool grads) {
    ಧ್ರುವ size_t obj_size = ggml_graph_nbytes(size, grads);
    ಸಂರಚನಾ ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    ಸಂರಚನಾ ggml_cgraph * cgraph = (ಸಂರಚನಾ ggml_cgraph *) ((ವರ್ಣ *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    ಶೂನ್ಯ * p = cgraph + 1;

    ಸಂರಚನಾ ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಸಂರಚನಾ ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(int32_t), ಪರಿಮಾಣ(int32_t));
    ಸಂರಚನಾ ggml_tensor ** hash_ಕೀಲಿಗಳು_ptr  =         incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    ಸಂರಚನಾ ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)) : ರಿಕ್ತ;
    ಸಂರಚನಾ ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *), ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *)) : ರಿಕ್ತ;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * ಪರಿಮಾಣ(ggml_bitset_t), ಪರಿಮಾಣ(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ಪ್ರತಿಜ್ಞಾ(obj_size == (size_t)((ವರ್ಣ *)p - (ವರ್ಣ *)cgraph));

    *cgraph = (ಸಂರಚನಾ ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_ಕೀಲಿಗಳು_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_ಎಡ_TO_ಬಲ,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    ಯದಿ (grads) {
        memset(cgraph->grads,     0, hash_size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    }

    ಪ್ರತ್ಯಾಗಮ cgraph;
}

ಸಂರಚನಾ ggml_cgraph * ggml_new_graph(ಸಂರಚನಾ ggml_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

ಸಂರಚನಾ ggml_cgraph ggml_graph_view(ಸಂರಚನಾ ggml_cgraph * cgraph0, ಪೂರ್ಣಾಙ್ಕ i0, ಪೂರ್ಣಾಙ್ಕ i1) {
    ಸಂರಚನಾ ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ರಿಕ್ತ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ರಿಕ್ತ,
        /*.leafs            =*/ ರಿಕ್ತ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    ಪ್ರತ್ಯಾಗಮ cgraph;
}

ಶೂನ್ಯ ggml_graph_cpy(ಸಂರಚನಾ ggml_cgraph * src, ಸಂರಚನಾ ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    ಪ್ರತಿ (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        ಯದಿ (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_ಸೇರಿಸು(&dst->visited_hash_set, src->visited_hash_set.ಕೀಲಿಗಳು[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    ಯದಿ (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_tensor *));
    }
    ಯದಿ (src->grads) {
        GGML_ASSERT(dst->grads     != ರಿಕ್ತ);
        GGML_ASSERT(dst->grad_accs != ರಿಕ್ತ);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < src->n_nodes; ++i) {
            ಧ್ರುವ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ಧ್ರುವ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ಸಂರಚನಾ ggml_cgraph * ggml_graph_dup(ಸಂರಚನಾ ggml_context * ctx, ಸಂರಚನಾ ggml_cgraph * cgraph, bool force_grads) {
    ಸಂರಚನಾ ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸಂರಚನಾ ggml_tensor * ggml_set_zero(ಸಂರಚನಾ ggml_tensor * tensor) {
    ಯದಿ (ggml_is_empty(tensor)) {
        ಪ್ರತ್ಯಾಗಮ tensor;
    }
    ಯದಿ (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    ಪ್ರತ್ಯಾಗಮ tensor;
}

ಶೂನ್ಯ ggml_graph_reset(ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಯದಿ (!cgraph) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    GGML_ASSERT(cgraph->grads != ರಿಕ್ತ);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು     = cgraph->nodes[i];
        ಸಂರಚನಾ ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, ಗಂಟು);

        ಯದಿ (ಗಂಟು->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(ಗಂಟು->src[2]);
            ggml_set_zero(ಗಂಟು->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        ಯದಿ (grad_acc) {
            ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ಧ್ರುವ ಭಿನ್ನ onef = 1.0f;
                ಯದಿ (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
                } ಅನ್ಯಥಾ {
                    GGML_ASSERT(grad_acc->data);
                    *((ಭಿನ್ನ *) grad_acc->data) = onef;
                }
            } ಅನ್ಯಥಾ {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

ಶೂನ್ಯ ggml_graph_clear(ಸಂರಚನಾ ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

ಪೂರ್ಣಾಙ್ಕ ggml_graph_size(ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಪ್ರತ್ಯಾಗಮ cgraph->size;
}

ಸಂರಚನಾ ggml_tensor * ggml_graph_ಗಂಟು(ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i) {
    ಯದಿ (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ಪ್ರತ್ಯಾಗಮ cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ಪ್ರತ್ಯಾಗಮ cgraph->nodes[i];
}

ಸಂರಚನಾ ggml_tensor ** ggml_graph_nodes(ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಪ್ರತ್ಯಾಗಮ cgraph->nodes;
}

ಪೂರ್ಣಾಙ್ಕ ggml_graph_n_nodes(ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಪ್ರತ್ಯಾಗಮ cgraph->n_nodes;
}

ಶೂನ್ಯ ggml_graph_add_ಗಂಟು(ಸಂರಚನಾ ggml_cgraph * cgraph, ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ಸಂರಚನಾ ggml_tensor * ggml_graph_get_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ವರ್ಣ * name) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * leaf = cgraph->leafs[i];

        ಯದಿ (strcmp(leaf->name, name) == 0) {
            ಪ್ರತ್ಯಾಗಮ leaf;
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಯದಿ (strcmp(ಗಂಟು->name, name) == 0) {
            ಪ್ರತ್ಯಾಗಮ ಗಂಟು;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

ಸಂರಚನಾ ggml_tensor * ggml_graph_get_grad(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    ಧ್ರುವ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು);
    ಪ್ರತ್ಯಾಗಮ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : ರಿಕ್ತ;
}

ಸಂರಚನಾ ggml_tensor * ggml_graph_get_grad_acc(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    ಧ್ರುವ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು);
    ಪ್ರತ್ಯಾಗಮ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ರಿಕ್ತ;
}

ಶೂನ್ಯ ggml_graph_print(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ಗಂಟು->ne[0], ಗಂಟು->ne[1], ಗಂಟು->ne[2],
                ggml_op_name(ಗಂಟು->op), (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, ಗಂಟು) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ಗಂಟು->ne[0], ಗಂಟು->ne[1],
                ggml_op_name(ಗಂಟು->op),
                ggml_get_name(ಗಂಟು));
    }

    GGML_LOG_INFO("========================================\n");
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_ಗಂಟು_list_find_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                idxs,
                                      ಪೂರ್ಣಾಙ್ಕ                        ಎಣಿಕೆ,
                                      ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಎಣಿಕೆ; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx = idxs[i];

        ಯದಿ (ಗಂಟು_idx >= cgraph->n_nodes) {
            ಪ್ರತ್ಯಾಗಮ -1;
        }
        ಯದಿ (cgraph->nodes[ಗಂಟು_idx] == tensor) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }
    ಪ್ರತ್ಯಾಗಮ -1;
}

ಸ್ಥಿರ bool ggml_is_constant(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor->buffer != ರಿಕ್ತ && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                ಗಂಟು_idxs,
                                ಪೂರ್ಣಾಙ್ಕ                        ಎಣಿಕೆ,
                                ಧ್ರುವ ಗಣನಾ ggml_op *       ops,
                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                outputs,
                                ಪೂರ್ಣಾಙ್ಕ                        ಅಂಕೆ_outputs) {
    GGML_ASSERT(outputs && ಅಂಕೆ_outputs > 0);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಎಣಿಕೆ; ++i) {
        ಯದಿ (ಗಂಟು_idxs[i] >= cgraph->n_nodes) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[i]];

        ಯದಿ (ಗಂಟು->op != ops[i]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        ಯದಿ (ggml_ಗಂಟು_list_find_tensor(cgraph, outputs, ಅಂಕೆ_outputs, ಗಂಟು) != -1) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        ಪೂರ್ಣಾಙ್ಕ subgraph_uses = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = i + 1; j < ಎಣಿಕೆ; ++j) {
            ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * other_ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[j]];
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ಯದಿ (other_ಗಂಟು->src[src_idx] == ಗಂಟು) {
                    subgraph_uses++;
                }
            }
        }

        ಯದಿ (subgraph_uses != ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idxs[i])) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ಸಂರಚನಾ ggml_tensor * view_src = ಗಂಟು->view_src;
        ಯಾವತ್ (view_src) {
            ಯದಿ (ggml_ಗಂಟು_list_find_tensor(cgraph, ಗಂಟು_idxs, ಎಣಿಕೆ, view_src) == -1 && !ggml_is_constant(view_src)) {
                ಪ್ರತ್ಯಾಗಮ false;
            }
            view_src = view_src->view_src;
        }
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

// check if node is part of the graph
ಸ್ಥಿರ bool ggml_graph_find(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    ಯದಿ (cgraph == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ true;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಯದಿ (cgraph->nodes[i] == ಗಂಟು) {
            ಪ್ರತ್ಯಾಗಮ true;
        }
    }

    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_tensor * ggml_graph_get_parent(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * parent = cgraph->nodes[i];
        ಸಂರಚನಾ ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        ಯದಿ (grad == ಗಂಟು) {
            ಪ್ರತ್ಯಾಗಮ parent;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_graph_dump_dot_ಗಂಟು_edge(FILE * fp, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * gb, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಸಂರಚನಾ ggml_tensor * parent, ಧ್ರುವ ವರ್ಣ * label)  {
    ಸಂರಚನಾ ggml_tensor * gparent = ggml_graph_get_parent(gb, ಗಂಟು);
    ಸಂರಚನಾ ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (ಶೂನ್ಯ *) gparent0 : (ಶೂನ್ಯ *) parent,
            gparent ? (ಶೂನ್ಯ *) gparent : (ಶೂನ್ಯ *) ಗಂಟು,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_graph_dump_dot_leaf_edge(FILE * fp, ಸಂರಚನಾ ggml_tensor * ಗಂಟು, ಸಂರಚನಾ ggml_tensor * parent, ಧ್ರುವ ವರ್ಣ * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (ಶೂನ್ಯ *) parent,
            (ಶೂನ್ಯ *) ಗಂಟು,
            label);
}

ಶೂನ್ಯ ggml_graph_dump_dot(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * gb, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ವರ್ಣ * filename) {
    ವರ್ಣ color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gb->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = gb->nodes[i];
        ಸಂರಚನಾ ggml_tensor * grad = ggml_graph_get_grad(gb, ಗಂಟು);

        ಯದಿ (ggml_graph_get_parent(gb, ಗಂಟು) != ರಿಕ್ತ) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, ಪರಿಮಾಣ(color), "yellow");
        } ಅನ್ಯಥಾ ಯದಿ (grad) {
            ಯದಿ (ggml_graph_find(cgraph, ಗಂಟು)) {
                snprintf(color, ಪರಿಮಾಣ(color), "green");
            } ಅನ್ಯಥಾ {
                snprintf(color, ಪರಿಮಾಣ(color), "lightblue");
            }
        } ಅನ್ಯಥಾ {
            snprintf(color, ಪರಿಮಾಣ(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (ಶೂನ್ಯ *) ಗಂಟು, color);

        ಯದಿ (strlen(ಗಂಟು->name) > 0) {
            fprintf(fp, "%s (%s)|", ಗಂಟು->name, ggml_type_name(ಗಂಟು->type));
        } ಅನ್ಯಥಾ {
            fprintf(fp, "(%s)|", ggml_type_name(ಗಂಟು->type));
        }

        ಯದಿ (ggml_is_matrix(ಗಂಟು)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1], ggml_op_ಚಿಹ್ನೆ(ಗಂಟು->op));
        } ಅನ್ಯಥಾ {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1], ಗಂಟು->ne[2], ggml_op_ಚಿಹ್ನೆ(ಗಂಟು->op));
        }

        ಯದಿ (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_ಚಿಹ್ನೆ(grad->op));
        } ಅನ್ಯಥಾ {
            fprintf(fp, "\"; ]\n");
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gb->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = gb->leafs[i];

        snprintf(color, ಪರಿಮಾಣ(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (ಶೂನ್ಯ *) ಗಂಟು, color);

        ಯದಿ (strlen(ಗಂಟು->name) > 0) {
            fprintf(fp, "%s (%s)|", ಗಂಟು->name, ggml_type_name(ಗಂಟು->type));
        } ಅನ್ಯಥಾ {
            fprintf(fp, "(%s)|", ggml_type_name(ಗಂಟು->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ಗಂಟು->ne[0], ಗಂಟು->ne[1]);
        ಯದಿ (ggml_nelements(ಗಂಟು) < 5 && ಗಂಟು->data != ರಿಕ್ತ) {
            fprintf(fp, " | (");
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ggml_nelements(ಗಂಟು); j++) {
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
                ಯದಿ (j < ggml_nelements(ಗಂಟು) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gb->n_nodes; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = gb->nodes[i];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಯದಿ (ಗಂಟು->src[j]) {
                ವರ್ಣ label[16];
                snprintf(label, ಪರಿಮಾಣ(label), "src %d", j);
                ggml_graph_dump_dot_ಗಂಟು_edge(fp, gb, ಗಂಟು, ಗಂಟು->src[j], label);
            }
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gb->n_leafs; i++) {
        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = gb->leafs[i];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಯದಿ (ಗಂಟು->src[j]) {
                ವರ್ಣ label[16];
                snprintf(label, ಪರಿಮಾಣ(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, ಗಂಟು, ಗಂಟು->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_set_ಒಳಹರಿವು(ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_ಒಳಹರಿವು;
}

ಶೂನ್ಯ ggml_set_output(ಸಂರಚನಾ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಸಂರಚನಾ ggml_tensor * cur = tensor; cur != ರಿಕ್ತ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

ಶೂನ್ಯ ggml_set_param(ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

ಶೂನ್ಯ ggml_set_loss(ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_quantize_init(ಗಣನಾ ggml_type type) {
    ggml_critical_section_start();

    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ಭಙ್ಗ;
        ಸ್ವಭಾವ: // nothing
            ಭಙ್ಗ;
    }

    ggml_critical_section_end();
}

ಶೂನ್ಯ ggml_quantize_free(ಶೂನ್ಯ) {
    ggml_critical_section_start();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_end();
}

bool ggml_quantize_requires_imatrix(ಗಣನಾ ggml_type type) {
    ಪ್ರತ್ಯಾಗಮ
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        ಗಣನಾ ggml_type   type,
           ಧ್ರುವ ಭಿನ್ನ * src,
                  ಶೂನ್ಯ * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ಧ್ರುವ ಭಿನ್ನ * imatrix) {
    ಧ್ರುವ int64_t n = nrows * n_per_row;

    ಯದಿ (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ರಿಕ್ತ);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ಧ್ರುವ size_t start_row = start / n_per_row;
    ಧ್ರುವ size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (ವರ್ಣ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            {
                size_t elemsize = ಪರಿಮಾಣ(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_BF16:
            {
                size_t elemsize = ಪರಿಮಾಣ(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F32:
            {
                size_t elemsize = ಪರಿಮಾಣ(ಭಿನ್ನ);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತಿಜ್ಞಾ(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    ಪ್ರತ್ಯಾಗಮ result;
}

////////////////////////////////////////////////////////////////////////////////

ಶೂನ್ಯ ggml_log_get(ggml_log_callback * log_callback, ಶೂನ್ಯ ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

ಶೂನ್ಯ ggml_log_set(ggml_log_callback log_callback, ಶೂನ್ಯ * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

ಶೂನ್ಯ ggml_threadpool_params_init(ಸಂರಚನಾ ggml_threadpool_params * p, ಪೂರ್ಣಾಙ್ಕ n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

ಸಂರಚನಾ ggml_threadpool_params ggml_threadpool_params_default(ಪೂರ್ಣಾಙ್ಕ n_threads) {
    ಸಂರಚನಾ ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    ಪ್ರತ್ಯಾಗಮ p;
}

bool ggml_threadpool_params_match(ಧ್ರುವ ಸಂರಚನಾ ggml_threadpool_params * p0, ಧ್ರುವ ಸಂರಚನಾ ggml_threadpool_params * p1) {
    ಯದಿ (p0->n_threads  != p1->n_threads  ) ಪ್ರತ್ಯಾಗಮ false;
    ಯದಿ (p0->prio       != p1->prio       ) ಪ್ರತ್ಯಾಗಮ false;
    ಯದಿ (p0->poll       != p1->poll       ) ಪ್ರತ್ಯಾಗಮ false;
    ಯದಿ (p0->strict_cpu != p1->strict_cpu ) ಪ್ರತ್ಯಾಗಮ false;
    ಪ್ರತ್ಯಾಗಮ memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
