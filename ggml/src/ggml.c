#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#അന്തര്ഭാവ "ggml-version.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-threading.h"
#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ggml.h"

// FIXME: required here for quantization functions
#അന്തര്ഭാവ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#അന്തര്ഭാവ <hbwmalloc.h>
#endif

#എങ്കിൽ defined(_MSC_VER) || defined(__MINGW32__)
#അന്തര്ഭാവ <malloc.h> // using malloc.h with MSC/MINGW
#അല്ലെങ്കിലെങ്കിൽ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#അന്തര്ഭാവ <alloca.h>
#endif

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <errno.h>
#അന്തര്ഭാവ <time.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <inttypes.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <float.h>
#അന്തര്ഭാവ <limits.h>
#അന്തര്ഭാവ <stdarg.h>
#അന്തര്ഭാവ <signal.h>
#എങ്കിൽ defined(__gnu_linux__)
#അന്തര്ഭാവ <syscall.h>
#endif

#എങ്കിൽ defined(__APPLE__)
#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ <mach/mach.h>
#അന്തര്ഭാവ <TargetConditionals.h>
#endif

#എങ്കിൽ defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#അന്തര്ഭാവ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(ശൂന്യം) {
#ifdef _MSC_VER
#എങ്കിൽ defined(_WIN32)
    സ്ഥിതം അസ്ഥിര LONG counter = 1;
    തിരികെ (uint64_t) InterlockedIncrement(&counter) - 1;
#അല്ലെങ്കിൽ
    സ്ഥിതം അസ്ഥിര long long counter = 1;
    തിരികെ (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#അല്ലെങ്കിൽ
    സ്ഥിതം uint64_t counter = 1;
    തിരികെ __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#എങ്കിൽ defined(__AVX512BF16__)
#എങ്കിൽ defined(_MSC_VER)
#define m512i(p) p
#അല്ലെങ്കിൽ
#അന്തര്ഭാവ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#എങ്കിൽ defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <sys/stat.h>
#അന്തര്ഭാവ <sys/wait.h>
#എങ്കിൽ defined(__linux__)
#അന്തര്ഭാവ <sys/prctl.h>
#endif

#എങ്കിൽ defined(__ANDROID__)
#അന്തര്ഭാവ <unwind.h>
#അന്തര്ഭാവ <dlfcn.h>
#അന്തര്ഭാവ <stdio.h>

ഘടന backtrace_state {
    ശൂന്യം ** current;
    ശൂന്യം ** end;
};

സ്ഥിതം _Unwind_Reason_Code unwind_callback(ഘടന _Unwind_Context* context, ശൂന്യം* arg) {
    ഘടന backtrace_state * state = (ഘടന backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    എങ്കിൽ (pc) {
        എങ്കിൽ (state->current == state->end) {
            തിരികെ URC_END_OF_അട്ടി;
        } അല്ലെങ്കിൽ {
            *state->current++ = (ശൂന്യം*)pc;
        }
    }
    തിരികെ _URC_NO_REASON;
}

സ്ഥിതം ശൂന്യം ggml_print_backtrace_symbols(ശൂന്യം) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ max = 100;
    ശൂന്യം* buffer[max];

    ഘടന backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    പൂർണ്ണസംഖ്യ എണ്ണം = state.current - buffer;

    വേണ്ടി (പൂർണ്ണസംഖ്യ idx = 0; idx < എണ്ണം; ++idx) {
        സ്ഥിരം ശൂന്യം * addr = buffer[idx];
        സ്ഥിരം അക്ഷരം * ചിഹ്നം = "";

        Dl_info info;
        എങ്കിൽ (dladdr(addr, &info) && info.dli_sname) {
            ചിഹ്നം = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, ചിഹ്നം);
    }
}
#അല്ലെങ്കിലെങ്കിൽ defined(__linux__) && defined(__GLIBC__)
#അന്തര്ഭാവ <execinfo.h>
സ്ഥിതം ശൂന്യം ggml_print_backtrace_symbols(ശൂന്യം) {
    ശൂന്യം * trace[100];
    പൂർണ്ണസംഖ്യ nptrs = backtrace(trace, വലുപ്പം(trace)/വലുപ്പം(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#അല്ലെങ്കിലെങ്കിൽ defined(__APPLE__)
#അന്തര്ഭാവ <execinfo.h>
സ്ഥിതം ശൂന്യം ggml_print_backtrace_symbols(ശൂന്യം) {
    ശൂന്യം * trace[100];
    പൂർണ്ണസംഖ്യ nptrs = backtrace(trace, വലുപ്പം(trace)/വലുപ്പം(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#അല്ലെങ്കിൽ
സ്ഥിതം ശൂന്യം ggml_print_backtrace_symbols(ശൂന്യം) {
    // platform not supported
}
#endif

ശൂന്യം ggml_print_backtrace(ശൂന്യം) {
    സ്ഥിരം അക്ഷരം * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    എങ്കിൽ (GGML_NO_BACKTRACE) {
        തിരികെ;
    }
#എങ്കിൽ defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    സ്ഥിരം അക്ഷരം * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    എങ്കിൽ (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        തിരികെ;
    }
#endif
#എങ്കിൽ defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    അക്ഷരം * വരി = ശൂന്യ;
    ssize_t length = 0;
    ആകുമ്പോൾ ((length = getline(&വരി, &size, f)) > 0) {
        എങ്കിൽ (!strncmp(വരി, "TracerPid:", വലുപ്പം("TracerPid:") - 1) &&
            (length != വലുപ്പം("TracerPid:\t0\n") - 1 || വരി[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            സ്വതന്ത്രം(വരി);
            fclose(f);
            തിരികെ;
        }
    }
    സ്വതന്ത്രം(വരി);
    fclose(f);
    പൂർണ്ണസംഖ്യ lock[2] = { -1, -1 };
    (ശൂന്യം) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    സ്ഥിരം പൂർണ്ണസംഖ്യ parent_pid = getpid();
    സ്ഥിരം പൂർണ്ണസംഖ്യ child_pid = fork();
    എങ്കിൽ (child_pid < 0) { // error
#എങ്കിൽ defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        തിരികെ;
    } അല്ലെങ്കിൽ എങ്കിൽ (child_pid == 0) { // child
        അക്ഷരം attach[32];
        snprintf(attach, വലുപ്പം(attach), "attach %d", parent_pid);
#എങ്കിൽ defined(__linux__)
        close(lock[1]);
        (ശൂന്യം) !പഠ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (അക്ഷരം *) ശൂന്യ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[വലുപ്പം("attach ") - 1],
            (അക്ഷരം *) ശൂന്യ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } അല്ലെങ്കിൽ { // parent
#എങ്കിൽ defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, ശൂന്യ, 0);
    }
}
#അല്ലെങ്കിൽ
ശൂന്യം ggml_print_backtrace(ശൂന്യം) {
    // platform not supported
}
#endif

സ്ഥിതം ggml_abort_callback_t g_abort_callback = ശൂന്യ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    തിരികെ ret_val;
}

ശൂന്യം ggml_abort(സ്ഥിരം അക്ഷരം * file, പൂർണ്ണസംഖ്യ വരി, സ്ഥിരം അക്ഷരം * fmt, ...) {
    fflush(stdout);

    അക്ഷരം message[2048];
    പൂർണ്ണസംഖ്യ offset = snprintf(message, വലുപ്പം(message), "%s:%d: ", file, വരി);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, വലുപ്പം(message) - offset, fmt, args);
    va_end(args);

    എങ്കിൽ (g_abort_callback) {
        g_abort_callback(message);
    } അല്ലെങ്കിൽ {
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

ഘടന ggml_logger_state {
    ggml_log_callback log_callback;
    ശൂന്യം * log_callback_user_data;
};
സ്ഥിതം ഘടന ggml_logger_state g_logger_state = {ggml_log_callback_default, ശൂന്യ};

സ്ഥിതം ശൂന്യം ggml_log_internal_v(എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * format, va_list args) {
    എങ്കിൽ (format == ശൂന്യ) {
        തിരികെ;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    അക്ഷരം buffer[128];
    പൂർണ്ണസംഖ്യ len = vsnprintf(buffer, 128, format, args);
    എങ്കിൽ (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } അല്ലെങ്കിൽ {
        അക്ഷരം * buffer2 = (അക്ഷരം *) calloc(len + 1, വലുപ്പം(അക്ഷരം));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        സ്വതന്ത്രം(buffer2);
    }
    va_end(args_copy);
}

ശൂന്യം ggml_log_internal(എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

ശൂന്യം ggml_log_callback_default(എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * text, ശൂന്യം * user_data) {
    (ശൂന്യം) level;
    (ശൂന്യം) user_data;
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
#എങ്കിൽ defined(__has_feature)
#എങ്കിൽ __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#അല്ലെങ്കിൽ
#എങ്കിൽ defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

ശൂന്യം * ggml_aligned_malloc(size_t size) {
#എങ്കിൽ defined(__s390x__)
    സ്ഥിരം പൂർണ്ണസംഖ്യ alignment = 256;
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ alignment = 64;
#endif

#എങ്കിൽ defined(_MSC_VER) || defined(__MINGW32__)
    തിരികെ _aligned_malloc(size, alignment);
#അല്ലെങ്കിൽ
    എങ്കിൽ (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        തിരികെ ശൂന്യ;
    }
    ശൂന്യം * aligned_memory = ശൂന്യ;
  #ifdef GGML_USE_CPU_HBM
    പൂർണ്ണസംഖ്യ result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #അല്ലെങ്കിലെങ്കിൽ TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    പൂർണ്ണസംഖ്യ result = EFAULT;
    തിരഞ്ഞെടുപ്പ് (alloc_status) {
        അവസ്ഥ KERN_SUCCESS:
            result = 0;
            നിർത്തുക;
        അവസ്ഥ KERN_INVALID_ADDRESS:
            result = EINVAL;
            നിർത്തുക;
        അവസ്ഥ KERN_NO_SPACE:
            result = ENOMEM;
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            result = EFAULT;
            നിർത്തുക;
    }
  #അല്ലെങ്കിൽ
    പൂർണ്ണസംഖ്യ result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    എങ്കിൽ (result != 0) {
        // Handle allocation failure
        സ്ഥിരം അക്ഷരം *error_desc = "unknown allocation error";
        തിരഞ്ഞെടുപ്പ് (result) {
            അവസ്ഥ EINVAL:
                error_desc = "invalid alignment value";
                നിർത്തുക;
            അവസ്ഥ ENOMEM:
                error_desc = "insufficient memory";
                നിർത്തുക;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        തിരികെ ശൂന്യ;
    }
    തിരികെ aligned_memory;
#endif
}

ശൂന്യം ggml_aligned_free(ശൂന്യം * ptr, size_t size) {
    GGML_UNUSED(size);
#എങ്കിൽ defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#അല്ലെങ്കിലെങ്കിൽ GGML_USE_CPU_HBM
    എങ്കിൽ (ptr != ശൂന്യ) {
        hbw_free(ptr);
    }
#അല്ലെങ്കിലെങ്കിൽ TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    എങ്കിൽ (ptr != ശൂന്യ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#അല്ലെങ്കിൽ
    സ്വതന്ത്രം(ptr);
#endif
}


വരിയിൽ സ്ഥിതം ശൂന്യം * ggml_malloc(size_t size) {
    എങ്കിൽ (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        തിരികെ ശൂന്യ;
    }
    ശൂന്യം * result = malloc(size);
    എങ്കിൽ (result == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    തിരികെ result;
}

// calloc
വരിയിൽ സ്ഥിതം ശൂന്യം * ggml_calloc(size_t അക്കം, size_t size) {
    എങ്കിൽ (അക്കം == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        തിരികെ ശൂന്യ;
    }
    ശൂന്യം * result = calloc(അക്കം, size);
    എങ്കിൽ (result == ശൂന്യ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    തിരികെ result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(അക്കം, size) ggml_calloc(അക്കം, size)

#define GGML_FREE(ptr) സ്വതന്ത്രം(ptr)

സ്ഥിരം അക്ഷരം * ggml_status_to_string(എണ്ണൽ ggml_status status) {
    തിരഞ്ഞെടുപ്പ് (status) {
        അവസ്ഥ GGML_STATUS_ALLOC_FAILED: തിരികെ "GGML status: error (failed to allocate memory)";
        അവസ്ഥ GGML_STATUS_FAILED:       തിരികെ "GGML status: error (operation failed)";
        അവസ്ഥ GGML_STATUS_SUCCESS:      തിരികെ "GGML status: success";
        അവസ്ഥ GGML_STATUS_ABORTED:      തിരികെ "GGML status: warning (operation aborted)";
    }

    തിരികെ "GGML status: unknown";
}

ദശാംശം ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    തിരികെ GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(ദശാംശം x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    തിരികെ GGML_FP32_TO_FP16(x);
}

ദശാംശം ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    തിരികെ GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(ദശാംശം x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    തിരികെ GGML_FP32_TO_BF16(x);
}

ശൂന്യം ggml_fp16_to_fp32_row(സ്ഥിരം ggml_fp16_t * x, ദശാംശം * y, int64_t n) {
    വേണ്ടി (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

ശൂന്യം ggml_fp32_to_fp16_row(സ്ഥിരം ദശാംശം * x, ggml_fp16_t * y, int64_t n) {
    പൂർണ്ണസംഖ്യ i = 0;
    വേണ്ടി (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

ശൂന്യം ggml_bf16_to_fp32_row(സ്ഥിരം ggml_bf16_t * x, ദശാംശം * y, int64_t n) {
    പൂർണ്ണസംഖ്യ i = 0;
    വേണ്ടി (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

ശൂന്യം ggml_fp32_to_bf16_row_ref(സ്ഥിരം ദശാംശം * x, ggml_bf16_t * y, int64_t n) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

ശൂന്യം ggml_fp32_to_bf16_row(സ്ഥിരം ദശാംശം * x, ggml_bf16_t * y, int64_t n) {
  പൂർണ്ണസംഖ്യ i = 0;
#എങ്കിൽ defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  വേണ്ടി (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    വേണ്ടി (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    തിരികെ memcmp(guid_a, guid_b, വലുപ്പം(ggml_guid)) == 0;
}

സ്ഥിരം അക്ഷരം * ggml_version(ശൂന്യം) {
    തിരികെ GGML_VERSION;
}

സ്ഥിരം അക്ഷരം * ggml_commit(ശൂന്യം) {
    തിരികെ GGML_COMMIT;
}

//
// timing
//

#എങ്കിൽ defined(_MSC_VER) || defined(__MINGW32__)
സ്ഥിതം int64_t timer_freq, timer_start;
സ്ഥിതം BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    തിരികെ TRUE;
}
ശൂന്യം ggml_time_init(ശൂന്യം) {
    സ്ഥിതം INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, ശൂന്യ, ശൂന്യ);
}
int64_t ggml_time_ms(ശൂന്യം) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    തിരികെ ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(ശൂന്യം) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    തിരികെ ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#അല്ലെങ്കിൽ
ശൂന്യം ggml_time_init(ശൂന്യം) {}
int64_t ggml_time_ms(ശൂന്യം) {
    ഘടന timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    തിരികെ (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(ശൂന്യം) {
    ഘടന timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    തിരികെ (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(ശൂന്യം) {
    തിരികെ clock();
}

int64_t ggml_cycles_per_ms(ശൂന്യം) {
    തിരികെ CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
സ്ഥിതം wchar_t * ggml_mbstowcs(സ്ഥിരം അക്ഷരം * mbs) {
    പൂർണ്ണസംഖ്യ wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, ശൂന്യ, 0);
    എങ്കിൽ (!wlen) {
        errno = EINVAL;
        തിരികെ ശൂന്യ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * വലുപ്പം(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    എങ്കിൽ (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        തിരികെ ശൂന്യ;
    }

    തിരികെ wbuf;
}
#endif

FILE * ggml_fopen(സ്ഥിരം അക്ഷരം * fname, സ്ഥിരം അക്ഷരം * mode) {
#ifdef _WIN32
    FILE * file = ശൂന്യ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    എങ്കിൽ (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        എങ്കിൽ (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    തിരികെ file;
#അല്ലെങ്കിൽ
    തിരികെ fopen(fname, mode);
#endif

}

സ്ഥിതം സ്ഥിരം ഘടന ggml_type_traits type_traits[GGML_TYPE_എണ്ണം] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(ഇരട്ട),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(ദശാംശം),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = വലുപ്പം(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = വലുപ്പം(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = വലുപ്പം(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = വലുപ്പം(block_q4_1),
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
        .type_size                = വലുപ്പം(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = വലുപ്പം(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = വലുപ്പം(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = വലുപ്പം(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = വലുപ്പം(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = വലുപ്പം(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = ശൂന്യ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = ശൂന്യ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = ശൂന്യ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = ശൂന്യ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = വലുപ്പം(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = വലുപ്പം(ggml_bf16_t),
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
        .type_size                = വലുപ്പം(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = വലുപ്പം(block_tq2_0),
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

സ്ഥിരം ഘടന ggml_type_traits * ggml_get_type_traits(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ &type_traits[type];
}

//
// ggml object
//

ഘടന ggml_object {
    size_t offs;
    size_t size;

    ഘടന ggml_object * next;

    എണ്ണൽ ggml_object_type type;

    അക്ഷരം padding[4];
};

സ്ഥിതം സ്ഥിരം size_t GGML_OBJECT_SIZE = വലുപ്പം(ഘടന ggml_object);

//
// ggml context
//

ഘടന ggml_context {
    size_t mem_size;
    ശൂന്യം * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    പൂർണ്ണസംഖ്യ    n_objects;

    ഘടന ggml_object * objects_begin;
    ഘടന ggml_object * objects_end;
};

//
// data types
//

സ്ഥിതം സ്ഥിരം അക്ഷരം * GGML_OP_NAME[GGML_OP_എണ്ണം] = {
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

static_assert(GGML_OP_എണ്ണം == 102, "GGML_OP_COUNT != 102");

സ്ഥിതം സ്ഥിരം അക്ഷരം * GGML_OP_ചിഹ്നം[GGML_OP_എണ്ണം] = {
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

static_assert(GGML_OP_എണ്ണം == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_എണ്ണം == 2, "GGML_OP_POOL_COUNT != 2");

സ്ഥിതം സ്ഥിരം അക്ഷരം * GGML_UNARY_OP_NAME[GGML_UNARY_OP_എണ്ണം] = {
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

static_assert(GGML_UNARY_OP_എണ്ണം == 22, "GGML_UNARY_OP_COUNT != 22");

സ്ഥിതം സ്ഥിരം അക്ഷരം * GGML_GLU_OP_NAME[GGML_GLU_OP_എണ്ണം] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_എണ്ണം == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(വലുപ്പം(ഘടന ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(വലുപ്പം(ഘടന ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

ശൂന്യം ggml_print_object(സ്ഥിരം ഘടന ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (സ്ഥിരം ശൂന്യം *) obj->next);
}

ശൂന്യം ggml_print_objects(സ്ഥിരം ഘടന ggml_context * ctx) {
    ഘടന ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (സ്ഥിരം ശൂന്യം *) ctx);

    ആകുമ്പോൾ (obj != ശൂന്യ) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
        എങ്കിൽ (tensor->ne[i] <= 0) {
            തിരികെ 0;
        }
    }

    size_t nbytes;
    സ്ഥിരം size_t blck_size = ggml_blck_size(tensor->type);
    എങ്കിൽ (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    അല്ലെങ്കിൽ {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    തിരികെ nbytes;
}

size_t ggml_nbytes_pad(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ type_traits[type].blck_size;
}

size_t ggml_type_size(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ type_traits[type].type_size;
}

size_t ggml_row_size(എണ്ണൽ ggml_type type, int64_t ne) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    പ്രതിജ്ഞാ(ne % ggml_blck_size(type) == 0);
    തിരികെ ggml_type_size(type)*ne/ggml_blck_size(type);
}

ഇരട്ട ggml_type_sizef(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ ((ഇരട്ട)(type_traits[type].type_size))/type_traits[type].blck_size;
}

സ്ഥിരം അക്ഷരം * ggml_type_name(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ type_traits[type].type_name;
}

bool ggml_is_quantized(എണ്ണൽ ggml_type type) {
    പ്രതിജ്ഞാ(type >= 0);
    പ്രതിജ്ഞാ(type < GGML_TYPE_എണ്ണം);
    തിരികെ type_traits[type].is_quantized;
}

സ്ഥിരം അക്ഷരം * ggml_op_name(എണ്ണൽ ggml_op op) {
    തിരികെ GGML_OP_NAME[op];
}

സ്ഥിരം അക്ഷരം * ggml_op_ചിഹ്നം(എണ്ണൽ ggml_op op) {
    തിരികെ GGML_OP_ചിഹ്നം[op];
}

സ്ഥിരം അക്ഷരം * ggml_unary_op_name(എണ്ണൽ ggml_unary_op op) {
    തിരികെ GGML_UNARY_OP_NAME[op];
}

സ്ഥിരം അക്ഷരം * ggml_glu_op_name(എണ്ണൽ ggml_glu_op op) {
    തിരികെ GGML_GLU_OP_NAME[op];
}

സ്ഥിരം അക്ഷരം * ggml_op_desc(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (t->op == GGML_OP_UNARY) {
        എണ്ണൽ ggml_unary_op uop = ggml_get_unary_op(t);
        തിരികെ ggml_unary_op_name(uop);
    }
    എങ്കിൽ (t->op == GGML_OP_GLU) {
        എണ്ണൽ ggml_glu_op gop = ggml_get_glu_op(t);
        തിരികെ ggml_glu_op_name(gop);
    }
    തിരികെ ggml_op_name(t->op);
}

size_t ggml_element_size(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_type_size(tensor->type);
}

bool ggml_is_scalar(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ tensor->ne[3] == 1;
}

പൂർണ്ണസംഖ്യ ggml_n_dims(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        എങ്കിൽ (tensor->ne[i] > 1) {
            തിരികെ i + 1;
        }
    }
    തിരികെ 1;
}

എണ്ണൽ ggml_type ggml_ftype_to_ggml_type(എണ്ണൽ ggml_ftype ftype) {
    എണ്ണൽ ggml_type wtype = GGML_TYPE_എണ്ണം;

    തിരഞ്ഞെടുപ്പ് (ftype) {
        അവസ്ഥ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_എണ്ണം; നിർത്തുക;
        അവസ്ഥ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_എണ്ണം; നിർത്തുക;
    }

    GGML_ASSERT(wtype != GGML_TYPE_എണ്ണം);

    തിരികെ wtype;
}

size_t ggml_tensor_overhead(ശൂന്യം) {
    തിരികെ GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ tensor->nb[0] > tensor->nb[1];
}

സ്ഥിതം bool ggml_is_contiguous_m_n(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ m, പൂർണ്ണസംഖ്യ n) {
    size_t next_nb = ggml_type_size(tensor->type);
    എങ്കിൽ (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        തിരികെ false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < n; i++) {
        എങ്കിൽ (i > m) {
            എങ്കിൽ (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                തിരികെ false;
            }
            next_nb *= tensor->ne[i];
        } അല്ലെങ്കിൽ {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    തിരികെ true;
}

bool ggml_is_contiguous(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

സ്ഥിതം വരിയിൽ bool ggml_is_padded_1d(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
        എങ്കിൽ (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            തിരികെ true;
        }
    }
    തിരികെ false;
}

bool ggml_are_same_shape(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(സ്ഥിരം ഘടന ggml_tensor * t) {
    തിരികെ ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

സ്ഥിതം വരിയിൽ bool ggml_can_repeat_rows(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ഘടന ggml_context * ggml_init(ഘടന ggml_init_params params) {
    സ്ഥിതം bool is_first_call = true;

    ggml_critical_section_start();

    എങ്കിൽ (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    ഘടന ggml_context * ctx = GGML_MALLOC(വലുപ്പം(ഘടന ggml_context));

    // allow to call ggml_init with 0 size
    എങ്കിൽ (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    സ്ഥിരം size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (ഘടന ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ശൂന്യ,
        /*.objects_end        =*/ ശൂന്യ,
    };

    GGML_ASSERT(ctx->mem_buffer != ശൂന്യ);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    തിരികെ ctx;
}

ശൂന്യം ggml_reset(ഘടന ggml_context * ctx) {
    എങ്കിൽ (ctx == ശൂന്യ) {
        തിരികെ;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ശൂന്യ;
    ctx->objects_end   = ശൂന്യ;
}

ശൂന്യം ggml_free(ഘടന ggml_context * ctx) {
    എങ്കിൽ (ctx == ശൂന്യ) {
        തിരികെ;
    }

    എങ്കിൽ (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(സ്ഥിരം ഘടന ggml_context * ctx) {
    തിരികെ ctx->objects_end == ശൂന്യ ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(ഘടന ggml_context * ctx) {
    തിരികെ ctx->no_alloc;
}

ശൂന്യം ggml_set_no_alloc(ഘടന ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

ശൂന്യം * ggml_get_mem_buffer(സ്ഥിരം ഘടന ggml_context * ctx) {
    തിരികെ ctx->mem_buffer;
}

size_t ggml_get_mem_size(സ്ഥിരം ഘടന ggml_context * ctx) {
    തിരികെ ctx->mem_size;
}

size_t ggml_get_max_tensor_size(സ്ഥിരം ഘടന ggml_context * ctx) {
    size_t max_size = 0;

    വേണ്ടി (ഘടന ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != ശൂന്യ; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    തിരികെ max_size;
}

////////////////////////////////////////////////////////////////////////////////

സ്ഥിതം ഘടന ggml_object * ggml_new_object(ഘടന ggml_context * ctx, എണ്ണൽ ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    ഘടന ggml_object * obj_cur = ctx->objects_end;

    സ്ഥിരം size_t cur_offs = obj_cur == ശൂന്യ ? 0 : obj_cur->offs;
    സ്ഥിരം size_t cur_size = obj_cur == ശൂന്യ ? 0 : obj_cur->size;
    സ്ഥിരം size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    അക്ഷരം * സ്ഥിരം mem_buffer = ctx->mem_buffer;
    ഘടന ggml_object * സ്ഥിരം obj_new = (ഘടന ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    എങ്കിൽ (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        തിരികെ ശൂന്യ;
    }
    എങ്കിൽ (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        തിരികെ ശൂന്യ;
    }

    എങ്കിൽ (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        തിരികെ ശൂന്യ;
    }

    *obj_new = (ഘടന ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = ശൂന്യ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    എങ്കിൽ (obj_cur != ശൂന്യ) {
        obj_cur->next = obj_new;
    } അല്ലെങ്കിൽ {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    തിരികെ obj_new;
}

സ്ഥിതം ഘടന ggml_tensor * ggml_new_tensor_impl(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type      type,
        പൂർണ്ണസംഖ്യ                   n_dims,
        സ്ഥിരം int64_t       * ne,
        ഘടന ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_എണ്ണം);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    എങ്കിൽ (view_src != ശൂന്യ && view_src->view_src != ശൂന്യ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < n_dims; i++) {
        എങ്കിൽ (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == ശൂന്യ || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    ശൂന്യം * data = view_src != ശൂന്യ ? view_src->data : ശൂന്യ;
    എങ്കിൽ (data != ശൂന്യ) {
        data = (അക്ഷരം *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    എങ്കിൽ (view_src == ശൂന്യ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    ഘടന ggml_object * സ്ഥിരം obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    ഘടന ggml_tensor * സ്ഥിരം result = (ഘടന ggml_tensor *)((അക്ഷരം *)ctx->mem_buffer + obj_new->offs);

    *result = (ഘടന ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ശൂന്യ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ശൂന്യ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (ശൂന്യം *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ശൂന്യ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_new_tensor(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type      type,
        പൂർണ്ണസംഖ്യ                   n_dims,
        സ്ഥിരം int64_t       * ne) {
    തിരികെ ggml_new_tensor_impl(ctx, type, n_dims, ne, ശൂന്യ, 0);
}

ഘടന ggml_tensor * ggml_new_tensor_1d(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type      type,
        int64_t ne0) {
    തിരികെ ggml_new_tensor(ctx, type, 1, &ne0);
}

ഘടന ggml_tensor * ggml_new_tensor_2d(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    സ്ഥിരം int64_t ne[2] = { ne0, ne1 };
    തിരികെ ggml_new_tensor(ctx, type, 2, ne);
}

ഘടന ggml_tensor * ggml_new_tensor_3d(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    സ്ഥിരം int64_t ne[3] = { ne0, ne1, ne2 };
    തിരികെ ggml_new_tensor(ctx, type, 3, ne);
}

ഘടന ggml_tensor * ggml_new_tensor_4d(
        ഘടന ggml_context * ctx,
        എണ്ണൽ   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    സ്ഥിരം int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    തിരികെ ggml_new_tensor(ctx, type, 4, ne);
}

ശൂന്യം * ggml_new_buffer(ഘടന ggml_context * ctx, size_t nbytes) {
    ഘടന ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    തിരികെ (uint8_t *)ctx->mem_buffer + obj->offs;
}

ഘടന ggml_tensor * ggml_dup_tensor(ഘടന ggml_context * ctx, സ്ഥിരം ഘടന ggml_tensor * src) {
    തിരികെ ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

ശൂന്യം ggml_unravel_index(സ്ഥിരം ഘടന ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    സ്ഥിരം int64_t ne2 = tensor->ne[2];
    സ്ഥിരം int64_t ne1 = tensor->ne[1];
    സ്ഥിരം int64_t ne0 = tensor->ne[0];

    സ്ഥിരം int64_t i3_ = (i/(ne2*ne1*ne0));
    സ്ഥിരം int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    സ്ഥിരം int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    സ്ഥിരം int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    എങ്കിൽ (i0) {
        * i0 = i0_;
    }
    എങ്കിൽ (i1) {
        * i1 = i1_;
    }
    എങ്കിൽ (i2) {
        * i2 = i2_;
    }
    എങ്കിൽ (i3) {
        * i3 = i3_;
    }
}

ശൂന്യം * ggml_get_data(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ tensor->data;
}

ദശാംശം * ggml_get_data_f32(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    പ്രതിജ്ഞാ(tensor->type == GGML_TYPE_F32);
    തിരികെ (ദശാംശം *)(tensor->data);
}

എണ്ണൽ ggml_unary_op ggml_get_unary_op(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    തിരികെ (എണ്ണൽ ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

എണ്ണൽ ggml_glu_op ggml_get_glu_op(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    തിരികെ (എണ്ണൽ ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

സ്ഥിരം അക്ഷരം * ggml_get_name(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ tensor->name;
}

ഘടന ggml_tensor * ggml_set_name(ഘടന ggml_tensor * tensor, സ്ഥിരം അക്ഷരം * name) {
    size_t i;
    വേണ്ടി (i = 0; i < വലുപ്പം(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    തിരികെ tensor;
}

ഘടന ggml_tensor * ggml_format_name(ഘടന ggml_tensor * tensor, സ്ഥിരം അക്ഷരം * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, വലുപ്പം(tensor->name), fmt, args);
    va_end(args);
    തിരികെ tensor;
}

ഘടന ggml_tensor * ggml_view_tensor(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * src) {
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_get_first_tensor(സ്ഥിരം ഘടന ggml_context * ctx) {
    ഘടന ggml_object * obj = ctx->objects_begin;

    അക്ഷരം * സ്ഥിരം mem_buffer = ctx->mem_buffer;

    ആകുമ്പോൾ (obj != ശൂന്യ) {
        എങ്കിൽ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            തിരികെ (ഘടന ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    തിരികെ ശൂന്യ;
}

ഘടന ggml_tensor * ggml_get_next_tensor(സ്ഥിരം ഘടന ggml_context * ctx, ഘടന ggml_tensor * tensor) {
    ഘടന ggml_object * obj = (ഘടന ggml_object *) ((അക്ഷരം *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    അക്ഷരം * സ്ഥിരം mem_buffer = ctx->mem_buffer;

    ആകുമ്പോൾ (obj != ശൂന്യ) {
        എങ്കിൽ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            തിരികെ (ഘടന ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    തിരികെ ശൂന്യ;
}

ഘടന ggml_tensor * ggml_get_tensor(ഘടന ggml_context * ctx, സ്ഥിരം അക്ഷരം * name) {
    ഘടന ggml_object * obj = ctx->objects_begin;

    അക്ഷരം * സ്ഥിരം mem_buffer = ctx->mem_buffer;

    ആകുമ്പോൾ (obj != ശൂന്യ) {
        എങ്കിൽ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ഘടന ggml_tensor * cur = (ഘടന ggml_tensor *)(mem_buffer + obj->offs);
            എങ്കിൽ (strcmp(cur->name, name) == 0) {
                തിരികെ cur;
            }
        }

        obj = obj->next;
    }

    തിരികെ ശൂന്യ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

സ്ഥിതം ഘടന ggml_tensor * ggml_dup_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_dup(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_dup_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_dup_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_dup_impl(ctx, a, true);
}

// ggml_add

സ്ഥിതം ഘടന ggml_tensor * ggml_add_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_add(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_add_impl(ctx, a, b, false);
}

ഘടന ggml_tensor * ggml_add_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

സ്ഥിതം ഘടന ggml_tensor * ggml_add_cast_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        എണ്ണൽ   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_add_cast(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        എണ്ണൽ   ggml_type      type) {
    തിരികെ ggml_add_cast_impl(ctx, a, b, type);
}

ഘടന ggml_tensor * ggml_add_id(
            ഘടന ggml_context * ctx,
            ഘടന ggml_tensor  * a,
            ഘടന ggml_tensor  * b,
            ഘടന ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    തിരികെ result;
}

// ggml_add1

സ്ഥിതം ഘടന ggml_tensor * ggml_add1_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_add1(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_add1_impl(ctx, a, b, false);
}

ഘടന ggml_tensor * ggml_add1_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

സ്ഥിതം ഘടന ggml_tensor * ggml_acc_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_acc(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    തിരികെ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ഘടന ggml_tensor * ggml_acc_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    തിരികെ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

സ്ഥിതം ഘടന ggml_tensor * ggml_sub_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_sub(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_sub_impl(ctx, a, b, false);
}

ഘടന ggml_tensor * ggml_sub_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

സ്ഥിതം ഘടന ggml_tensor * ggml_mul_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_mul(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_mul_impl(ctx, a, b, false);
}

ഘടന ggml_tensor * ggml_mul_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

സ്ഥിതം ഘടന ggml_tensor * ggml_div_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_div(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_div_impl(ctx, a, b, false);
}

ഘടന ggml_tensor * ggml_div_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

സ്ഥിതം ഘടന ggml_tensor * ggml_sqr_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_sqr(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sqr_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_sqr_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

സ്ഥിതം ഘടന ggml_tensor * ggml_sqrt_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_sqrt(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sqrt_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_sqrt_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

സ്ഥിതം ഘടന ggml_tensor * ggml_log_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_log(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_log_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_log_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_log_impl(ctx, a, true);
}

ഘടന ggml_tensor * ggml_expm1(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ഘടന ggml_tensor * ggml_expm1_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ഘടന ggml_tensor * ggml_softplus(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ഘടന ggml_tensor * ggml_softplus_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

സ്ഥിതം ഘടന ggml_tensor * ggml_sin_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_sin(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sin_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_sin_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_sin_impl(ctx, a, true);
}

// ggml_cos

സ്ഥിതം ഘടന ggml_tensor * ggml_cos_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_cos(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_cos_impl(ctx, a, false);
}

ഘടന ggml_tensor * ggml_cos_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ഘടന ggml_tensor * ggml_sum(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_sum_rows

ഘടന ggml_tensor * ggml_sum_rows(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_cumsum

ഘടന ggml_tensor * ggml_cumsum(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_mean

ഘടന ggml_tensor * ggml_mean(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_argmax

ഘടന ggml_tensor * ggml_argmax(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_count_equal

ഘടന ggml_tensor * ggml_എണ്ണം_equal(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_എണ്ണം_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_repeat

ഘടന ggml_tensor * ggml_repeat(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_repeat_4d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    സ്ഥിരം bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_repeat_back

ഘടന ggml_tensor * ggml_repeat_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_concat

ഘടന ggml_tensor * ggml_concat(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ഘടന ggml_tensor  * b,
    പൂർണ്ണസംഖ്യ                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < GGML_MAX_DIMS; ++d) {
        എങ്കിൽ (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            തുടരുക;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_abs

ഘടന ggml_tensor * ggml_abs(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ഘടന ggml_tensor * ggml_abs_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ഘടന ggml_tensor * ggml_sgn(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ഘടന ggml_tensor * ggml_sgn_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ഘടന ggml_tensor * ggml_neg(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ഘടന ggml_tensor * ggml_neg_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ഘടന ggml_tensor * ggml_step(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

ഘടന ggml_tensor * ggml_step_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

ഘടന ggml_tensor * ggml_tanh(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ഘടന ggml_tensor * ggml_tanh_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ഘടന ggml_tensor * ggml_elu(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ഘടന ggml_tensor * ggml_elu_inplace(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ഘടന ggml_tensor * ggml_relu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ഘടന ggml_tensor * ggml_relu_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ഘടന ggml_tensor * ggml_leaky_relu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 negative_slope,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, വലുപ്പം(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_sigmoid

ഘടന ggml_tensor * ggml_sigmoid(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ഘടന ggml_tensor * ggml_sigmoid_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ഘടന ggml_tensor * ggml_gelu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ഘടന ggml_tensor * ggml_gelu_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ഘടന ggml_tensor * ggml_gelu_erf(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ഘടന ggml_tensor * ggml_gelu_erf_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ഘടന ggml_tensor * ggml_gelu_quick(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

ഘടന ggml_tensor * ggml_gelu_quick_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

ഘടന ggml_tensor * ggml_silu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ഘടന ggml_tensor * ggml_silu_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ഘടന ggml_tensor * ggml_xielu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം alpha_n,
        ദശാംശം alpha_p,
        ദശാംശം beta,
        ദശാംശം eps) {
    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_silu_back

ഘടന ggml_tensor * ggml_silu_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml hardswish

ഘടന ggml_tensor * ggml_hardswish(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ഘടന ggml_tensor * ggml_hardsigmoid(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ഘടന ggml_tensor * ggml_exp(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ഘടന ggml_tensor * ggml_exp_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

സ്ഥിതം ഘടന ggml_tensor * ggml_glu_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        എണ്ണൽ ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    എങ്കിൽ (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ശൂന്യ, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_floor

ഘടന ggml_tensor * ggml_floor(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ഘടന ggml_tensor * ggml_floor_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ഘടന ggml_tensor * ggml_ceil(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ഘടന ggml_tensor * ggml_ceil_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ഘടന ggml_tensor * ggml_round(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

ഘടന ggml_tensor * ggml_round_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

ഘടന ggml_tensor * ggml_trunc(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ഘടന ggml_tensor * ggml_trunc_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ഘടന ggml_tensor * ggml_glu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_glu_op      op,
        bool                  swapped) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, op, swapped);
}

ഘടന ggml_tensor * ggml_glu_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        എണ്ണൽ ggml_glu_op      op) {
    തിരികെ ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ഘടന ggml_tensor * ggml_reglu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_REGLU, false);
}

ഘടന ggml_tensor * ggml_reglu_swapped(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_REGLU, true);
}

ഘടന ggml_tensor * ggml_reglu_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ഘടന ggml_tensor * ggml_geglu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU, false);
}

ഘടന ggml_tensor * ggml_geglu_swapped(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU, true);
}

ഘടന ggml_tensor * ggml_geglu_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ഘടന ggml_tensor * ggml_swiglu(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_SWIGLU, false);
}

ഘടന ggml_tensor * ggml_swiglu_swapped(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_SWIGLU, true);
}

ഘടന ggml_tensor * ggml_swiglu_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ഘടന ggml_tensor * ggml_geglu_erf(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU_ERF, false);
}

ഘടന ggml_tensor * ggml_geglu_erf_swapped(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU_ERF, true);
}

ഘടന ggml_tensor * ggml_geglu_erf_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ഘടന ggml_tensor * ggml_geglu_quick(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU_QUICK, false);
}

ഘടന ggml_tensor * ggml_geglu_quick_swapped(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_glu_impl(ctx, a, ശൂന്യ, GGML_GLU_OP_GEGLU_QUICK, true);
}

ഘടന ggml_tensor * ggml_geglu_quick_split(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

ഘടന ggml_tensor * ggml_swiglu_oai(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 alpha,
        ദശാംശം                 limit) {
    ഘടന ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_swiglu_clamp(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 limit) {
    ഘടന ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    തിരികെ result;
}

// ggml_norm

സ്ഥിതം ഘടന ggml_tensor * ggml_norm_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, വലുപ്പം(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_norm(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_norm_impl(ctx, a, eps, false);
}

ഘടന ggml_tensor * ggml_norm_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

സ്ഥിതം ഘടന ggml_tensor * ggml_rms_norm_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, വലുപ്പം(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_rms_norm(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_rms_norm_impl(ctx, a, eps, false);
}

ഘടന ggml_tensor * ggml_rms_norm_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ഘടന ggml_tensor * ggml_rms_norm_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 eps) {
    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, വലുപ്പം(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_group_norm

സ്ഥിതം ഘടന ggml_tensor * ggml_group_norm_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_groups,
        ദശാംശം                 eps,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_group_norm(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_groups,
        ദശാംശം                 eps) {
    തിരികെ ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

ഘടന ggml_tensor * ggml_group_norm_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_groups,
        ദശാംശം                 eps) {
    തിരികെ ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

സ്ഥിതം ഘടന ggml_tensor * ggml_l2_norm_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_l2_norm(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_l2_norm_impl(ctx, a, eps, false);
}

ഘടന ggml_tensor * ggml_l2_norm_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 eps) {
    തിരികെ ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        ഘടന ggml_tensor * a,
        എണ്ണൽ ggml_prec       prec) {
    തിരഞ്ഞെടുപ്പ് (a->op) {
        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            {
                സ്ഥിരം int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            {
                സ്ഥിരം int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            തിരികെ false;
    };

    തിരികെ true;
}

bool ggml_prec_set_src(
        ഘടന ggml_tensor * a,
        എണ്ണൽ ggml_prec       prec,
        പൂർണ്ണസംഖ്യ                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    തിരഞ്ഞെടുപ്പ് (a->op) {
        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            {
                എങ്കിൽ (idx != 1) {
                    തിരികെ false;
                }

                സ്ഥിരം int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            തിരികെ false;
    };

    തിരികെ true;
}

// ggml_mul_mat

സ്ഥിതം വരിയിൽ bool ggml_can_mul_mat(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ഘടന ggml_tensor * ggml_mul_mat(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    സ്ഥിരം int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ശൂന്യം ggml_mul_mat_set_prec(
        ഘടന ggml_tensor * a,
        എണ്ണൽ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    സ്ഥിരം int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

ശൂന്യം ggml_mul_mat_set_hint(
        ഘടന ggml_tensor * a,
        എണ്ണൽ ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    സ്ഥിരം int32_t hint_i32 = (int32_t) hint;

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
ഘടന ggml_tensor * ggml_mul_mat_id(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * as,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    സ്ഥിരം int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    തിരികെ result;
}

// ggml_out_prod

സ്ഥിതം വരിയിൽ bool ggml_can_out_prod(സ്ഥിരം ഘടന ggml_tensor * t0, സ്ഥിരം ഘടന ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ഘടന ggml_tensor * ggml_out_prod(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    സ്ഥിരം int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_scale

സ്ഥിതം ഘടന ggml_tensor * ggml_scale_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 s,
        ദശാംശം                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ദശാംശം params[2] = { s, b };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_scale(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 s) {
    തിരികെ ggml_scale_impl(ctx, a, s, 0.0, false);
}

ഘടന ggml_tensor * ggml_scale_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 s) {
    തിരികെ ggml_scale_impl(ctx, a, s, 0.0, true);
}

ഘടന ggml_tensor * ggml_scale_bias(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 s,
        ദശാംശം                 b) {
    തിരികെ ggml_scale_impl(ctx, a, s, b, false);
}

ഘടന ggml_tensor * ggml_scale_bias_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 s,
        ദശാംശം                 b) {
    തിരികെ ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

സ്ഥിതം ഘടന ggml_tensor * ggml_set_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_set(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ഘടന ggml_tensor * ggml_set_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

ഘടന ggml_tensor * ggml_set_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

ഘടന ggml_tensor * ggml_set_1d_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

ഘടന ggml_tensor * ggml_set_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

ഘടന ggml_tensor * ggml_set_2d_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    തിരികെ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

സ്ഥിതം ഘടന ggml_tensor * ggml_cpy_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, b);
    എങ്കിൽ (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } അല്ലെങ്കിൽ {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_cpy(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor * a,
        ഘടന ggml_tensor * b) {
    തിരികെ ggml_cpy_impl(ctx, a, b);
}

ഘടന ggml_tensor * ggml_cast(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ   ggml_type      type) {
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    തിരികെ result;
}

// ggml_cont

സ്ഥിതം ഘടന ggml_tensor * ggml_cont_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_cont(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor * a) {
    തിരികെ ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ഘടന ggml_tensor * ggml_cont_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0) {
    തിരികെ ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ഘടന ggml_tensor * ggml_cont_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    തിരികെ ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ഘടന ggml_tensor * ggml_cont_3d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    തിരികെ ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ഘടന ggml_tensor * ggml_cont_4d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_reshape

ഘടന ggml_tensor * ggml_reshape(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor * a,
        ഘടന ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_reshape_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    സ്ഥിരം int64_t ne[1] = { ne0 };
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_reshape_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    സ്ഥിരം int64_t ne[2] = { ne0, ne1 };
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_reshape_3d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    സ്ഥിരം int64_t ne[3] = { ne0, ne1, ne2 };
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_reshape_4d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    സ്ഥിരം int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    തിരികെ result;
}

സ്ഥിതം ഘടന ggml_tensor * ggml_view_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_dims,
        സ്ഥിരം int64_t       * ne,
        size_t                offset) {
    ഘടന ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, വലുപ്പം(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_view_1d

ഘടന ggml_tensor * ggml_view_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    ഘടന ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    തിരികെ result;
}

// ggml_view_2d

ഘടന ggml_tensor * ggml_view_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    സ്ഥിരം int64_t ne[2] = { ne0, ne1 };

    ഘടന ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    തിരികെ result;
}

// ggml_view_3d

ഘടന ggml_tensor * ggml_view_3d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    സ്ഥിരം int64_t ne[3] = { ne0, ne1, ne2 };

    ഘടന ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    തിരികെ result;
}

// ggml_view_4d

ഘടന ggml_tensor * ggml_view_4d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    സ്ഥിരം int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ഘടന ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    തിരികെ result;
}

// ggml_permute

ഘടന ggml_tensor * ggml_permute(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   axis0,
        പൂർണ്ണസംഖ്യ                   axis1,
        പൂർണ്ണസംഖ്യ                   axis2,
        പൂർണ്ണസംഖ്യ                   axis3) {
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

    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, വലുപ്പം(params));

    തിരികെ result;
}

// ggml_transpose

ഘടന ggml_tensor * ggml_transpose(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_get_rows

ഘടന ggml_tensor * ggml_get_rows(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    എണ്ണൽ ggml_type type = GGML_TYPE_F32;
    എങ്കിൽ (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_get_rows_back

ഘടന ggml_tensor * ggml_get_rows_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ഘടന ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_set_rows

ഘടന ggml_tensor * ggml_set_rows(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c) {
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

    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    തിരികെ result;
}

// ggml_diag

ഘടന ggml_tensor * ggml_diag(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    സ്ഥിരം int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_diag_mask_inf

സ്ഥിതം ഘടന ggml_tensor * ggml_diag_mask_inf_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_diag_mask_inf(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past) {
    തിരികെ ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

ഘടന ggml_tensor * ggml_diag_mask_inf_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past) {
    തിരികെ ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

സ്ഥിതം ഘടന ggml_tensor * ggml_diag_mask_zero_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_diag_mask_zero(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past) {
    തിരികെ ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

ഘടന ggml_tensor * ggml_diag_mask_zero_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_past) {
    തിരികെ ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

സ്ഥിതം ഘടന ggml_tensor * ggml_clamp_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ദശാംശം                 min,
        ദശാംശം                 max,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ദശാംശം params[] = { min, max };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_clamp(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ദശാംശം                 min,
    ദശാംശം                 max) {
    തിരികെ ggml_clamp_impl(ctx, a, min, max, false);
}

ഘടന ggml_tensor * ggml_clamp_inplace(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ദശാംശം                 min,
    ദശാംശം                 max) {
    തിരികെ ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

സ്ഥിതം ഘടന ggml_tensor * ggml_soft_max_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * mask,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    എങ്കിൽ (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    എങ്കിൽ (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ദശാംശം params[] = { scale, max_bias };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_soft_max(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_soft_max_impl(ctx, a, ശൂന്യ, 1.0f, 0.0f, false);
}

ഘടന ggml_tensor * ggml_soft_max_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a) {
    തിരികെ ggml_soft_max_impl(ctx, a, ശൂന്യ, 1.0f, 0.0f, true);
}

ഘടന ggml_tensor * ggml_soft_max_ext(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * mask,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias) {
    തിരികെ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

ഘടന ggml_tensor * ggml_soft_max_ext_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * mask,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias) {
    തിരികെ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

ശൂന്യം ggml_soft_max_add_sinks(
        ഘടന ggml_tensor * a,
        ഘടന ggml_tensor * sinks) {
    എങ്കിൽ (!sinks) {
        a->src[2] = ശൂന്യ;
        തിരികെ;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ശൂന്യ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

സ്ഥിതം ഘടന ggml_tensor * ggml_soft_max_ext_back_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias,
        bool                  inplace) {
    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((ദശാംശം *) result->op_params + 0, &scale,    വലുപ്പം(ദശാംശം));
    memcpy((ദശാംശം *) result->op_params + 1, &max_bias, വലുപ്പം(ദശാംശം));

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_soft_max_ext_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias) {
    തിരികെ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

ഘടന ggml_tensor * ggml_soft_max_ext_back_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias) {
    തിരികെ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

സ്ഥിതം ഘടന ggml_tensor * ggml_rope_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   sections[GGML_MROPE_SECTIONS],
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    എങ്കിൽ (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    എങ്കിൽ (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    വലുപ്പം(ദശാംശം));
    memcpy(params +  6, &freq_scale,   വലുപ്പം(ദശാംശം));
    memcpy(params +  7, &ext_factor,   വലുപ്പം(ദശാംശം));
    memcpy(params +  8, &attn_factor,  വലുപ്പം(ദശാംശം));
    memcpy(params +  9, &beta_fast,    വലുപ്പം(ദശാംശം));
    memcpy(params + 10, &beta_slow,    വലുപ്പം(ദശാംശം));
    എങ്കിൽ (mrope_used && sections) {
        memcpy(params + 11, sections,  വലുപ്പം(int32_t) * GGML_MROPE_SECTIONS);
    } അല്ലെങ്കിൽ {
        memset(params + 11, 0,         വലുപ്പം(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_rope(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, ശൂന്യ, n_dims, ശൂന്യ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ഘടന ggml_tensor * ggml_rope_multi(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   sections[GGML_MROPE_SECTIONS],
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ഘടന ggml_tensor * ggml_rope_multi_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   sections[GGML_MROPE_SECTIONS],
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ഘടന ggml_tensor * ggml_rope_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, ശൂന്യ, n_dims, ശൂന്യ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ഘടന ggml_tensor * ggml_rope_ext(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, c, n_dims, ശൂന്യ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ഘടന ggml_tensor * ggml_rope_ext_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, c, n_dims, ശൂന്യ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ഘടന ggml_tensor * ggml_rope_custom(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, ശൂന്യ, n_dims, ശൂന്യ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ഘടന ggml_tensor * ggml_rope_custom_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    തിരികെ ggml_rope_impl(
        ctx, a, b, ശൂന്യ, n_dims, ശൂന്യ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
സ്ഥിതം ദശാംശം ggml_rope_yarn_corr_dim(പൂർണ്ണസംഖ്യ n_dims, പൂർണ്ണസംഖ്യ n_ctx_orig, ദശാംശം n_rot, ദശാംശം base) {
    തിരികെ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ദശാംശം)M_PI)) / (2 * logf(base));
}

ശൂന്യം ggml_rope_yarn_corr_dims(
    പൂർണ്ണസംഖ്യ n_dims, പൂർണ്ണസംഖ്യ n_ctx_orig, ദശാംശം freq_base, ദശാംശം beta_fast, ദശാംശം beta_slow, ദശാംശം dims[2]
) {
    // start and end correction dims
    ദശാംശം start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ദശാംശം end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

ഘടന ggml_tensor * ggml_rope_ext_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    ഘടന ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    തിരികെ result;
}

ഘടന ggml_tensor * ggml_rope_multi_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c,
        പൂർണ്ണസംഖ്യ                   n_dims,
        പൂർണ്ണസംഖ്യ                   sections[4],
        പൂർണ്ണസംഖ്യ                   mode,
        പൂർണ്ണസംഖ്യ                   n_ctx_orig,
        ദശാംശം                 freq_base,
        ദശാംശം                 freq_scale,
        ദശാംശം                 ext_factor,
        ദശാംശം                 attn_factor,
        ദശാംശം                 beta_fast,
        ദശാംശം                 beta_slow) {
    ഘടന ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    തിരികെ result;
}

ഘടന ggml_tensor * ggml_rope_set_offset(
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    സ്ഥിരം int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    തിരികെ a;
}

സ്ഥിതം int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, പൂർണ്ണസംഖ്യ s, പൂർണ്ണസംഖ്യ p, പൂർണ്ണസംഖ്യ d) {
    തിരികെ (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ഘടന ggml_tensor * ggml_im2col(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   d0,
        പൂർണ്ണസംഖ്യ                   d1,
        bool                  is_2D,
        എണ്ണൽ ggml_type        dst_type) {
    എങ്കിൽ (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } അല്ലെങ്കിൽ {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    സ്ഥിരം int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    സ്ഥിരം int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    സ്ഥിരം int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_im2col_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        int64_t             * ne,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   d0,
        പൂർണ്ണസംഖ്യ                   d1,
        bool                  is_2D) {
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_conv_1d

ഘടന ggml_tensor * ggml_conv_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   d0) {
    ഘടന ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ഘടന ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    തിരികെ result;
}

// ggml_conv_1d_ph

ഘടന ggml_tensor* ggml_conv_1d_ph(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s,
        പൂർണ്ണസംഖ്യ                   d) {
    തിരികെ ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ഘടന ggml_tensor * ggml_conv_1d_dw(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   d0) {
    ഘടന ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ഘടന ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ഘടന ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    തിരികെ result;
}

// ggml_conv_1d_dw_ph

ഘടന ggml_tensor * ggml_conv_1d_dw_ph(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   d0) {
    തിരികെ ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ഘടന ggml_tensor * ggml_col2im_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   oc,
        പൂർണ്ണസംഖ്യ                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    സ്ഥിരം int64_t K_OC = a->ne[0];
    സ്ഥിരം int64_t T_in = a->ne[1];
    സ്ഥിരം int64_t K = K_OC / oc;
    സ്ഥിരം int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    സ്ഥിരം int64_t ne[4] = { T_out, oc, 1, 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_conv_transpose_1d

സ്ഥിതം int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, പൂർണ്ണസംഖ്യ s, പൂർണ്ണസംഖ്യ p, പൂർണ്ണസംഖ്യ d) {
    തിരികെ (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ഘടന ggml_tensor * ggml_conv_transpose_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    സ്ഥിരം int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ഘടന ggml_tensor * ggml_conv_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   d0,
        പൂർണ്ണസംഖ്യ                   d1) {
    ഘടന ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ഘടന ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    തിരികെ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ഘടന ggml_tensor * ggml_im2col_3d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        int64_t               IC,
        പൂർണ്ണസംഖ്യ                   s0, // stride width
        പൂർണ്ണസംഖ്യ                   s1, // stride height
        പൂർണ്ണസംഖ്യ                   s2, // stride depth
        പൂർണ്ണസംഖ്യ                   p0, // padding width
        പൂർണ്ണസംഖ്യ                   p1, // padding height
        പൂർണ്ണസംഖ്യ                   p2, // padding depth
        പൂർണ്ണസംഖ്യ                   d0, // dilation width
        പൂർണ്ണസംഖ്യ                   d1, // dilation height
        പൂർണ്ണസംഖ്യ                   d2, // dilation depth
        എണ്ണൽ ggml_type        dst_type) {
    സ്ഥിരം int64_t N = b->ne[3] / IC;
    സ്ഥിരം int64_t ID = b->ne[2];
    സ്ഥിരം int64_t IH = b->ne[1];
    സ്ഥിരം int64_t IW = b->ne[0];

    സ്ഥിരം int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    സ്ഥിരം int64_t KD = a->ne[2];
    സ്ഥിരം int64_t KH = a->ne[1];
    സ്ഥിരം int64_t KW = a->ne[0];
    സ്ഥിരം int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    സ്ഥിരം int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    സ്ഥിരം int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    സ്ഥിരം int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ഘടന ggml_tensor * ggml_conv_3d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        int64_t               IC,
        പൂർണ്ണസംഖ്യ                   s0, // stride width
        പൂർണ്ണസംഖ്യ                   s1, // stride height
        പൂർണ്ണസംഖ്യ                   s2, // stride depth
        പൂർണ്ണസംഖ്യ                   p0, // padding width
        പൂർണ്ണസംഖ്യ                   p1, // padding height
        പൂർണ്ണസംഖ്യ                   p2, // padding depth
        പൂർണ്ണസംഖ്യ                   d0, // dilation width
        പൂർണ്ണസംഖ്യ                   d1, // dilation height
        പൂർണ്ണസംഖ്യ                   d2  // dilation depth
        ) {
    ഘടന ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ഘടന ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    തിരികെ result;
}

// ggml_conv_2d_sk_p0

ഘടന ggml_tensor * ggml_conv_2d_sk_p0(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ഘടന ggml_tensor * ggml_conv_2d_s1_ph(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    തിരികെ ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ഘടന ggml_tensor * ggml_conv_2d_dw(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   d0,
        പൂർണ്ണസംഖ്യ                   d1) {
    ഘടന ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ഘടന ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ഘടന ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ഘടന ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    തിരികെ result;
}

// ggml_conv_2d_dw_direct

ഘടന ggml_tensor * ggml_conv_2d_dw_direct(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   stride0,
        പൂർണ്ണസംഖ്യ                   stride1,
        പൂർണ്ണസംഖ്യ                   pad0,
        പൂർണ്ണസംഖ്യ                   pad1,
        പൂർണ്ണസംഖ്യ                   dilation0,
        പൂർണ്ണസംഖ്യ                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    എങ്കിൽ (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        സ്ഥിരം int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    തിരികെ result;
}

// ggml_conv_2d_direct

ഘടന ggml_tensor * ggml_conv_2d_direct(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ഘടന ggml_tensor  * b,   // input data [W, H, C, N]
        പൂർണ്ണസംഖ്യ                   s0,  // stride dimension 0
        പൂർണ്ണസംഖ്യ                   s1,  // stride dimension 1
        പൂർണ്ണസംഖ്യ                   p0,  // padding dimension 0
        പൂർണ്ണസംഖ്യ                   p1,  // padding dimension 1
        പൂർണ്ണസംഖ്യ                   d0,  // dilation dimension 0
        പൂർണ്ണസംഖ്യ                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_conv_3d_direct

ഘടന ggml_tensor * ggml_conv_3d_direct(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        പൂർണ്ണസംഖ്യ                   s2,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   p2,
        പൂർണ്ണസംഖ്യ                   d0,
        പൂർണ്ണസംഖ്യ                   d1,
        പൂർണ്ണസംഖ്യ                   d2,
        പൂർണ്ണസംഖ്യ                   c,
        പൂർണ്ണസംഖ്യ                   n,
        പൂർണ്ണസംഖ്യ                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    തിരികെ result;
}

// ggml_conv_transpose_2d_p0

സ്ഥിതം int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, പൂർണ്ണസംഖ്യ s, പൂർണ്ണസംഖ്യ p) {
    തിരികെ (ins - 1) * s - 2 * p + ks;
}

ഘടന ggml_tensor * ggml_conv_transpose_2d_p0(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        പൂർണ്ണസംഖ്യ                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    സ്ഥിരം int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ഘടന ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_pool_*

സ്ഥിതം int64_t ggml_calc_pool_output_size(int64_t ins, പൂർണ്ണസംഖ്യ ks, പൂർണ്ണസംഖ്യ s, ദശാംശം p) {
    തിരികെ (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ഘടന ggml_tensor * ggml_pool_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_op_pool     op,
        പൂർണ്ണസംഖ്യ                   k0,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   p0) {
    സ്ഥിരം int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_pool_2d

ഘടന ggml_tensor * ggml_pool_2d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_op_pool     op,
        പൂർണ്ണസംഖ്യ                   k0,
        പൂർണ്ണസംഖ്യ                   k1,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        ദശാംശം                 p0,
        ദശാംശം                 p1) {
    ഘടന ggml_tensor * result;
    സ്ഥിരം int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_pool_2d_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * af,
        എണ്ണൽ ggml_op_pool     op,
        പൂർണ്ണസംഖ്യ                   k0,
        പൂർണ്ണസംഖ്യ                   k1,
        പൂർണ്ണസംഖ്യ                   s0,
        പൂർണ്ണസംഖ്യ                   s1,
        ദശാംശം                 p0,
        ദശാംശം                 p1) {
    ഘടന ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    തിരികെ result;
}

// ggml_upscale / ggml_interpolate

സ്ഥിതം ഘടന ggml_tensor * ggml_interpolate_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_എണ്ണം);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_upscale(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   scale_factor,
        എണ്ണൽ ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    തിരികെ ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

ഘടന ggml_tensor * ggml_upscale_ext(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   ne0,
        പൂർണ്ണസംഖ്യ                   ne1,
        പൂർണ്ണസംഖ്യ                   ne2,
        പൂർണ്ണസംഖ്യ                   ne3,
        എണ്ണൽ ggml_scale_mode  mode) {
    തിരികെ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

ഘടന ggml_tensor * ggml_interpolate(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    തിരികെ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

ഘടന ggml_tensor * ggml_pad(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   p2,
        പൂർണ്ണസംഖ്യ                   p3) {
    തിരികെ ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ഘടന ggml_tensor * ggml_pad_circular(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1,
        പൂർണ്ണസംഖ്യ                   p2,
        പൂർണ്ണസംഖ്യ                   p3) {
    തിരികെ ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ഘടന ggml_tensor * ggml_pad_ext(
            ഘടന ggml_context * ctx,
            ഘടന ggml_tensor  * a,
            പൂർണ്ണസംഖ്യ                  lp0,
            പൂർണ്ണസംഖ്യ                  rp0,
            പൂർണ്ണസംഖ്യ                  lp1,
            പൂർണ്ണസംഖ്യ                  rp1,
            പൂർണ്ണസംഖ്യ                  lp2,
            പൂർണ്ണസംഖ്യ                  rp2,
            പൂർണ്ണസംഖ്യ                  lp3,
            പൂർണ്ണസംഖ്യ                  rp3
            ) {
    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    തിരികെ result;
}

// ggml_pad_ext_circular

ഘടന ggml_tensor * ggml_pad_ext_circular(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                  lp0,
        പൂർണ്ണസംഖ്യ                  rp0,
        പൂർണ്ണസംഖ്യ                  lp1,
        പൂർണ്ണസംഖ്യ                  rp1,
        പൂർണ്ണസംഖ്യ                  lp2,
        പൂർണ്ണസംഖ്യ                  rp2,
        പൂർണ്ണസംഖ്യ                  lp3,
        പൂർണ്ണസംഖ്യ                  rp3
        ) {
    ഘടന ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    തിരികെ result;
}

// ggml_pad_reflect_1d

ഘടന ggml_tensor * ggml_pad_reflect_1d(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   p0,
        പൂർണ്ണസംഖ്യ                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_roll

ഘടന ggml_tensor * ggml_roll(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   shift0,
        പൂർണ്ണസംഖ്യ                   shift1,
        പൂർണ്ണസംഖ്യ                   shift2,
        പൂർണ്ണസംഖ്യ                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_timestep_embedding

ഘടന ggml_tensor * ggml_timestep_embedding(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * timesteps,
        പൂർണ്ണസംഖ്യ                   dim,
        പൂർണ്ണസംഖ്യ                   max_period) {

    ഘടന ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    തിരികെ result;
}

// ggml_tri

ഘടന ggml_tensor * ggml_tri(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    എണ്ണൽ ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_fill

സ്ഥിതം ഘടന ggml_tensor * ggml_fill_impl(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ദശാംശം                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_fill(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ദശാംശം                 c) {
    തിരികെ ggml_fill_impl(ctx, a, c, false);
}

ഘടന ggml_tensor * ggml_fill_inplace(
    ഘടന ggml_context * ctx,
    ഘടന ggml_tensor  * a,
    ദശാംശം                 c) {
    തിരികെ ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ഘടന ggml_tensor * ggml_argsort(
        ഘടന ggml_context  * ctx,
        ഘടന ggml_tensor   * a,
        എണ്ണൽ ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_argsort_top_k

ഘടന ggml_tensor * ggml_argsort_മുകൾ_k(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ഘടന ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    തിരികെ result;
}

// ggml_top_k

ഘടന ggml_tensor * ggml_മുകൾ_k(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_മുകൾ_K;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_arange

ഘടന ggml_tensor * ggml_arange(
        ഘടന ggml_context * ctx,
        ദശാംശം                 start,
        ദശാംശം                 stop,
        ദശാംശം                 step) {
    GGML_ASSERT(stop > start);

    സ്ഥിരം int64_t steps = (int64_t) ceilf((stop - start) / step);

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    തിരികെ result;
}

// ggml_flash_attn_ext

ഘടന ggml_tensor * ggml_flash_attn_ext(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * q,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * mask,
        ദശാംശം                 scale,
        ദശാംശം                 max_bias,
        ദശാംശം                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    എങ്കിൽ (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    എങ്കിൽ (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ദശാംശം params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    തിരികെ result;
}


ശൂന്യം ggml_flash_attn_ext_set_prec(
        ഘടന ggml_tensor * a,
        എണ്ണൽ ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    സ്ഥിരം int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

എണ്ണൽ ggml_prec ggml_flash_attn_ext_get_prec(
        സ്ഥിരം ഘടന ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    സ്ഥിരം int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    തിരികെ (എണ്ണൽ ggml_prec) prec_i32;
}

ശൂന്യം ggml_flash_attn_ext_set_n_kv_max(
        ഘടന ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

ശൂന്യം ggml_flash_attn_ext_add_sinks(
        ഘടന ggml_tensor * a,
        ഘടന ggml_tensor * sinks) {
    എങ്കിൽ (!sinks) {
        a->src[4] = ശൂന്യ;
        തിരികെ;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ശൂന്യ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ഘടന ggml_tensor * ggml_flash_attn_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * q,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    സ്ഥിരം int64_t     D = q->ne[0];
    സ്ഥിരം int64_t     N = q->ne[1];
    സ്ഥിരം int64_t     M = k->ne[1];
    സ്ഥിരം int64_t   ne2 = q->ne[2];
    സ്ഥിരം int64_t   ne3 = q->ne[3];
    സ്ഥിരം int64_t kvne2 = k->ne[2];

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
    സ്ഥിരം int64_t elem_q = ggml_nelements(q);
    സ്ഥിരം int64_t elem_k = ggml_nelements(k);
    സ്ഥിരം int64_t elem_v = ggml_nelements(v);

    എണ്ണൽ ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    സ്ഥിരം size_t tsize = ggml_type_size(result_type);

    സ്ഥിരം size_t offs_q = 0;
    സ്ഥിരം size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    സ്ഥിരം size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    സ്ഥിരം size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    സ്ഥിരം size_t nelements = (end + tsize - 1)/tsize;

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, വലുപ്പം(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    തിരികെ result;
}

// ggml_ssm_conv

ഘടന ggml_tensor * ggml_ssm_conv(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * sx,
        ഘടന ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    സ്ഥിരം int64_t d_conv  = c->ne[0];
    സ്ഥിരം int64_t d_inner = c->ne[1];
    സ്ഥിരം int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    സ്ഥിരം int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ഘടന ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    തിരികെ result;
}

// ggml_ssm_scan

ഘടന ggml_tensor * ggml_ssm_scan(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * s,
        ഘടന ggml_tensor  * x,
        ഘടന ggml_tensor  * dt,
        ഘടന ggml_tensor  * A,
        ഘടന ggml_tensor  * B,
        ഘടന ggml_tensor  * C,
        ഘടന ggml_tensor  * ids,
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
        സ്ഥിരം int64_t d_state      = s->ne[0];
        സ്ഥിരം int64_t head_dim     = x->ne[0];
        സ്ഥിരം int64_t n_head       = x->ne[1];
        സ്ഥിരം int64_t n_seq_tokens = x->ne[2];
        സ്ഥിരം int64_t n_seqs       = x->ne[3];

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

        എങ്കിൽ (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    തിരികെ result;
}

// ggml_win_part

ഘടന ggml_tensor * ggml_win_part(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    സ്ഥിരം പൂർണ്ണസംഖ്യ px = (w - a->ne[1]%w)%w;
    സ്ഥിരം പൂർണ്ണസംഖ്യ py = (w - a->ne[2]%w)%w;

    സ്ഥിരം പൂർണ്ണസംഖ്യ npx = (px + a->ne[1])/w;
    സ്ഥിരം പൂർണ്ണസംഖ്യ npy = (py + a->ne[2])/w;
    സ്ഥിരം പൂർണ്ണസംഖ്യ np  = npx*npy;

    സ്ഥിരം int64_t ne[4] = { a->ne[0], w, w, np, };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_win_unpart

ഘടന ggml_tensor * ggml_win_unpart(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   w0,
        പൂർണ്ണസംഖ്യ                   h0,
        പൂർണ്ണസംഖ്യ                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, വലുപ്പം(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_get_rel_pos

ഘടന ggml_tensor * ggml_get_rel_pos(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        പൂർണ്ണസംഖ്യ                   qh,
        പൂർണ്ണസംഖ്യ                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    സ്ഥിരം int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    തിരികെ result;
}

// ggml_add_rel_pos

സ്ഥിതം ഘടന ggml_tensor * ggml_add_rel_pos_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * pw,
        ഘടന ggml_tensor  * ph,
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

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_add_rel_pos(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * pw,
        ഘടന ggml_tensor  * ph) {
    തിരികെ ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

ഘടന ggml_tensor * ggml_add_rel_pos_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * pw,
        ഘടന ggml_tensor  * ph) {
    തിരികെ ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ഘടന ggml_tensor * ggml_rwkv_wkv6(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * r,
        ഘടന ggml_tensor  * tf,
        ഘടന ggml_tensor  * td,
        ഘടന ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    സ്ഥിരം int64_t S = k->ne[0];
    സ്ഥിരം int64_t H = k->ne[1];
    സ്ഥിരം int64_t n_tokens = k->ne[2];
    സ്ഥിരം int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    സ്ഥിരം int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    തിരികെ result;
}

// ggml_gated_linear_attn

ഘടന ggml_tensor * ggml_gated_linear_attn(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * q,
        ഘടന ggml_tensor  * g,
        ഘടന ggml_tensor  * state,
        ദശാംശം scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    സ്ഥിരം int64_t S = k->ne[0];
    സ്ഥിരം int64_t H = k->ne[1];
    സ്ഥിരം int64_t n_tokens = k->ne[2];
    സ്ഥിരം int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    സ്ഥിരം int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    തിരികെ result;
}

// ggml_rwkv_wkv7

ഘടന ggml_tensor * ggml_rwkv_wkv7(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * r,
        ഘടന ggml_tensor  * w,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    സ്ഥിരം int64_t S = k->ne[0];
    സ്ഥിരം int64_t H = k->ne[1];
    സ്ഥിരം int64_t n_tokens = k->ne[2];
    സ്ഥിരം int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    സ്ഥിരം int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    തിരികെ result;
}

// ggml_unary

സ്ഥിതം ഘടന ggml_tensor * ggml_unary_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_unary(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_unary_op    op) {
    തിരികെ ggml_unary_impl(ctx, a, op, false);
}

ഘടന ggml_tensor * ggml_unary_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        എണ്ണൽ ggml_unary_op    op) {
    തിരികെ ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

സ്ഥിതം ഘടന ggml_tensor * ggml_map_custom1_impl(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        സ്ഥിരം  ggml_custom1_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ഘടന ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_map_custom1(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        സ്ഥിരം  ggml_custom1_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ഘടന ggml_tensor * ggml_map_custom1_inplace(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        സ്ഥിരം  ggml_custom1_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

സ്ഥിതം ഘടന ggml_tensor * ggml_map_custom2_impl(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        സ്ഥിരം  ggml_custom2_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ഘടന ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_map_custom2(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        സ്ഥിരം  ggml_custom2_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ഘടന ggml_tensor * ggml_map_custom2_inplace(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        സ്ഥിരം  ggml_custom2_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

സ്ഥിതം ഘടന ggml_tensor * ggml_map_custom3_impl(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        ഘടന ggml_tensor       * c,
        സ്ഥിരം  ggml_custom3_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ഘടന ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ഘടന ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_map_custom3(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        ഘടന ggml_tensor       * c,
        സ്ഥിരം  ggml_custom3_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ഘടന ggml_tensor * ggml_map_custom3_inplace(
        ഘടന ggml_context      * ctx,
        ഘടന ggml_tensor       * a,
        ഘടന ggml_tensor       * b,
        ഘടന ggml_tensor       * c,
        സ്ഥിരം  ggml_custom3_op_t   fun,
        പൂർണ്ണസംഖ്യ                        n_tasks,
        ശൂന്യം                     * userdata) {
    തിരികെ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ഘടന ggml_tensor * ggml_custom_4d(
        ഘടന ggml_context * ctx,
        എണ്ണൽ ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ഘടന ggml_tensor ** args,
        പൂർണ്ണസംഖ്യ                   n_args,
        ggml_custom_op_t      fun,
        പൂർണ്ണസംഖ്യ                   n_tasks,
        ശൂന്യം                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ഘടന ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op = GGML_OP_CUSTOM;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_custom_inplace(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor ** args,
        പൂർണ്ണസംഖ്യ                   n_args,
        ggml_custom_op_t      fun,
        പൂർണ്ണസംഖ്യ                   n_tasks,
        ശൂന്യം                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);

    ഘടന ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, വലുപ്പം(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    തിരികെ result;
}
// ggml_cross_entropy_loss

ഘടന ggml_tensor * ggml_cross_entropy_loss(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_cross_entropy_loss_back

ഘടന ggml_tensor * ggml_cross_entropy_loss_back(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        ഘടന ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ഘടന ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    തിരികെ result;
}

// opt_step_adamw

ഘടന ggml_tensor * ggml_opt_step_adamw(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * grad,
        ഘടന ggml_tensor  * m,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    തിരികെ result;
}

// opt_step_sgd

ഘടന ggml_tensor * ggml_opt_step_sgd(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * grad,
        ഘടന ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ഘടന ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    തിരികെ result;
}

// solve_tri

ഘടന ggml_tensor * ggml_solve_tri(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * a,
        ഘടന ggml_tensor  * b,
        bool                  ഇടത്,
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

    GGML_ASSERT(lower && ഇടത് && !uni); // TODO: support other variants

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    തിരികെ result;
}

// ggml_gated_delta_net

ഘടന ggml_tensor * ggml_gated_delta_net(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * q,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * v,
        ഘടന ggml_tensor  * g,
        ഘടന ggml_tensor  * beta,
        ഘടന ggml_tensor  * state,
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

    സ്ഥിരം int64_t S_v      = v->ne[0];
    സ്ഥിരം int64_t H        = v->ne[1];
    സ്ഥിരം int64_t n_tokens = v->ne[2];
    സ്ഥിരം int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    സ്ഥിരം int64_t state_rows = K * S_v * n_seqs;
    സ്ഥിരം int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    തിരികെ result;
}

// ggml_lightning_indexer

ഘടന ggml_tensor * ggml_lightning_indexer(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * q,
        ഘടന ggml_tensor  * k,
        ഘടന ggml_tensor  * weights,
        ഘടന ggml_tensor  * mask) {

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
    ഘടന ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    തിരികെ result;
}

// ggml_dsv4_hc_comb

ഘടന ggml_tensor * ggml_dsv4_hc_comb(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * mixes,
        ഘടന ggml_tensor  * scale,
        ഘടന ggml_tensor  * base,
        ദശാംശം                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    സ്ഥിരം int64_t hc_mix_dim = mixes->ne[0];
    സ്ഥിരം int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    വേണ്ടി (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        എങ്കിൽ ((2 + i)*i == hc_mix_dim) {
            hc = i;
            നിർത്തുക;
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

    ഘടന ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    തിരികെ result;
}

// ggml_dsv4_hc_pre

സ്ഥിതം ഘടന ggml_tensor * ggml_dsv4_hc_pre_impl(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * x,
        ഘടന ggml_tensor  * weights,
        ദശാംശം                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t hc       = x->ne[1];
    സ്ഥിരം int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    എങ്കിൽ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ഘടന ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_dsv4_hc_pre(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * x,
        ഘടന ggml_tensor  * weights) {
    തിരികെ ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ഘടന ggml_tensor * ggml_dsv4_hc_pre_gated(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * x,
        ഘടന ggml_tensor  * gate,
        ദശാംശം                 scale) {
    തിരികെ ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ഘടന ggml_tensor * ggml_dsv4_hc_post(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * x,
        ഘടന ggml_tensor  * residual,
        ഘടന ggml_tensor  * post,
        ഘടന ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t n_tokens = x->ne[1];
    സ്ഥിരം int64_t hc       = residual->ne[1];

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

    എങ്കിൽ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ഘടന ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    തിരികെ result;
}

// ggml_top_k_mask

ഘടന ggml_tensor * ggml_മുകൾ_k_mask(
        ഘടന ggml_context * ctx,
        ഘടന ggml_tensor  * mask,
        ഘടന ggml_tensor  * മുകൾ_k,
        ഘടന ggml_tensor  * score,
        ഘടന ggml_tensor  * blocks,
        ഘടന ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(മുകൾ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(മുകൾ_k->ne[1] % mask->ne[1] == 0 && മുകൾ_k->ne[2] % mask->ne[2] == 0 && മുകൾ_k->ne[3] % mask->ne[3] == 0);

    എങ്കിൽ (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == മുകൾ_k->ne[1] && score->ne[2] == മുകൾ_k->ne[2] && score->ne[3] == മുകൾ_k->ne[3]);
    }

    എങ്കിൽ (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    എങ്കിൽ (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == മുകൾ_k->ne[1] && extra->ne[2] == മുകൾ_k->ne[2] && extra->ne[3] == മുകൾ_k->ne[3]);
    }

    ഘടന ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], മുകൾ_k->ne[1], മുകൾ_k->ne[2], മുകൾ_k->ne[3]);

    result->op     = GGML_OP_മുകൾ_K_MASK;
    result->src[0] = mask;
    result->src[1] = മുകൾ_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    തിരികെ result;
}

////////////////////////////////////////////////////////////////////////////////

ഘടന ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    ഘടന ggml_hash_set result;
    result.size = size;
    result.താക്കോലുകൾ = GGML_MALLOC(വലുപ്പം(ഘടന ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), വലുപ്പം(ggml_bitset_t));
    തിരികെ result;
}

ശൂന്യം ggml_hash_set_reset(ഘടന ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, വലുപ്പം(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

ശൂന്യം ggml_hash_set_free(ഘടന ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->താക്കോലുകൾ);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    സ്ഥിതം സ്ഥിരം size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    സ്ഥിതം സ്ഥിരം size_t n_primes = വലുപ്പം(primes)/വലുപ്പം(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    ആകുമ്പോൾ (l < r) {
        size_t m = (l + r)/2;
        എങ്കിൽ (primes[m] < min_sz) {
            l = m + 1;
        } അല്ലെങ്കിൽ {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    തിരികെ sz;
}

ഘടന hash_map {
    ഘടന ggml_hash_set set;
    ഘടന ggml_tensor ** vals;
};

സ്ഥിതം ഘടന hash_map * ggml_new_hash_map(size_t size) {
    ഘടന hash_map * result = GGML_MALLOC(വലുപ്പം(ഘടന hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, വലുപ്പം(ഘടന ggml_tensor *));
    തിരികെ result;
}

സ്ഥിതം ശൂന്യം ggml_hash_map_free(ഘടന hash_map * map) {
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

സ്ഥിതം ശൂന്യം ggml_add_or_set(
        ഘടന ggml_context * ctx,
        ഘടന ggml_cgraph  * cgraph,
        size_t                isrc,
        ഘടന ggml_tensor  * tensor) {
    ഘടന ggml_tensor * src = cgraph->visited_hash_set.താക്കോലുകൾ[isrc];
    GGML_ASSERT(src);
    എങ്കിൽ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } അല്ലെങ്കിൽ {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

സ്ഥിതം ശൂന്യം ggml_acc_or_set(
        ഘടന ggml_context * ctx,
        ഘടന ggml_cgraph  * cgraph,
        size_t                isrc,
        ഘടന ggml_tensor  * tensor,
        സ്ഥിരം  size_t         nb1,
        സ്ഥിരം  size_t         nb2,
        സ്ഥിരം  size_t         nb3,
        സ്ഥിരം  size_t         offset) {
    ഘടന ggml_tensor * src = cgraph->visited_hash_set.താക്കോലുകൾ[isrc];
    GGML_ASSERT(src);
    എങ്കിൽ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } അല്ലെങ്കിൽ {
        ഘടന ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.താക്കോലുകൾ[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

സ്ഥിതം ശൂന്യം ggml_add1_or_set(
        ഘടന ggml_context * ctx,
        ഘടന ggml_cgraph  * cgraph,
        size_t                isrc,
        ഘടന ggml_tensor  * tensor) {
    ഘടന ggml_tensor * src = cgraph->visited_hash_set.താക്കോലുകൾ[isrc];
    GGML_ASSERT(src);
    എങ്കിൽ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } അല്ലെങ്കിൽ {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

സ്ഥിതം ശൂന്യം ggml_sub_or_set(
        ഘടന ggml_context * ctx,
        ഘടന ggml_cgraph  * cgraph,
        size_t                isrc,
        ഘടന ggml_tensor  * tensor) {
    ഘടന ggml_tensor * src = cgraph->visited_hash_set.താക്കോലുകൾ[isrc];
    GGML_ASSERT(src);
    എങ്കിൽ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } അല്ലെങ്കിൽ {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

സ്ഥിതം ശൂന്യം ggml_compute_backward(
        ഘടന ggml_context * ctx, ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i, സ്ഥിരം bool * grads_needed) {
    ഘടന ggml_tensor * tensor = cgraph->nodes[i];
    ഘടന ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    എങ്കിൽ (!grad) {
        തിരികെ;
    }

    ഘടന ggml_tensor * src0 = tensor->src[0];
    ഘടന ggml_tensor * src1 = tensor->src[1];
    ഘടന ggml_tensor * src2 = tensor->src[2];
    ഘടന ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    സ്ഥിരം size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    സ്ഥിരം size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    സ്ഥിരം size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    സ്ഥിരം bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    സ്ഥിരം bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    സ്ഥിരം bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    തിരഞ്ഞെടുപ്പ് (tensor->op) {
        അവസ്ഥ GGML_OP_DUP: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            എങ്കിൽ (src1_needs_grads) {
                ഘടന ggml_tensor * tmp = grad;
                എങ്കിൽ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD1: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            എങ്കിൽ (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_ACC: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            എങ്കിൽ (src1_needs_grads) {
                സ്ഥിരം size_t nb1    = ((int32_t *) tensor->op_params)[0];
                സ്ഥിരം size_t nb2    = ((int32_t *) tensor->op_params)[1];
                സ്ഥിരം size_t nb3    = ((int32_t *) tensor->op_params)[2];
                സ്ഥിരം size_t offset = ((int32_t *) tensor->op_params)[3];

                ഘടന ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUB: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            എങ്കിൽ (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            എങ്കിൽ (src1_needs_grads) {
                ഘടന ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                എങ്കിൽ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIV: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            എങ്കിൽ (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SQR: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SQRT: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_LOG: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SIN: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_COS: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM_ROWS: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_MEAN: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT_BACK: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM: {
            എങ്കിൽ (src0_needs_grads) {
                ദശാംശം eps;
                memcpy(&eps, tensor->op_params, വലുപ്പം(ദശാംശം));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT: {
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

            എങ്കിൽ (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ഘടന ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                എങ്കിൽ (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    സ്ഥിരം int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    സ്ഥിരം size_t nb2 = tmp->nb[2] * nr2;
                    സ്ഥിരം size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            എങ്കിൽ (src1_needs_grads) {
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
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SCALE: {
            എങ്കിൽ (src0_needs_grads) {
                ദശാംശം s;
                memcpy(&s, tensor->op_params, വലുപ്പം(ദശാംശം));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SET: {
            സ്ഥിരം size_t nb1    = ((സ്ഥിരം int32_t *) tensor->op_params)[0];
            സ്ഥിരം size_t nb2    = ((സ്ഥിരം int32_t *) tensor->op_params)[1];
            സ്ഥിരം size_t nb3    = ((സ്ഥിരം int32_t *) tensor->op_params)[2];
            സ്ഥിരം size_t offset = ((സ്ഥിരം int32_t *) tensor->op_params)[3];

            ഘടന ggml_tensor * tensor_grad_view = ശൂന്യ;

            എങ്കിൽ (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            എങ്കിൽ (src0_needs_grads) {
                ഘടന ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            എങ്കിൽ (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            എങ്കിൽ (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            എങ്കിൽ (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONT: {
            // same as cpy
            എങ്കിൽ (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_RESHAPE: {
            എങ്കിൽ (src0_needs_grads) {
                ഘടന ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_VIEW: {
            എങ്കിൽ (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, വലുപ്പം(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                എങ്കിൽ (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_PERMUTE: {
            എങ്കിൽ (src0_needs_grads) {
                സ്ഥിരം int32_t * axes = (സ്ഥിരം int32_t *) tensor->op_params;
                സ്ഥിരം പൂർണ്ണസംഖ്യ axis0 = axes[0] & 0x3;
                സ്ഥിരം പൂർണ്ണസംഖ്യ axis1 = axes[1] & 0x3;
                സ്ഥിരം പൂർണ്ണസംഖ്യ axis2 = axes[2] & 0x3;
                സ്ഥിരം പൂർണ്ണസംഖ്യ axis3 = axes[3] & 0x3;
                പൂർണ്ണസംഖ്യ axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_TRANSPOSE: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            എങ്കിൽ (src1_needs_grads) {
                // noop
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_INF: {
            എങ്കിൽ (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_past = ((സ്ഥിരം int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_ZERO: {
            എങ്കിൽ (src0_needs_grads) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_past = ((സ്ഥിരം int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX: {
            എങ്കിൽ (src0_needs_grads) {
                ദശാംശം scale    = 1.0f;
                ദശാംശം max_bias = 0.0f;

                memcpy(&scale,    (സ്ഥിരം ദശാംശം *) tensor->op_params + 0, വലുപ്പം(ദശാംശം));
                memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) tensor->op_params + 1, വലുപ്പം(ദശാംശം));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE: {
            എങ്കിൽ (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims     = ((സ്ഥിരം int32_t *) tensor->op_params)[1];
                സ്ഥിരം പൂർണ്ണസംഖ്യ mode       = ((സ്ഥിരം int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_ctx_orig = ((സ്ഥിരം int32_t *) tensor->op_params)[4];
                ദശാംശം freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                പൂർണ്ണസംഖ്യ sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (സ്ഥിരം ദശാംശം *) tensor->op_params +  5, വലുപ്പം(ദശാംശം));
                memcpy(&freq_scale,  (സ്ഥിരം ദശാംശം *) tensor->op_params +  6, വലുപ്പം(ദശാംശം));
                memcpy(&ext_factor,  (സ്ഥിരം ദശാംശം *) tensor->op_params +  7, വലുപ്പം(ദശാംശം));
                memcpy(&attn_factor, (സ്ഥിരം ദശാംശം *) tensor->op_params +  8, വലുപ്പം(ദശാംശം));
                memcpy(&beta_fast,   (സ്ഥിരം ദശാംശം *) tensor->op_params +  9, വലുപ്പം(ദശാംശം));
                memcpy(&beta_slow,   (സ്ഥിരം ദശാംശം *) tensor->op_params + 10, വലുപ്പം(ദശാംശം));
                memcpy(&sections,                    tensor->op_params + 11, വലുപ്പം(sections));

                ഘടന ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL: {
            എങ്കിൽ (src1_needs_grads) {
                സ്ഥിരം int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                സ്ഥിരം int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                സ്ഥിരം int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                സ്ഥിരം int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                സ്ഥിരം int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                സ്ഥിരം int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                സ്ഥിരം bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_2D: {
            എങ്കിൽ (src0_needs_grads) {
                സ്ഥിരം എണ്ണൽ ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                സ്ഥിരം      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                സ്ഥിരം      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                സ്ഥിരം      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                സ്ഥിരം      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                സ്ഥിരം      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                സ്ഥിരം      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_WIN_PART:
        അവസ്ഥ GGML_OP_WIN_UNPART:
        അവസ്ഥ GGML_OP_UNARY: {
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(tensor)) {
                അവസ്ഥ GGML_UNARY_OP_ABS: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SGN: {
                    // noop
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_NEG: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_STEP: {
                    // noop
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_RELU: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SILU: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_EXP: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_EXPM1: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS: {
                    എങ്കിൽ (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } നിർത്തുക;
                സ്ഥിരസ്ഥിതി: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS: {
            എങ്കിൽ (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_GLU: {
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(tensor)) {
                അവസ്ഥ GGML_GLU_OP_SWIGLU: {
                    എങ്കിൽ (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    എങ്കിൽ (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } നിർത്തുക;
                സ്ഥിരസ്ഥിതി: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_NONE: {
            // noop
        } നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം:
        സ്ഥിരസ്ഥിതി: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

സ്ഥിതം size_t ggml_visit_parents_graph(ഘടന ggml_cgraph * cgraph, ഘടന ggml_tensor * കണ്ണി, bool compute) {
    എങ്കിൽ (കണ്ണി->op != GGML_OP_NONE && compute) {
        കണ്ണി->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    സ്ഥിരം size_t കണ്ണി_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി);
    GGML_ASSERT(കണ്ണി_hash_pos != GGML_HASHSET_FULL);

    എങ്കിൽ (ggml_bitset_get(cgraph->visited_hash_set.used, കണ്ണി_hash_pos)) {
        // already visited

        എങ്കിൽ (compute) {
            // update the compute flag regardless
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; ++i) {
                ഘടന ggml_tensor * src = കണ്ണി->src[i];
                എങ്കിൽ (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        തിരികെ കണ്ണി_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.താക്കോലുകൾ[കണ്ണി_hash_pos] = കണ്ണി;
    ggml_bitset_set(cgraph->visited_hash_set.used, കണ്ണി_hash_pos);
    cgraph->use_counts[കണ്ണി_hash_pos] = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ഇടത്_TO_വലത്) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_വലത്_TO_ഇടത്) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ഘടന ggml_tensor * src = കണ്ണി->src[k];
        എങ്കിൽ (src) {
            സ്ഥിരം size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    എങ്കിൽ (കണ്ണി->op == GGML_OP_NONE && !(കണ്ണി->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        എങ്കിൽ (strlen(കണ്ണി->name) == 0) {
            ggml_format_name(കണ്ണി, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = കണ്ണി;
        cgraph->n_leafs++;
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        എങ്കിൽ (strlen(കണ്ണി->name) == 0) {
            ggml_format_name(കണ്ണി, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = കണ്ണി;
        cgraph->n_nodes++;
    }

    തിരികെ കണ്ണി_hash_pos;
}

സ്ഥിതം ശൂന്യം ggml_build_forward_impl(ഘടന ggml_cgraph * cgraph, ഘടന ggml_tensor * tensor, bool expand, bool compute) {
    എങ്കിൽ (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    എങ്കിൽ (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ഘടന ggml_tensor * ggml_build_forward_select(
        ഘടന ggml_cgraph  * cgraph,
        ഘടന ggml_tensor ** tensors,
        പൂർണ്ണസംഖ്യ                   n_tensors,
        പൂർണ്ണസംഖ്യ                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    തിരികെ tensors[idx];
}

ശൂന്യം ggml_build_forward_expand(ഘടന ggml_cgraph * cgraph, ഘടന ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

ശൂന്യം ggml_build_forward_order(ഘടന ggml_cgraph * cgraph, ഘടന ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

ശൂന്യം ggml_build_backward_expand(
        ഘടന ggml_context *  ctx,
        ഘടന ggml_cgraph  *  cgraph,
        ഘടന ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*വലുപ്പം(ഘടന ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*വലുപ്പം(ഘടന ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, വലുപ്പം(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_nodes_f; ++i) {
            ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];
            any_params = any_params || (കണ്ണി->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (കണ്ണി->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_nodes_f; ++i) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        എങ്കിൽ (കണ്ണി->type == GGML_TYPE_I32) {
            തുടരുക;
        }

        bool കണ്ണി_needs_grad = (കണ്ണി->flags & GGML_TENSOR_FLAG_PARAM) || (കണ്ണി->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            അവസ്ഥ GGML_OP_IM2COL:      // only used for its shape
            അവസ്ഥ GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                നിർത്തുക;
            അവസ്ഥ GGML_OP_UNARY: {
                സ്ഥിരം എണ്ണൽ ggml_unary_op uop = ggml_get_unary_op(കണ്ണി);
                // SGN and STEP unary ops are piecewise constant
                എങ്കിൽ (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } നിർത്തുക;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            അവസ്ഥ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            അവസ്ഥ GGML_OP_GET_ROWS:      // row indices not differentiable
            അവസ്ഥ GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            അവസ്ഥ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                നിർത്തുക;

            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; ++j) {
            എങ്കിൽ (!കണ്ണി->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി->src[j])]) {
                തുടരുക;
            }
            GGML_ASSERT(കണ്ണി->src[j]->type == GGML_TYPE_F32 || കണ്ണി->src[j]->type == GGML_TYPE_F16);
            കണ്ണി_needs_grad = true;
            നിർത്തുക;
        }
        എങ്കിൽ (!കണ്ണി_needs_grad) {
            തുടരുക;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!കണ്ണി->view_src || കണ്ണി->op == GGML_OP_CPY || കണ്ണി->op == GGML_OP_SET_ROWS || കണ്ണി->op == GGML_OP_VIEW ||
            കണ്ണി->op == GGML_OP_RESHAPE || കണ്ണി->op == GGML_OP_PERMUTE || കണ്ണി->op == GGML_OP_TRANSPOSE);

        സ്ഥിരം size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        എങ്കിൽ (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, കണ്ണി->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    സ്വതന്ത്രം(grads_needed);
}

സ്ഥിതം ശൂന്യം * incr_ptr_aligned(ശൂന്യം ** p, size_t size, size_t align) {
    ശൂന്യം * ptr = *p;
    ptr = (ശൂന്യം *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ശൂന്യം *) ((അക്ഷരം *) ptr + size);
    തിരികെ ptr;
}

സ്ഥിതം size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    ശൂന്യം * p = (അക്ഷരം *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, വലുപ്പം(ഘടന ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * വലുപ്പം(int32_t), വലുപ്പം(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)); // hash keys
    എങ്കിൽ (grads) {
        incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * വലുപ്പം(ggml_bitset_t), വലുപ്പം(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    തിരികെ nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    തിരികെ GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(ശൂന്യം) {
    തിരികെ ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

ഘടന ggml_cgraph * ggml_new_graph_custom(ഘടന ggml_context * ctx, size_t size, bool grads) {
    സ്ഥിരം size_t obj_size = ggml_graph_nbytes(size, grads);
    ഘടന ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    ഘടന ggml_cgraph * cgraph = (ഘടന ggml_cgraph *) ((അക്ഷരം *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    ശൂന്യം * p = cgraph + 1;

    ഘടന ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *));
    ഘടന ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * വലുപ്പം(int32_t), വലുപ്പം(int32_t));
    ഘടന ggml_tensor ** hash_താക്കോലുകൾ_ptr  =         incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *));
    ഘടന ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)) : ശൂന്യ;
    ഘടന ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * വലുപ്പം(ഘടന ggml_tensor *), വലുപ്പം(ഘടന ggml_tensor *)) : ശൂന്യ;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * വലുപ്പം(ggml_bitset_t), വലുപ്പം(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    പ്രതിജ്ഞാ(obj_size == (size_t)((അക്ഷരം *)p - (അക്ഷരം *)cgraph));

    *cgraph = (ഘടന ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_താക്കോലുകൾ_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_ഇടത്_TO_വലത്,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    എങ്കിൽ (grads) {
        memset(cgraph->grads,     0, hash_size*വലുപ്പം(ഘടന ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*വലുപ്പം(ഘടന ggml_tensor *));
    }

    തിരികെ cgraph;
}

ഘടന ggml_cgraph * ggml_new_graph(ഘടന ggml_context * ctx) {
    തിരികെ ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

ഘടന ggml_cgraph ggml_graph_view(ഘടന ggml_cgraph * cgraph0, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1) {
    ഘടന ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ശൂന്യ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ശൂന്യ,
        /*.leafs            =*/ ശൂന്യ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    തിരികെ cgraph;
}

ശൂന്യം ggml_graph_cpy(ഘടന ggml_cgraph * src, ഘടന ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    വേണ്ടി (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        എങ്കിൽ (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_തിരുകുക(&dst->visited_hash_set, src->visited_hash_set.താക്കോലുകൾ[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    എങ്കിൽ (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*വലുപ്പം(ഘടന ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*വലുപ്പം(ഘടന ggml_tensor *));
    }
    എങ്കിൽ (src->grads) {
        GGML_ASSERT(dst->grads     != ശൂന്യ);
        GGML_ASSERT(dst->grad_accs != ശൂന്യ);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < src->n_nodes; ++i) {
            സ്ഥിരം size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            സ്ഥിരം size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ഘടന ggml_cgraph * ggml_graph_dup(ഘടന ggml_context * ctx, ഘടന ggml_cgraph * cgraph, bool force_grads) {
    ഘടന ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    തിരികെ result;
}

ഘടന ggml_tensor * ggml_set_zero(ഘടന ggml_tensor * tensor) {
    എങ്കിൽ (ggml_is_empty(tensor)) {
        തിരികെ tensor;
    }
    എങ്കിൽ (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    തിരികെ tensor;
}

ശൂന്യം ggml_graph_reset(ഘടന ggml_cgraph * cgraph) {
    എങ്കിൽ (!cgraph) {
        തിരികെ;
    }
    GGML_ASSERT(cgraph->grads != ശൂന്യ);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി     = cgraph->nodes[i];
        ഘടന ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, കണ്ണി);

        എങ്കിൽ (കണ്ണി->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(കണ്ണി->src[2]);
            ggml_set_zero(കണ്ണി->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        എങ്കിൽ (grad_acc) {
            എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                സ്ഥിരം ദശാംശം onef = 1.0f;
                എങ്കിൽ (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, വലുപ്പം(ദശാംശം));
                } അല്ലെങ്കിൽ {
                    GGML_ASSERT(grad_acc->data);
                    *((ദശാംശം *) grad_acc->data) = onef;
                }
            } അല്ലെങ്കിൽ {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

ശൂന്യം ggml_graph_clear(ഘടന ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

പൂർണ്ണസംഖ്യ ggml_graph_size(ഘടന ggml_cgraph * cgraph) {
    തിരികെ cgraph->size;
}

ഘടന ggml_tensor * ggml_graph_കണ്ണി(ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i) {
    എങ്കിൽ (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        തിരികെ cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    തിരികെ cgraph->nodes[i];
}

ഘടന ggml_tensor ** ggml_graph_nodes(ഘടന ggml_cgraph * cgraph) {
    തിരികെ cgraph->nodes;
}

പൂർണ്ണസംഖ്യ ggml_graph_n_nodes(ഘടന ggml_cgraph * cgraph) {
    തിരികെ cgraph->n_nodes;
}

ശൂന്യം ggml_graph_add_കണ്ണി(ഘടന ggml_cgraph * cgraph, ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ഘടന ggml_tensor * ggml_graph_get_tensor(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം അക്ഷരം * name) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_leafs; i++) {
        ഘടന ggml_tensor * leaf = cgraph->leafs[i];

        എങ്കിൽ (strcmp(leaf->name, name) == 0) {
            തിരികെ leaf;
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        എങ്കിൽ (strcmp(കണ്ണി->name, name) == 0) {
            തിരികെ കണ്ണി;
        }
    }

    തിരികെ ശൂന്യ;
}

ഘടന ggml_tensor * ggml_graph_get_grad(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം ഘടന ggml_tensor * കണ്ണി) {
    സ്ഥിരം size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി);
    തിരികെ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : ശൂന്യ;
}

ഘടന ggml_tensor * ggml_graph_get_grad_acc(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം ഘടന ggml_tensor * കണ്ണി) {
    സ്ഥിരം size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി);
    തിരികെ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ശൂന്യ;
}

ശൂന്യം ggml_graph_print(സ്ഥിരം ഘടന ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                കണ്ണി->ne[0], കണ്ണി->ne[1], കണ്ണി->ne[2],
                ggml_op_name(കണ്ണി->op), (കണ്ണി->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, കണ്ണി) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_leafs; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                കണ്ണി->ne[0], കണ്ണി->ne[1],
                ggml_op_name(കണ്ണി->op),
                ggml_get_name(കണ്ണി));
    }

    GGML_LOG_INFO("========================================\n");
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_കണ്ണി_list_find_tensor(സ്ഥിരം ഘടന ggml_cgraph * cgraph,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ *                idxs,
                                      പൂർണ്ണസംഖ്യ                        എണ്ണം,
                                      സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < എണ്ണം; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx = idxs[i];

        എങ്കിൽ (കണ്ണി_idx >= cgraph->n_nodes) {
            തിരികെ -1;
        }
        എങ്കിൽ (cgraph->nodes[കണ്ണി_idx] == tensor) {
            തിരികെ i;
        }
    }
    തിരികെ -1;
}

സ്ഥിതം bool ggml_is_constant(സ്ഥിരം ഘടന ggml_tensor * tensor) {
    തിരികെ tensor->buffer != ശൂന്യ && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(സ്ഥിരം ഘടന ggml_cgraph * cgraph,
                                സ്ഥിരം പൂർണ്ണസംഖ്യ *                കണ്ണി_idxs,
                                പൂർണ്ണസംഖ്യ                        എണ്ണം,
                                സ്ഥിരം എണ്ണൽ ggml_op *       ops,
                                സ്ഥിരം പൂർണ്ണസംഖ്യ *                outputs,
                                പൂർണ്ണസംഖ്യ                        അക്കം_outputs) {
    GGML_ASSERT(outputs && അക്കം_outputs > 0);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < എണ്ണം; ++i) {
        എങ്കിൽ (കണ്ണി_idxs[i] >= cgraph->n_nodes) {
            തിരികെ false;
        }

        സ്ഥിരം ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idxs[i]];

        എങ്കിൽ (കണ്ണി->op != ops[i]) {
            തിരികെ false;
        }

        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തിരികെ false;
        }

        എങ്കിൽ (ggml_കണ്ണി_list_find_tensor(cgraph, outputs, അക്കം_outputs, കണ്ണി) != -1) {
            തുടരുക;
        }

        എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_OUTPUT) {
            തിരികെ false;
        }

        പൂർണ്ണസംഖ്യ subgraph_uses = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = i + 1; j < എണ്ണം; ++j) {
            സ്ഥിരം ഘടന ggml_tensor * other_കണ്ണി = cgraph->nodes[കണ്ണി_idxs[j]];
            വേണ്ടി (പൂർണ്ണസംഖ്യ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                എങ്കിൽ (other_കണ്ണി->src[src_idx] == കണ്ണി) {
                    subgraph_uses++;
                }
            }
        }

        എങ്കിൽ (subgraph_uses != ggml_കണ്ണി_get_use_എണ്ണം(cgraph, കണ്ണി_idxs[i])) {
            തിരികെ false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ഘടന ggml_tensor * view_src = കണ്ണി->view_src;
        ആകുമ്പോൾ (view_src) {
            എങ്കിൽ (ggml_കണ്ണി_list_find_tensor(cgraph, കണ്ണി_idxs, എണ്ണം, view_src) == -1 && !ggml_is_constant(view_src)) {
                തിരികെ false;
            }
            view_src = view_src->view_src;
        }
    }

    തിരികെ true;
}

// check if node is part of the graph
സ്ഥിതം bool ggml_graph_find(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം ഘടന ggml_tensor * കണ്ണി) {
    എങ്കിൽ (cgraph == ശൂന്യ) {
        തിരികെ true;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        എങ്കിൽ (cgraph->nodes[i] == കണ്ണി) {
            തിരികെ true;
        }
    }

    തിരികെ false;
}

സ്ഥിതം ഘടന ggml_tensor * ggml_graph_get_parent(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം ഘടന ggml_tensor * കണ്ണി) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * parent = cgraph->nodes[i];
        ഘടന ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        എങ്കിൽ (grad == കണ്ണി) {
            തിരികെ parent;
        }
    }

    തിരികെ ശൂന്യ;
}

സ്ഥിതം ശൂന്യം ggml_graph_dump_dot_കണ്ണി_edge(FILE * fp, സ്ഥിരം ഘടന ggml_cgraph * gb, ഘടന ggml_tensor * കണ്ണി, ഘടന ggml_tensor * parent, സ്ഥിരം അക്ഷരം * label)  {
    ഘടന ggml_tensor * gparent = ggml_graph_get_parent(gb, കണ്ണി);
    ഘടന ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (ശൂന്യം *) gparent0 : (ശൂന്യം *) parent,
            gparent ? (ശൂന്യം *) gparent : (ശൂന്യം *) കണ്ണി,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

സ്ഥിതം ശൂന്യം ggml_graph_dump_dot_leaf_edge(FILE * fp, ഘടന ggml_tensor * കണ്ണി, ഘടന ggml_tensor * parent, സ്ഥിരം അക്ഷരം * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (ശൂന്യം *) parent,
            (ശൂന്യം *) കണ്ണി,
            label);
}

ശൂന്യം ggml_graph_dump_dot(സ്ഥിരം ഘടന ggml_cgraph * gb, സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം അക്ഷരം * filename) {
    അക്ഷരം color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gb->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = gb->nodes[i];
        ഘടന ggml_tensor * grad = ggml_graph_get_grad(gb, കണ്ണി);

        എങ്കിൽ (ggml_graph_get_parent(gb, കണ്ണി) != ശൂന്യ) {
            തുടരുക;
        }

        എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, വലുപ്പം(color), "yellow");
        } അല്ലെങ്കിൽ എങ്കിൽ (grad) {
            എങ്കിൽ (ggml_graph_find(cgraph, കണ്ണി)) {
                snprintf(color, വലുപ്പം(color), "green");
            } അല്ലെങ്കിൽ {
                snprintf(color, വലുപ്പം(color), "lightblue");
            }
        } അല്ലെങ്കിൽ {
            snprintf(color, വലുപ്പം(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (ശൂന്യം *) കണ്ണി, color);

        എങ്കിൽ (strlen(കണ്ണി->name) > 0) {
            fprintf(fp, "%s (%s)|", കണ്ണി->name, ggml_type_name(കണ്ണി->type));
        } അല്ലെങ്കിൽ {
            fprintf(fp, "(%s)|", ggml_type_name(കണ്ണി->type));
        }

        എങ്കിൽ (ggml_is_matrix(കണ്ണി)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, കണ്ണി->ne[0], കണ്ണി->ne[1], ggml_op_ചിഹ്നം(കണ്ണി->op));
        } അല്ലെങ്കിൽ {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, കണ്ണി->ne[0], കണ്ണി->ne[1], കണ്ണി->ne[2], ggml_op_ചിഹ്നം(കണ്ണി->op));
        }

        എങ്കിൽ (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_ചിഹ്നം(grad->op));
        } അല്ലെങ്കിൽ {
            fprintf(fp, "\"; ]\n");
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gb->n_leafs; i++) {
        ഘടന ggml_tensor * കണ്ണി = gb->leafs[i];

        snprintf(color, വലുപ്പം(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (ശൂന്യം *) കണ്ണി, color);

        എങ്കിൽ (strlen(കണ്ണി->name) > 0) {
            fprintf(fp, "%s (%s)|", കണ്ണി->name, ggml_type_name(കണ്ണി->type));
        } അല്ലെങ്കിൽ {
            fprintf(fp, "(%s)|", ggml_type_name(കണ്ണി->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, കണ്ണി->ne[0], കണ്ണി->ne[1]);
        എങ്കിൽ (ggml_nelements(കണ്ണി) < 5 && കണ്ണി->data != ശൂന്യ) {
            fprintf(fp, " | (");
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ggml_nelements(കണ്ണി); j++) {
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
                എങ്കിൽ (j < ggml_nelements(കണ്ണി) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gb->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = gb->nodes[i];

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            എങ്കിൽ (കണ്ണി->src[j]) {
                അക്ഷരം label[16];
                snprintf(label, വലുപ്പം(label), "src %d", j);
                ggml_graph_dump_dot_കണ്ണി_edge(fp, gb, കണ്ണി, കണ്ണി->src[j], label);
            }
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gb->n_leafs; i++) {
        ഘടന ggml_tensor * കണ്ണി = gb->leafs[i];

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            എങ്കിൽ (കണ്ണി->src[j]) {
                അക്ഷരം label[16];
                snprintf(label, വലുപ്പം(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, കണ്ണി, കണ്ണി->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

ശൂന്യം ggml_set_നിവേശം(ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_നിവേശം;
}

ശൂന്യം ggml_set_output(ഘടന ggml_tensor * tensor) {
    വേണ്ടി (ഘടന ggml_tensor * cur = tensor; cur != ശൂന്യ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

ശൂന്യം ggml_set_param(ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

ശൂന്യം ggml_set_loss(ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

ശൂന്യം ggml_quantize_init(എണ്ണൽ ggml_type type) {
    ggml_critical_section_start();

    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); നിർത്തുക;
        സ്ഥിരസ്ഥിതി: // nothing
            നിർത്തുക;
    }

    ggml_critical_section_end();
}

ശൂന്യം ggml_quantize_free(ശൂന്യം) {
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

bool ggml_quantize_requires_imatrix(എണ്ണൽ ggml_type type) {
    തിരികെ
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        എണ്ണൽ ggml_type   type,
           സ്ഥിരം ദശാംശം * src,
                  ശൂന്യം * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           സ്ഥിരം ദശാംശം * imatrix) {
    സ്ഥിരം int64_t n = nrows * n_per_row;

    എങ്കിൽ (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ശൂന്യ);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    സ്ഥിരം size_t start_row = start / n_per_row;
    സ്ഥിരം size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (അക്ഷരം *) dst + start_row * row_size, nrows, n_per_row, imatrix); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                size_t elemsize = വലുപ്പം(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                size_t elemsize = വലുപ്പം(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                size_t elemsize = വലുപ്പം(ദശാംശം);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            പ്രതിജ്ഞാ(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    തിരികെ result;
}

////////////////////////////////////////////////////////////////////////////////

ശൂന്യം ggml_log_get(ggml_log_callback * log_callback, ശൂന്യം ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

ശൂന്യം ggml_log_set(ggml_log_callback log_callback, ശൂന്യം * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

ശൂന്യം ggml_threadpool_params_init(ഘടന ggml_threadpool_params * p, പൂർണ്ണസംഖ്യ n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

ഘടന ggml_threadpool_params ggml_threadpool_params_default(പൂർണ്ണസംഖ്യ n_threads) {
    ഘടന ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    തിരികെ p;
}

bool ggml_threadpool_params_match(സ്ഥിരം ഘടന ggml_threadpool_params * p0, സ്ഥിരം ഘടന ggml_threadpool_params * p1) {
    എങ്കിൽ (p0->n_threads  != p1->n_threads  ) തിരികെ false;
    എങ്കിൽ (p0->prio       != p1->prio       ) തിരികെ false;
    എങ്കിൽ (p0->poll       != p1->poll       ) തിരികെ false;
    എങ്കിൽ (p0->strict_cpu != p1->strict_cpu ) തിരികെ false;
    തിരികെ memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
