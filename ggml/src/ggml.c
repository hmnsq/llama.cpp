#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#ଅନ୍ତର୍ଭାଵ "ggml-version.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-threading.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

// FIXME: required here for quantization functions
#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#ଅନ୍ତର୍ଭାଵ <hbwmalloc.h>
#endif

#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
#ଅନ୍ତର୍ଭାଵ <malloc.h> // using malloc.h with MSC/MINGW
#ନଚେତଯଦି !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ଅନ୍ତର୍ଭାଵ <alloca.h>
#endif

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <errno.h>
#ଅନ୍ତର୍ଭାଵ <time.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <inttypes.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <float.h>
#ଅନ୍ତର୍ଭାଵ <limits.h>
#ଅନ୍ତର୍ଭାଵ <stdarg.h>
#ଅନ୍ତର୍ଭାଵ <signal.h>
#ଯଦି defined(__gnu_linux__)
#ଅନ୍ତର୍ଭାଵ <syscall.h>
#endif

#ଯଦି defined(__APPLE__)
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ <mach/mach.h>
#ଅନ୍ତର୍ଭାଵ <TargetConditionals.h>
#endif

#ଯଦି defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#ଅନ୍ତର୍ଭାଵ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(ଶୂନ୍ୟ) {
#ifdef _MSC_VER
#ଯଦି defined(_WIN32)
    ସ୍ଥିର ଅସ୍ଥିର LONG counter = 1;
    ଫେରନ୍ତୁ (uint64_t) InterlockedIncrement(&counter) - 1;
#ଅନ୍ୟଥା
    ସ୍ଥିର ଅସ୍ଥିର long long counter = 1;
    ଫେରନ୍ତୁ (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#ଅନ୍ୟଥା
    ସ୍ଥିର uint64_t counter = 1;
    ଫେରନ୍ତୁ __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#ଯଦି defined(__AVX512BF16__)
#ଯଦି defined(_MSC_VER)
#define m512i(p) p
#ଅନ୍ୟଥା
#ଅନ୍ତର୍ଭାଵ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#ଯଦି defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
#ଅନ୍ତର୍ଭାଵ <sys/wait.h>
#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <sys/prctl.h>
#endif

#ଯଦି defined(__ANDROID__)
#ଅନ୍ତର୍ଭାଵ <unwind.h>
#ଅନ୍ତର୍ଭାଵ <dlfcn.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>

ସଂରଚନା backtrace_state {
    ଶୂନ୍ୟ ** current;
    ଶୂନ୍ୟ ** end;
};

ସ୍ଥିର _Unwind_Reason_Code unwind_callback(ସଂରଚନା _Unwind_Context* context, ଶୂନ୍ୟ* arg) {
    ସଂରଚନା backtrace_state * state = (ସଂରଚନା backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    ଯଦି (pc) {
        ଯଦି (state->current == state->end) {
            ଫେରନ୍ତୁ URC_END_OF_ଗଦା;
        } ଅନ୍ୟଥା {
            *state->current++ = (ଶୂନ୍ୟ*)pc;
        }
    }
    ଫେରନ୍ତୁ _URC_NO_REASON;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_print_backtrace_symbols(ଶୂନ୍ୟ) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା max = 100;
    ଶୂନ୍ୟ* buffer[max];

    ସଂରଚନା backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣନା_2 = state.current - buffer;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = 0; idx < ଗଣନା_2; ++idx) {
        ଅଚଳ ଶୂନ୍ୟ * addr = buffer[idx];
        ଅଚଳ ଅକ୍ଷର * ପ୍ରତୀକ = "";

        Dl_info info;
        ଯଦି (dladdr(addr, &info) && info.dli_sname) {
            ପ୍ରତୀକ = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, ପ୍ରତୀକ);
    }
}
#ନଚେତଯଦି defined(__linux__) && defined(__GLIBC__)
#ଅନ୍ତର୍ଭାଵ <execinfo.h>
ସ୍ଥିର ଶୂନ୍ୟ ggml_print_backtrace_symbols(ଶୂନ୍ୟ) {
    ଶୂନ୍ୟ * trace[100];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nptrs = backtrace(trace, ଆକାର(trace)/ଆକାର(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ନଚେତଯଦି defined(__APPLE__)
#ଅନ୍ତର୍ଭାଵ <execinfo.h>
ସ୍ଥିର ଶୂନ୍ୟ ggml_print_backtrace_symbols(ଶୂନ୍ୟ) {
    ଶୂନ୍ୟ * trace[100];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nptrs = backtrace(trace, ଆକାର(trace)/ଆକାର(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଶୂନ୍ୟ ggml_print_backtrace_symbols(ଶୂନ୍ୟ) {
    // platform not supported
}
#endif

ଶୂନ୍ୟ ggml_print_backtrace(ଶୂନ୍ୟ) {
    ଅଚଳ ଅକ୍ଷର * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ଯଦି (GGML_NO_BACKTRACE) {
        ଫେରନ୍ତୁ;
    }
#ଯଦି defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ଅଚଳ ଅକ୍ଷର * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    ଯଦି (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ଫେରନ୍ତୁ;
    }
#endif
#ଯଦି defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    ଅକ୍ଷର * ଧାଡ଼ି = ଖାଲି;
    ssize_t length = 0;
    ଯେତେବେଳେ ((length = getline(&ଧାଡ଼ି, &size, f)) > 0) {
        ଯଦି (!strncmp(ଧାଡ଼ି, "TracerPid:", ଆକାର("TracerPid:") - 1) &&
            (length != ଆକାର("TracerPid:\t0\n") - 1 || ଧାଡ଼ି[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            ମୁକ୍ତ(ଧାଡ଼ି);
            fclose(f);
            ଫେରନ୍ତୁ;
        }
    }
    ମୁକ୍ତ(ଧାଡ଼ି);
    fclose(f);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା lock[2] = { -1, -1 };
    (ଶୂନ୍ୟ) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parent_pid = getpid();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା child_pid = fork();
    ଯଦି (child_pid < 0) { // error
#ଯଦି defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା ଯଦି (child_pid == 0) { // child
        ଅକ୍ଷର attach[32];
        snprintf(attach, ଆକାର(attach), "attach %d", parent_pid);
#ଯଦି defined(__linux__)
        close(lock[1]);
        (ଶୂନ୍ୟ) !ପଠ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (ଅକ୍ଷର *) ଖାଲି);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[ଆକାର("attach ") - 1],
            (ଅକ୍ଷର *) ଖାଲି);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } ଅନ୍ୟଥା { // parent
#ଯଦି defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, ଖାଲି, 0);
    }
}
#ଅନ୍ୟଥା
ଶୂନ୍ୟ ggml_print_backtrace(ଶୂନ୍ୟ) {
    // platform not supported
}
#endif

ସ୍ଥିର ggml_abort_callback_t g_abort_callback = ଖାଲି;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ଫେରନ୍ତୁ ret_val;
}

ଶୂନ୍ୟ ggml_abort(ଅଚଳ ଅକ୍ଷର * file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି, ଅଚଳ ଅକ୍ଷର * fmt, ...) {
    fflush(stdout);

    ଅକ୍ଷର message[2048];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = snprintf(message, ଆକାର(message), "%s:%d: ", file, ଧାଡ଼ି);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, ଆକାର(message) - offset, fmt, args);
    va_end(args);

    ଯଦି (g_abort_callback) {
        g_abort_callback(message);
    } ଅନ୍ୟଥା {
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

ସଂରଚନା ggml_logger_state {
    ggml_log_callback log_callback;
    ଶୂନ୍ୟ * log_callback_user_data;
};
ସ୍ଥିର ସଂରଚନା ggml_logger_state g_logger_state = {ggml_log_callback_default, ଖାଲି};

ସ୍ଥିର ଶୂନ୍ୟ ggml_log_internal_v(ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * format, va_list args) {
    ଯଦି (format == ଖାଲି) {
        ଫେରନ୍ତୁ;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    ଅକ୍ଷର buffer[128];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା len = vsnprintf(buffer, 128, format, args);
    ଯଦି (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } ଅନ୍ୟଥା {
        ଅକ୍ଷର * buffer2 = (ଅକ୍ଷର *) calloc(len + 1, ଆକାର(ଅକ୍ଷର));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        ମୁକ୍ତ(buffer2);
    }
    va_end(args_copy);
}

ଶୂନ୍ୟ ggml_log_internal(ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

ଶୂନ୍ୟ ggml_log_callback_default(ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * text, ଶୂନ୍ୟ * user_data) {
    (ଶୂନ୍ୟ) level;
    (ଶୂନ୍ୟ) user_data;
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
#ଯଦି defined(__has_feature)
#ଯଦି __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#ଅନ୍ୟଥା
#ଯଦି defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

ଶୂନ୍ୟ * ggml_aligned_malloc(size_t size) {
#ଯଦି defined(__s390x__)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା alignment = 256;
#ଅନ୍ୟଥା
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା alignment = 64;
#endif

#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
    ଫେରନ୍ତୁ _aligned_malloc(size, alignment);
#ଅନ୍ୟଥା
    ଯଦି (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଶୂନ୍ୟ * aligned_memory = ଖାଲି;
  #ifdef GGML_USE_CPU_HBM
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #ନଚେତଯଦି TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା result = EFAULT;
    ବାଛ (alloc_status) {
        ସ୍ଥିତି KERN_SUCCESS:
            result = 0;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି KERN_INVALID_ADDRESS:
            result = EINVAL;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି KERN_NO_SPACE:
            result = ENOMEM;
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            result = EFAULT;
            ଭାଙ୍ଗ;
    }
  #ଅନ୍ୟଥା
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    ଯଦି (result != 0) {
        // Handle allocation failure
        ଅଚଳ ଅକ୍ଷର *error_desc = "unknown allocation error";
        ବାଛ (result) {
            ସ୍ଥିତି EINVAL:
                error_desc = "invalid alignment value";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି ENOMEM:
                error_desc = "insufficient memory";
                ଭାଙ୍ଗ;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଫେରନ୍ତୁ aligned_memory;
#endif
}

ଶୂନ୍ୟ ggml_aligned_free(ଶୂନ୍ୟ * ptr, size_t size) {
    GGML_UNUSED(size);
#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#ନଚେତଯଦି GGML_USE_CPU_HBM
    ଯଦି (ptr != ଖାଲି) {
        hbw_free(ptr);
    }
#ନଚେତଯଦି TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    ଯଦି (ptr != ଖାଲି) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#ଅନ୍ୟଥା
    ମୁକ୍ତ(ptr);
#endif
}


ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ * ggml_malloc(size_t size) {
    ଯଦି (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଶୂନ୍ୟ * result = malloc(size);
    ଯଦି (result == ଖାଲି) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ଫେରନ୍ତୁ result;
}

// calloc
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ * ggml_calloc(size_t ଅଙ୍କ, size_t size) {
    ଯଦି (ଅଙ୍କ == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଶୂନ୍ୟ * result = calloc(ଅଙ୍କ, size);
    ଯଦି (result == ଖାଲି) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ଫେରନ୍ତୁ result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(ଅଙ୍କ, size) ggml_calloc(ଅଙ୍କ, size)

#define GGML_FREE(ptr) ମୁକ୍ତ(ptr)

ଅଚଳ ଅକ୍ଷର * ggml_status_to_string(ଗଣନା ggml_status status) {
    ବାଛ (status) {
        ସ୍ଥିତି GGML_STATUS_ALLOC_FAILED: ଫେରନ୍ତୁ "GGML status: error (failed to allocate memory)";
        ସ୍ଥିତି GGML_STATUS_FAILED:       ଫେରନ୍ତୁ "GGML status: error (operation failed)";
        ସ୍ଥିତି GGML_STATUS_SUCCESS:      ଫେରନ୍ତୁ "GGML status: success";
        ସ୍ଥିତି GGML_STATUS_ABORTED:      ଫେରନ୍ତୁ "GGML status: warning (operation aborted)";
    }

    ଫେରନ୍ତୁ "GGML status: unknown";
}

ଦଶମିକ ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    ଫେରନ୍ତୁ GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(ଦଶମିକ x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    ଫେରନ୍ତୁ GGML_FP32_TO_FP16(x);
}

ଦଶମିକ ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    ଫେରନ୍ତୁ GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(ଦଶମିକ x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    ଫେରନ୍ତୁ GGML_FP32_TO_BF16(x);
}

ଶୂନ୍ୟ ggml_fp16_to_fp32_row(ଅଚଳ ggml_fp16_t * x, ଦଶମିକ * y, int64_t n) {
    ପାଇଁ (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

ଶୂନ୍ୟ ggml_fp32_to_fp16_row(ଅଚଳ ଦଶମିକ * x, ggml_fp16_t * y, int64_t n) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
    ପାଇଁ (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

ଶୂନ୍ୟ ggml_bf16_to_fp32_row(ଅଚଳ ggml_bf16_t * x, ଦଶମିକ * y, int64_t n) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
    ପାଇଁ (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

ଶୂନ୍ୟ ggml_fp32_to_bf16_row_ref(ଅଚଳ ଦଶମିକ * x, ggml_bf16_t * y, int64_t n) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

ଶୂନ୍ୟ ggml_fp32_to_bf16_row(ଅଚଳ ଦଶମିକ * x, ggml_bf16_t * y, int64_t n) {
  ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
#ଯଦି defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  ପାଇଁ (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    ପାଇଁ (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ଫେରନ୍ତୁ memcmp(guid_a, guid_b, ଆକାର(ggml_guid)) == 0;
}

ଅଚଳ ଅକ୍ଷର * ggml_version(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ GGML_VERSION;
}

ଅଚଳ ଅକ୍ଷର * ggml_commit(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ GGML_COMMIT;
}

//
// timing
//

#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
ସ୍ଥିର int64_t timer_freq, timer_start;
ସ୍ଥିର BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    ଫେରନ୍ତୁ TRUE;
}
ଶୂନ୍ୟ ggml_time_init(ଶୂନ୍ୟ) {
    ସ୍ଥିର INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, ଖାଲି, ଖାଲି);
}
int64_t ggml_time_ms(ଶୂନ୍ୟ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ଫେରନ୍ତୁ ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(ଶୂନ୍ୟ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ଫେରନ୍ତୁ ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#ଅନ୍ୟଥା
ଶୂନ୍ୟ ggml_time_init(ଶୂନ୍ୟ) {}
int64_t ggml_time_ms(ଶୂନ୍ୟ) {
    ସଂରଚନା timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ଫେରନ୍ତୁ (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(ଶୂନ୍ୟ) {
    ସଂରଚନା timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ଫେରନ୍ତୁ (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ clock();
}

int64_t ggml_cycles_per_ms(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
ସ୍ଥିର wchar_t * ggml_mbstowcs(ଅଚଳ ଅକ୍ଷର * mbs) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, ଖାଲି, 0);
    ଯଦି (!wlen) {
        errno = EINVAL;
        ଫେରନ୍ତୁ ଖାଲି;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * ଆକାର(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    ଯଦି (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ wbuf;
}
#endif

FILE * ggml_fopen(ଅଚଳ ଅକ୍ଷର * fname, ଅଚଳ ଅକ୍ଷର * mode) {
#ifdef _WIN32
    FILE * file = ଖାଲି;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    ଯଦି (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        ଯଦି (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    ଫେରନ୍ତୁ file;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ fopen(fname, mode);
#endif

}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_type_traits type_traits[GGML_TYPE_ଗଣନା] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = ଆକାର(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = ଆକାର(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = ଆକାର(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = ଆକାର(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = ଆକାର(ଦ୍ୱିଗୁଣ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = ଆକାର(ଦଶମିକ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = ଆକାର(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = ଆକାର(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = ଆକାର(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = ଆକାର(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = ଆକାର(block_q4_1),
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
        .type_size                = ଆକାର(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = ଆକାର(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = ଆକାର(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = ଆକାର(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = ଆକାର(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = ଆକାର(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = ଖାଲି,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = ଖାଲି,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = ଖାଲି,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = ଖାଲି,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = ଆକାର(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = ଆକାର(ggml_bf16_t),
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
        .type_size                = ଆକାର(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = ଆକାର(block_tq2_0),
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

ଅଚଳ ସଂରଚନା ggml_type_traits * ggml_get_type_traits(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ &type_traits[type];
}

//
// ggml object
//

ସଂରଚନା ggml_object {
    size_t offs;
    size_t size;

    ସଂରଚନା ggml_object * next;

    ଗଣନା ggml_object_type type;

    ଅକ୍ଷର padding[4];
};

ସ୍ଥିର ଅଚଳ size_t GGML_OBJECT_SIZE = ଆକାର(ସଂରଚନା ggml_object);

//
// ggml context
//

ସଂରଚନା ggml_context {
    size_t mem_size;
    ଶୂନ୍ୟ * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    n_objects;

    ସଂରଚନା ggml_object * objects_begin;
    ସଂରଚନା ggml_object * objects_end;
};

//
// data types
//

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * GGML_OP_NAME[GGML_OP_ଗଣନା] = {
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

static_assert(GGML_OP_ଗଣନା == 102, "GGML_OP_COUNT != 102");

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * GGML_OP_ପ୍ରତୀକ[GGML_OP_ଗଣନା] = {
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

static_assert(GGML_OP_ଗଣନା == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_ଗଣନା == 2, "GGML_OP_POOL_COUNT != 2");

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * GGML_UNARY_OP_NAME[GGML_UNARY_OP_ଗଣନା] = {
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

static_assert(GGML_UNARY_OP_ଗଣନା == 22, "GGML_UNARY_OP_COUNT != 22");

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * GGML_GLU_OP_NAME[GGML_GLU_OP_ଗଣନା] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_ଗଣନା == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(ଆକାର(ସଂରଚନା ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(ଆକାର(ସଂରଚନା ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

ଶୂନ୍ୟ ggml_print_object(ଅଚଳ ସଂରଚନା ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ଅଚଳ ଶୂନ୍ୟ *) obj->next);
}

ଶୂନ୍ୟ ggml_print_objects(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    ସଂରଚନା ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ଅଚଳ ଶୂନ୍ୟ *) ctx);

    ଯେତେବେଳେ (obj != ଖାଲି) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
        ଯଦି (tensor->ne[i] <= 0) {
            ଫେରନ୍ତୁ 0;
        }
    }

    size_t nbytes;
    ଅଚଳ size_t blck_size = ggml_blck_size(tensor->type);
    ଯଦି (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    ଅନ୍ୟଥା {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ଫେରନ୍ତୁ nbytes;
}

size_t ggml_nbytes_pad(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ type_traits[type].blck_size;
}

size_t ggml_type_size(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ type_traits[type].type_size;
}

size_t ggml_row_size(ଗଣନା ggml_type type, int64_t ne) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ପ୍ରତିଜ୍ଞା(ne % ggml_blck_size(type) == 0);
    ଫେରନ୍ତୁ ggml_type_size(type)*ne/ggml_blck_size(type);
}

ଦ୍ୱିଗୁଣ ggml_type_sizef(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ ((ଦ୍ୱିଗୁଣ)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ଅଚଳ ଅକ୍ଷର * ggml_type_name(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ type_traits[type].type_name;
}

bool ggml_is_quantized(ଗଣନା ggml_type type) {
    ପ୍ରତିଜ୍ଞା(type >= 0);
    ପ୍ରତିଜ୍ଞା(type < GGML_TYPE_ଗଣନା);
    ଫେରନ୍ତୁ type_traits[type].is_quantized;
}

ଅଚଳ ଅକ୍ଷର * ggml_op_name(ଗଣନା ggml_op op) {
    ଫେରନ୍ତୁ GGML_OP_NAME[op];
}

ଅଚଳ ଅକ୍ଷର * ggml_op_ପ୍ରତୀକ(ଗଣନା ggml_op op) {
    ଫେରନ୍ତୁ GGML_OP_ପ୍ରତୀକ[op];
}

ଅଚଳ ଅକ୍ଷର * ggml_unary_op_name(ଗଣନା ggml_unary_op op) {
    ଫେରନ୍ତୁ GGML_UNARY_OP_NAME[op];
}

ଅଚଳ ଅକ୍ଷର * ggml_glu_op_name(ଗଣନା ggml_glu_op op) {
    ଫେରନ୍ତୁ GGML_GLU_OP_NAME[op];
}

ଅଚଳ ଅକ୍ଷର * ggml_op_desc(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (t->op == GGML_OP_UNARY) {
        ଗଣନା ggml_unary_op uop = ggml_get_unary_op(t);
        ଫେରନ୍ତୁ ggml_unary_op_name(uop);
    }
    ଯଦି (t->op == GGML_OP_GLU) {
        ଗଣନା ggml_glu_op gop = ggml_get_glu_op(t);
        ଫେରନ୍ତୁ ggml_glu_op_name(gop);
    }
    ଫେରନ୍ତୁ ggml_op_name(t->op);
}

size_t ggml_element_size(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor->ne[3] == 1;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_n_dims(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        ଯଦି (tensor->ne[i] > 1) {
            ଫେରନ୍ତୁ i + 1;
        }
    }
    ଫେରନ୍ତୁ 1;
}

ଗଣନା ggml_type ggml_ftype_to_ggml_type(ଗଣନା ggml_ftype ftype) {
    ଗଣନା ggml_type wtype = GGML_TYPE_ଗଣନା;

    ବାଛ (ftype) {
        ସ୍ଥିତି GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_ଗଣନା; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_ଗଣନା; ଭାଙ୍ଗ;
    }

    GGML_ASSERT(wtype != GGML_TYPE_ଗଣନା);

    ଫେରନ୍ତୁ wtype;
}

size_t ggml_tensor_overhead(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor->nb[0] > tensor->nb[1];
}

ସ୍ଥିର bool ggml_is_contiguous_m_n(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    size_t next_nb = ggml_type_size(tensor->type);
    ଯଦି (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        ଫେରନ୍ତୁ false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < n; i++) {
        ଯଦି (i > m) {
            ଯଦି (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                ଫେରନ୍ତୁ false;
            }
            next_nb *= tensor->ne[i];
        } ଅନ୍ୟଥା {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ଫେରନ୍ତୁ true;
}

bool ggml_is_contiguous(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_is_padded_1d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
        ଯଦି (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ଫେରନ୍ତୁ true;
        }
    }
    ଫେରନ୍ତୁ false;
}

bool ggml_are_same_shape(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_repeat_rows(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ସଂରଚନା ggml_context * ggml_init(ସଂରଚନା ggml_init_params params) {
    ସ୍ଥିର bool is_first_call = true;

    ggml_critical_section_start();

    ଯଦି (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    ସଂରଚନା ggml_context * ctx = GGML_MALLOC(ଆକାର(ସଂରଚନା ggml_context));

    // allow to call ggml_init with 0 size
    ଯଦି (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ଅଚଳ size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (ସଂରଚନା ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ଖାଲି,
        /*.objects_end        =*/ ଖାଲି,
    };

    GGML_ASSERT(ctx->mem_buffer != ଖାଲି);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ଫେରନ୍ତୁ ctx;
}

ଶୂନ୍ୟ ggml_reset(ସଂରଚନା ggml_context * ctx) {
    ଯଦି (ctx == ଖାଲି) {
        ଫେରନ୍ତୁ;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ଖାଲି;
    ctx->objects_end   = ଖାଲି;
}

ଶୂନ୍ୟ ggml_free(ସଂରଚନା ggml_context * ctx) {
    ଯଦି (ctx == ଖାଲି) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    ଫେରନ୍ତୁ ctx->objects_end == ଖାଲି ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(ସଂରଚନା ggml_context * ctx) {
    ଫେରନ୍ତୁ ctx->no_alloc;
}

ଶୂନ୍ୟ ggml_set_no_alloc(ସଂରଚନା ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

ଶୂନ୍ୟ * ggml_get_mem_buffer(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    ଫେରନ୍ତୁ ctx->mem_buffer;
}

size_t ggml_get_mem_size(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    ଫେରନ୍ତୁ ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    size_t max_size = 0;

    ପାଇଁ (ସଂରଚନା ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != ଖାଲି; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    ଫେରନ୍ତୁ max_size;
}

////////////////////////////////////////////////////////////////////////////////

ସ୍ଥିର ସଂରଚନା ggml_object * ggml_new_object(ସଂରଚନା ggml_context * ctx, ଗଣନା ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    ସଂରଚନା ggml_object * obj_cur = ctx->objects_end;

    ଅଚଳ size_t cur_offs = obj_cur == ଖାଲି ? 0 : obj_cur->offs;
    ଅଚଳ size_t cur_size = obj_cur == ଖାଲି ? 0 : obj_cur->size;
    ଅଚଳ size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    ଅକ୍ଷର * ଅଚଳ mem_buffer = ctx->mem_buffer;
    ସଂରଚନା ggml_object * ଅଚଳ obj_new = (ସଂରଚନା ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    ଯଦି (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଯଦି (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଯଦି (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        ଫେରନ୍ତୁ ଖାଲି;
    }

    *obj_new = (ସଂରଚନା ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = ଖାଲି,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    ଯଦି (obj_cur != ଖାଲି) {
        obj_cur->next = obj_new;
    } ଅନ୍ୟଥା {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ଫେରନ୍ତୁ obj_new;
}

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_new_tensor_impl(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type      type,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ଅଚଳ int64_t       * ne,
        ସଂରଚନା ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_ଗଣନା);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    ଯଦି (view_src != ଖାଲି && view_src->view_src != ଖାଲି) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < n_dims; i++) {
        ଯଦି (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == ଖାଲି || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    ଶୂନ୍ୟ * data = view_src != ଖାଲି ? view_src->data : ଖାଲି;
    ଯଦି (data != ଖାଲି) {
        data = (ଅକ୍ଷର *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    ଯଦି (view_src == ଖାଲି && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    ସଂରଚନା ggml_object * ଅଚଳ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    ସଂରଚନା ggml_tensor * ଅଚଳ result = (ସଂରଚନା ggml_tensor *)((ଅକ୍ଷର *)ctx->mem_buffer + obj_new->offs);

    *result = (ସଂରଚନା ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ଖାଲି,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ଖାଲି },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (ଶୂନ୍ୟ *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ଖାଲି,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_new_tensor(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type      type,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ଅଚଳ int64_t       * ne) {
    ଫେରନ୍ତୁ ggml_new_tensor_impl(ctx, type, n_dims, ne, ଖାଲି, 0);
}

ସଂରଚନା ggml_tensor * ggml_new_tensor_1d(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type      type,
        int64_t ne0) {
    ଫେରନ୍ତୁ ggml_new_tensor(ctx, type, 1, &ne0);
}

ସଂରଚନା ggml_tensor * ggml_new_tensor_2d(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ଅଚଳ int64_t ne[2] = { ne0, ne1 };
    ଫେରନ୍ତୁ ggml_new_tensor(ctx, type, 2, ne);
}

ସଂରଚନା ggml_tensor * ggml_new_tensor_3d(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ଅଚଳ int64_t ne[3] = { ne0, ne1, ne2 };
    ଫେରନ୍ତୁ ggml_new_tensor(ctx, type, 3, ne);
}

ସଂରଚନା ggml_tensor * ggml_new_tensor_4d(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ଅଚଳ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ଫେରନ୍ତୁ ggml_new_tensor(ctx, type, 4, ne);
}

ଶୂନ୍ୟ * ggml_new_buffer(ସଂରଚନା ggml_context * ctx, size_t nbytes) {
    ସଂରଚନା ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    ଫେରନ୍ତୁ (uint8_t *)ctx->mem_buffer + obj->offs;
}

ସଂରଚନା ggml_tensor * ggml_dup_tensor(ସଂରଚନା ggml_context * ctx, ଅଚଳ ସଂରଚନା ggml_tensor * src) {
    ଫେରନ୍ତୁ ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

ଶୂନ୍ୟ ggml_unravel_index(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ଅଚଳ int64_t ne2 = tensor->ne[2];
    ଅଚଳ int64_t ne1 = tensor->ne[1];
    ଅଚଳ int64_t ne0 = tensor->ne[0];

    ଅଚଳ int64_t i3_ = (i/(ne2*ne1*ne0));
    ଅଚଳ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ଅଚଳ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ଅଚଳ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    ଯଦି (i0) {
        * i0 = i0_;
    }
    ଯଦି (i1) {
        * i1 = i1_;
    }
    ଯଦି (i2) {
        * i2 = i2_;
    }
    ଯଦି (i3) {
        * i3 = i3_;
    }
}

ଶୂନ୍ୟ * ggml_get_data(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor->data;
}

ଦଶମିକ * ggml_get_data_f32(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ପ୍ରତିଜ୍ଞା(tensor->type == GGML_TYPE_F32);
    ଫେରନ୍ତୁ (ଦଶମିକ *)(tensor->data);
}

ଗଣନା ggml_unary_op ggml_get_unary_op(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ଫେରନ୍ତୁ (ଗଣନା ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

ଗଣନା ggml_glu_op ggml_get_glu_op(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ଫେରନ୍ତୁ (ଗଣନା ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ଅଚଳ ଅକ୍ଷର * ggml_get_name(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor->name;
}

ସଂରଚନା ggml_tensor * ggml_set_name(ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଅକ୍ଷର * name) {
    size_t i;
    ପାଇଁ (i = 0; i < ଆକାର(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ଫେରନ୍ତୁ tensor;
}

ସଂରଚନା ggml_tensor * ggml_format_name(ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଅକ୍ଷର * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, ଆକାର(tensor->name), fmt, args);
    va_end(args);
    ଫେରନ୍ତୁ tensor;
}

ସଂରଚନା ggml_tensor * ggml_view_tensor(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * src) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_get_first_tensor(ଅଚଳ ସଂରଚନା ggml_context * ctx) {
    ସଂରଚନା ggml_object * obj = ctx->objects_begin;

    ଅକ୍ଷର * ଅଚଳ mem_buffer = ctx->mem_buffer;

    ଯେତେବେଳେ (obj != ଖାଲି) {
        ଯଦି (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ଫେରନ୍ତୁ (ସଂରଚନା ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

ସଂରଚନା ggml_tensor * ggml_get_next_tensor(ଅଚଳ ସଂରଚନା ggml_context * ctx, ସଂରଚନା ggml_tensor * tensor) {
    ସଂରଚନା ggml_object * obj = (ସଂରଚନା ggml_object *) ((ଅକ୍ଷର *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    ଅକ୍ଷର * ଅଚଳ mem_buffer = ctx->mem_buffer;

    ଯେତେବେଳେ (obj != ଖାଲି) {
        ଯଦି (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ଫେରନ୍ତୁ (ସଂରଚନା ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

ସଂରଚନା ggml_tensor * ggml_get_tensor(ସଂରଚନା ggml_context * ctx, ଅଚଳ ଅକ୍ଷର * name) {
    ସଂରଚନା ggml_object * obj = ctx->objects_begin;

    ଅକ୍ଷର * ଅଚଳ mem_buffer = ctx->mem_buffer;

    ଯେତେବେଳେ (obj != ଖାଲି) {
        ଯଦି (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ସଂରଚନା ggml_tensor * cur = (ସଂରଚନା ggml_tensor *)(mem_buffer + obj->offs);
            ଯଦି (strcmp(cur->name, name) == 0) {
                ଫେରନ୍ତୁ cur;
            }
        }

        obj = obj->next;
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_dup_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_dup(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_dup_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_dup_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_dup_impl(ctx, a, true);
}

// ggml_add

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_add_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_add(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_add_impl(ctx, a, b, false);
}

ସଂରଚନା ggml_tensor * ggml_add_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_add_cast_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଗଣନା   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_add_cast(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଗଣନା   ggml_type      type) {
    ଫେରନ୍ତୁ ggml_add_cast_impl(ctx, a, b, type);
}

ସଂରଚନା ggml_tensor * ggml_add_id(
            ସଂରଚନା ggml_context * ctx,
            ସଂରଚନା ggml_tensor  * a,
            ସଂରଚନା ggml_tensor  * b,
            ସଂରଚନା ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    ଫେରନ୍ତୁ result;
}

// ggml_add1

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_add1_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_add1(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_add1_impl(ctx, a, b, false);
}

ସଂରଚନା ggml_tensor * ggml_add1_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_acc_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_acc(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ସଂରଚନା ggml_tensor * ggml_acc_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_sub_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_sub(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_sub_impl(ctx, a, b, false);
}

ସଂରଚନା ggml_tensor * ggml_sub_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_mul_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_mul(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_mul_impl(ctx, a, b, false);
}

ସଂରଚନା ggml_tensor * ggml_mul_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_div_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_div(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_div_impl(ctx, a, b, false);
}

ସଂରଚନା ggml_tensor * ggml_div_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_sqr_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_sqr(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sqr_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_sqr_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_sqrt_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_sqrt(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sqrt_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_sqrt_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_log_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_log(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_log_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_log_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_log_impl(ctx, a, true);
}

ସଂରଚନା ggml_tensor * ggml_expm1(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ସଂରଚନା ggml_tensor * ggml_expm1_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ସଂରଚନା ggml_tensor * ggml_softplus(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ସଂରଚନା ggml_tensor * ggml_softplus_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_sin_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_sin(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sin_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_sin_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_cos_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_cos(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_cos_impl(ctx, a, false);
}

ସଂରଚନା ggml_tensor * ggml_cos_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ସଂରଚନା ggml_tensor * ggml_sum(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_sum_rows

ସଂରଚନା ggml_tensor * ggml_sum_rows(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_cumsum

ସଂରଚନା ggml_tensor * ggml_cumsum(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_mean

ସଂରଚନା ggml_tensor * ggml_mean(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_argmax

ସଂରଚନା ggml_tensor * ggml_argmax(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_count_equal

ସଂରଚନା ggml_tensor * ggml_ଗଣନା_equal(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_ଗଣନା_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_repeat

ସଂରଚନା ggml_tensor * ggml_repeat(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_repeat_4d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ଅଚଳ bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_repeat_back

ସଂରଚନା ggml_tensor * ggml_repeat_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_concat

ସଂରଚନା ggml_tensor * ggml_concat(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ସଂରଚନା ggml_tensor  * b,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < GGML_MAX_DIMS; ++d) {
        ଯଦି (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ଜାରି;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_abs

ସଂରଚନା ggml_tensor * ggml_abs(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ସଂରଚନା ggml_tensor * ggml_abs_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ସଂରଚନା ggml_tensor * ggml_sgn(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ସଂରଚନା ggml_tensor * ggml_sgn_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ସଂରଚନା ggml_tensor * ggml_neg(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ସଂରଚନା ggml_tensor * ggml_neg_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ସଂରଚନା ggml_tensor * ggml_step(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

ସଂରଚନା ggml_tensor * ggml_step_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

ସଂରଚନା ggml_tensor * ggml_tanh(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ସଂରଚନା ggml_tensor * ggml_tanh_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ସଂରଚନା ggml_tensor * ggml_elu(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ସଂରଚନା ggml_tensor * ggml_elu_inplace(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ସଂରଚନା ggml_tensor * ggml_relu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ସଂରଚନା ggml_tensor * ggml_relu_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ସଂରଚନା ggml_tensor * ggml_leaky_relu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 negative_slope,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, ଆକାର(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_sigmoid

ସଂରଚନା ggml_tensor * ggml_sigmoid(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ସଂରଚନା ggml_tensor * ggml_sigmoid_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ସଂରଚନା ggml_tensor * ggml_gelu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ସଂରଚନା ggml_tensor * ggml_gelu_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ସଂରଚନା ggml_tensor * ggml_gelu_erf(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ସଂରଚନା ggml_tensor * ggml_gelu_erf_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ସଂରଚନା ggml_tensor * ggml_gelu_quick(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

ସଂରଚନା ggml_tensor * ggml_gelu_quick_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

ସଂରଚନା ggml_tensor * ggml_silu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ସଂରଚନା ggml_tensor * ggml_silu_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ସଂରଚନା ggml_tensor * ggml_xielu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ alpha_n,
        ଦଶମିକ alpha_p,
        ଦଶମିକ beta,
        ଦଶମିକ eps) {
    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_silu_back

ସଂରଚନା ggml_tensor * ggml_silu_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml hardswish

ସଂରଚନା ggml_tensor * ggml_hardswish(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ସଂରଚନା ggml_tensor * ggml_hardsigmoid(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ସଂରଚନା ggml_tensor * ggml_exp(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ସଂରଚନା ggml_tensor * ggml_exp_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_glu_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଗଣନା ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    ଯଦି (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ଖାଲି, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_floor

ସଂରଚନା ggml_tensor * ggml_floor(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ସଂରଚନା ggml_tensor * ggml_floor_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ସଂରଚନା ggml_tensor * ggml_ceil(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ସଂରଚନା ggml_tensor * ggml_ceil_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ସଂରଚନା ggml_tensor * ggml_round(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

ସଂରଚନା ggml_tensor * ggml_round_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

ସଂରଚନା ggml_tensor * ggml_trunc(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ସଂରଚନା ggml_tensor * ggml_trunc_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ସଂରଚନା ggml_tensor * ggml_glu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_glu_op      op,
        bool                  swapped) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, op, swapped);
}

ସଂରଚନା ggml_tensor * ggml_glu_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଗଣନା ggml_glu_op      op) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ସଂରଚନା ggml_tensor * ggml_reglu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_REGLU, false);
}

ସଂରଚନା ggml_tensor * ggml_reglu_swapped(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_REGLU, true);
}

ସଂରଚନା ggml_tensor * ggml_reglu_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ସଂରଚନା ggml_tensor * ggml_geglu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU, false);
}

ସଂରଚନା ggml_tensor * ggml_geglu_swapped(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU, true);
}

ସଂରଚନା ggml_tensor * ggml_geglu_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ସଂରଚନା ggml_tensor * ggml_swiglu(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_SWIGLU, false);
}

ସଂରଚନା ggml_tensor * ggml_swiglu_swapped(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_SWIGLU, true);
}

ସଂରଚନା ggml_tensor * ggml_swiglu_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ସଂରଚନା ggml_tensor * ggml_geglu_erf(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU_ERF, false);
}

ସଂରଚନା ggml_tensor * ggml_geglu_erf_swapped(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU_ERF, true);
}

ସଂରଚନା ggml_tensor * ggml_geglu_erf_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ସଂରଚନା ggml_tensor * ggml_geglu_quick(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU_QUICK, false);
}

ସଂରଚନା ggml_tensor * ggml_geglu_quick_swapped(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, ଖାଲି, GGML_GLU_OP_GEGLU_QUICK, true);
}

ସଂରଚନା ggml_tensor * ggml_geglu_quick_split(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

ସଂରଚନା ggml_tensor * ggml_swiglu_oai(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 alpha,
        ଦଶମିକ                 limit) {
    ସଂରଚନା ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_swiglu_clamp(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 limit) {
    ସଂରଚନା ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    ଫେରନ୍ତୁ result;
}

// ggml_norm

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_norm_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ଆକାର(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_norm(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_norm_impl(ctx, a, eps, false);
}

ସଂରଚନା ggml_tensor * ggml_norm_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_rms_norm_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ଆକାର(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_rms_norm(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_rms_norm_impl(ctx, a, eps, false);
}

ସଂରଚନା ggml_tensor * ggml_rms_norm_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ସଂରଚନା ggml_tensor * ggml_rms_norm_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 eps) {
    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, ଆକାର(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_group_norm

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_group_norm_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_groups,
        ଦଶମିକ                 eps,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_group_norm(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_groups,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

ସଂରଚନା ggml_tensor * ggml_group_norm_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_groups,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_l2_norm_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_l2_norm(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_l2_norm_impl(ctx, a, eps, false);
}

ସଂରଚନା ggml_tensor * ggml_l2_norm_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 eps) {
    ଫେରନ୍ତୁ ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        ସଂରଚନା ggml_tensor * a,
        ଗଣନା ggml_prec       prec) {
    ବାଛ (a->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            {
                ଅଚଳ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            {
                ଅଚଳ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ false;
    };

    ଫେରନ୍ତୁ true;
}

bool ggml_prec_set_src(
        ସଂରଚନା ggml_tensor * a,
        ଗଣନା ggml_prec       prec,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ବାଛ (a->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            {
                ଯଦି (idx != 1) {
                    ଫେରନ୍ତୁ false;
                }

                ଅଚଳ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ false;
    };

    ଫେରନ୍ତୁ true;
}

// ggml_mul_mat

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_mul_mat(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ସଂରଚନା ggml_tensor * ggml_mul_mat(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ଅଚଳ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ ggml_mul_mat_set_prec(
        ସଂରଚନା ggml_tensor * a,
        ଗଣନା ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ଅଚଳ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

ଶୂନ୍ୟ ggml_mul_mat_set_hint(
        ସଂରଚନା ggml_tensor * a,
        ଗଣନା ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ଅଚଳ int32_t hint_i32 = (int32_t) hint;

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
ସଂରଚନା ggml_tensor * ggml_mul_mat_id(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * as,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ଅଚଳ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    ଫେରନ୍ତୁ result;
}

// ggml_out_prod

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_out_prod(ଅଚଳ ସଂରଚନା ggml_tensor * t0, ଅଚଳ ସଂରଚନା ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ସଂରଚନା ggml_tensor * ggml_out_prod(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ଅଚଳ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_scale

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_scale_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 s,
        ଦଶମିକ                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ଦଶମିକ params[2] = { s, b };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_scale(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 s) {
    ଫେରନ୍ତୁ ggml_scale_impl(ctx, a, s, 0.0, false);
}

ସଂରଚନା ggml_tensor * ggml_scale_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 s) {
    ଫେରନ୍ତୁ ggml_scale_impl(ctx, a, s, 0.0, true);
}

ସଂରଚନା ggml_tensor * ggml_scale_bias(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 s,
        ଦଶମିକ                 b) {
    ଫେରନ୍ତୁ ggml_scale_impl(ctx, a, s, b, false);
}

ସଂରଚନା ggml_tensor * ggml_scale_bias_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 s,
        ଦଶମିକ                 b) {
    ଫେରନ୍ତୁ ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_set_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_set(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ସଂରଚନା ggml_tensor * ggml_set_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

ସଂରଚନା ggml_tensor * ggml_set_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

ସଂରଚନା ggml_tensor * ggml_set_1d_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

ସଂରଚନା ggml_tensor * ggml_set_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

ସଂରଚନା ggml_tensor * ggml_set_2d_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ଫେରନ୍ତୁ ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_cpy_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, b);
    ଯଦି (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } ଅନ୍ୟଥା {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_cpy(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor * a,
        ସଂରଚନା ggml_tensor * b) {
    ଫେରନ୍ତୁ ggml_cpy_impl(ctx, a, b);
}

ସଂରଚନା ggml_tensor * ggml_cast(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା   ggml_type      type) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ଫେରନ୍ତୁ result;
}

// ggml_cont

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_cont_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_cont(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor * a) {
    ଫେରନ୍ତୁ ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ସଂରଚନା ggml_tensor * ggml_cont_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0) {
    ଫେରନ୍ତୁ ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ସଂରଚନା ggml_tensor * ggml_cont_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ଫେରନ୍ତୁ ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ସଂରଚନା ggml_tensor * ggml_cont_3d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ଫେରନ୍ତୁ ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ସଂରଚନା ggml_tensor * ggml_cont_4d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_reshape

ସଂରଚନା ggml_tensor * ggml_reshape(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor * a,
        ସଂରଚନା ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_reshape_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ଅଚଳ int64_t ne[1] = { ne0 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_reshape_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ଅଚଳ int64_t ne[2] = { ne0, ne1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_reshape_3d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ଅଚଳ int64_t ne[3] = { ne0, ne1, ne2 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_reshape_4d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ଅଚଳ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_view_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ଅଚଳ int64_t       * ne,
        size_t                offset) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, ଆକାର(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_view_1d

ସଂରଚନା ggml_tensor * ggml_view_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    ସଂରଚନା ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    ଫେରନ୍ତୁ result;
}

// ggml_view_2d

ସଂରଚନା ggml_tensor * ggml_view_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ଅଚଳ int64_t ne[2] = { ne0, ne1 };

    ସଂରଚନା ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    ଫେରନ୍ତୁ result;
}

// ggml_view_3d

ସଂରଚନା ggml_tensor * ggml_view_3d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ଅଚଳ int64_t ne[3] = { ne0, ne1, ne2 };

    ସଂରଚନା ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    ଫେରନ୍ତୁ result;
}

// ggml_view_4d

ସଂରଚନା ggml_tensor * ggml_view_4d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ଅଚଳ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ସଂରଚନା ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    ଫେରନ୍ତୁ result;
}

// ggml_permute

ସଂରଚନା ggml_tensor * ggml_permute(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   axis0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   axis1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   axis2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   axis3) {
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

    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, ଆକାର(params));

    ଫେରନ୍ତୁ result;
}

// ggml_transpose

ସଂରଚନା ggml_tensor * ggml_transpose(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_get_rows

ସଂରଚନା ggml_tensor * ggml_get_rows(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    ଗଣନା ggml_type type = GGML_TYPE_F32;
    ଯଦି (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_get_rows_back

ସଂରଚନା ggml_tensor * ggml_get_rows_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_set_rows

ସଂରଚନା ggml_tensor * ggml_set_rows(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c) {
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

    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ଫେରନ୍ତୁ result;
}

// ggml_diag

ସଂରଚନା ggml_tensor * ggml_diag(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ଅଚଳ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_diag_mask_inf

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_diag_mask_inf_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_diag_mask_inf(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past) {
    ଫେରନ୍ତୁ ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

ସଂରଚନା ggml_tensor * ggml_diag_mask_inf_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past) {
    ଫେରନ୍ତୁ ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_diag_mask_zero_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_diag_mask_zero(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past) {
    ଫେରନ୍ତୁ ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

ସଂରଚନା ggml_tensor * ggml_diag_mask_zero_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_past) {
    ଫେରନ୍ତୁ ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_clamp_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଦଶମିକ                 min,
        ଦଶମିକ                 max,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ଦଶମିକ params[] = { min, max };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_clamp(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଦଶମିକ                 min,
    ଦଶମିକ                 max) {
    ଫେରନ୍ତୁ ggml_clamp_impl(ctx, a, min, max, false);
}

ସଂରଚନା ggml_tensor * ggml_clamp_inplace(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଦଶମିକ                 min,
    ଦଶମିକ                 max) {
    ଫେରନ୍ତୁ ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_soft_max_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * mask,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    ଯଦି (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    ଯଦି (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ଦଶମିକ params[] = { scale, max_bias };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_soft_max(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_soft_max_impl(ctx, a, ଖାଲି, 1.0f, 0.0f, false);
}

ସଂରଚନା ggml_tensor * ggml_soft_max_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a) {
    ଫେରନ୍ତୁ ggml_soft_max_impl(ctx, a, ଖାଲି, 1.0f, 0.0f, true);
}

ସଂରଚନା ggml_tensor * ggml_soft_max_ext(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * mask,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias) {
    ଫେରନ୍ତୁ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

ସଂରଚନା ggml_tensor * ggml_soft_max_ext_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * mask,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias) {
    ଫେରନ୍ତୁ ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

ଶୂନ୍ୟ ggml_soft_max_add_sinks(
        ସଂରଚନା ggml_tensor * a,
        ସଂରଚନା ggml_tensor * sinks) {
    ଯଦି (!sinks) {
        a->src[2] = ଖାଲି;
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ଖାଲି);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_soft_max_ext_back_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias,
        bool                  inplace) {
    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((ଦଶମିକ *) result->op_params + 0, &scale,    ଆକାର(ଦଶମିକ));
    memcpy((ଦଶମିକ *) result->op_params + 1, &max_bias, ଆକାର(ଦଶମିକ));

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_soft_max_ext_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias) {
    ଫେରନ୍ତୁ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

ସଂରଚନା ggml_tensor * ggml_soft_max_ext_back_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias) {
    ଫେରନ୍ତୁ ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_rope_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   sections[GGML_MROPE_SECTIONS],
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    ଯଦି (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } ଅନ୍ୟଥା {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    ଯଦି (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    ଆକାର(ଦଶମିକ));
    memcpy(params +  6, &freq_scale,   ଆକାର(ଦଶମିକ));
    memcpy(params +  7, &ext_factor,   ଆକାର(ଦଶମିକ));
    memcpy(params +  8, &attn_factor,  ଆକାର(ଦଶମିକ));
    memcpy(params +  9, &beta_fast,    ଆକାର(ଦଶମିକ));
    memcpy(params + 10, &beta_slow,    ଆକାର(ଦଶମିକ));
    ଯଦି (mrope_used && sections) {
        memcpy(params + 11, sections,  ଆକାର(int32_t) * GGML_MROPE_SECTIONS);
    } ଅନ୍ୟଥା {
        memset(params + 11, 0,         ଆକାର(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_rope(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, ଖାଲି, n_dims, ଖାଲି, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_multi(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   sections[GGML_MROPE_SECTIONS],
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_multi_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   sections[GGML_MROPE_SECTIONS],
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, ଖାଲି, n_dims, ଖାଲି, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_ext(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, c, n_dims, ଖାଲି, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_ext_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, c, n_dims, ଖାଲି, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_custom(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, ଖାଲି, n_dims, ଖାଲି, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ସଂରଚନା ggml_tensor * ggml_rope_custom_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ଫେରନ୍ତୁ ggml_rope_impl(
        ctx, a, b, ଖାଲି, n_dims, ଖାଲି, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ସ୍ଥିର ଦଶମିକ ggml_rope_yarn_corr_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig, ଦଶମିକ n_rot, ଦଶମିକ base) {
    ଫେରନ୍ତୁ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ଦଶମିକ)M_PI)) / (2 * logf(base));
}

ଶୂନ୍ୟ ggml_rope_yarn_corr_dims(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig, ଦଶମିକ freq_base, ଦଶମିକ beta_fast, ଦଶମିକ beta_slow, ଦଶମିକ dims[2]
) {
    // start and end correction dims
    ଦଶମିକ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ଦଶମିକ end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

ସଂରଚନା ggml_tensor * ggml_rope_ext_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ସଂରଚନା ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_rope_multi_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_dims,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   sections[4],
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_ctx_orig,
        ଦଶମିକ                 freq_base,
        ଦଶମିକ                 freq_scale,
        ଦଶମିକ                 ext_factor,
        ଦଶମିକ                 attn_factor,
        ଦଶମିକ                 beta_fast,
        ଦଶମିକ                 beta_slow) {
    ସଂରଚନା ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_rope_set_offset(
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ଅଚଳ int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    ଫେରନ୍ତୁ a;
}

ସ୍ଥିର int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, ପୂର୍ଣ୍ଣସଂଖ୍ୟା s, ପୂର୍ଣ୍ଣସଂଖ୍ୟା p, ପୂର୍ଣ୍ଣସଂଖ୍ୟା d) {
    ଫେରନ୍ତୁ (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ସଂରଚନା ggml_tensor * ggml_im2col(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1,
        bool                  is_2D,
        ଗଣନା ggml_type        dst_type) {
    ଯଦି (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } ଅନ୍ୟଥା {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ଅଚଳ int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ଅଚଳ int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ଅଚଳ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_im2col_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        int64_t             * ne,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1,
        bool                  is_2D) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_conv_1d

ସଂରଚନା ggml_tensor * ggml_conv_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0) {
    ସଂରଚନା ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ସଂରଚନା ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    ଫେରନ୍ତୁ result;
}

// ggml_conv_1d_ph

ସଂରଚନା ggml_tensor* ggml_conv_1d_ph(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d) {
    ଫେରନ୍ତୁ ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ସଂରଚନା ggml_tensor * ggml_conv_1d_dw(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0) {
    ସଂରଚନା ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ସଂରଚନା ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ସଂରଚନା ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    ଫେରନ୍ତୁ result;
}

// ggml_conv_1d_dw_ph

ସଂରଚନା ggml_tensor * ggml_conv_1d_dw_ph(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0) {
    ଫେରନ୍ତୁ ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ସଂରଚନା ggml_tensor * ggml_col2im_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   oc,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ଅଚଳ int64_t K_OC = a->ne[0];
    ଅଚଳ int64_t T_in = a->ne[1];
    ଅଚଳ int64_t K = K_OC / oc;
    ଅଚଳ int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ଅଚଳ int64_t ne[4] = { T_out, oc, 1, 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_conv_transpose_1d

ସ୍ଥିର int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, ପୂର୍ଣ୍ଣସଂଖ୍ୟା s, ପୂର୍ଣ୍ଣସଂଖ୍ୟା p, ପୂର୍ଣ୍ଣସଂଖ୍ୟା d) {
    ଫେରନ୍ତୁ (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ସଂରଚନା ggml_tensor * ggml_conv_transpose_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ଅଚଳ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ସଂରଚନା ggml_tensor * ggml_conv_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1) {
    ସଂରଚନା ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ସଂରଚନା ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ଫେରନ୍ତୁ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ସଂରଚନା ggml_tensor * ggml_im2col_3d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        int64_t               IC,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0, // stride width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1, // stride height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s2, // stride depth
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0, // padding width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1, // padding height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p2, // padding depth
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0, // dilation width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1, // dilation height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d2, // dilation depth
        ଗଣନା ggml_type        dst_type) {
    ଅଚଳ int64_t N = b->ne[3] / IC;
    ଅଚଳ int64_t ID = b->ne[2];
    ଅଚଳ int64_t IH = b->ne[1];
    ଅଚଳ int64_t IW = b->ne[0];

    ଅଚଳ int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    ଅଚଳ int64_t KD = a->ne[2];
    ଅଚଳ int64_t KH = a->ne[1];
    ଅଚଳ int64_t KW = a->ne[0];
    ଅଚଳ int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    ଅଚଳ int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    ଅଚଳ int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ଅଚଳ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ସଂରଚନା ggml_tensor * ggml_conv_3d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        int64_t               IC,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0, // stride width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1, // stride height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s2, // stride depth
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0, // padding width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1, // padding height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p2, // padding depth
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0, // dilation width
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1, // dilation height
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d2  // dilation depth
        ) {
    ସଂରଚନା ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ସଂରଚନା ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ଫେରନ୍ତୁ result;
}

// ggml_conv_2d_sk_p0

ସଂରଚନା ggml_tensor * ggml_conv_2d_sk_p0(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ସଂରଚନା ggml_tensor * ggml_conv_2d_s1_ph(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    ଫେରନ୍ତୁ ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ସଂରଚନା ggml_tensor * ggml_conv_2d_dw(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1) {
    ସଂରଚନା ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ସଂରଚନା ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ସଂରଚନା ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ସଂରଚନା ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ଫେରନ୍ତୁ result;
}

// ggml_conv_2d_dw_direct

ସଂରଚନା ggml_tensor * ggml_conv_2d_dw_direct(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   stride0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   stride1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   pad0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   pad1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   dilation0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ଯଦି (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ଅଚଳ int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    ଫେରନ୍ତୁ result;
}

// ggml_conv_2d_direct

ସଂରଚନା ggml_tensor * ggml_conv_2d_direct(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ସଂରଚନା ggml_tensor  * b,   // input data [W, H, C, N]
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,  // stride dimension 0
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,  // stride dimension 1
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,  // padding dimension 0
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,  // padding dimension 1
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,  // dilation dimension 0
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_conv_3d_direct

ସଂରଚନା ggml_tensor * ggml_conv_3d_direct(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   d2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   c,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    ଫେରନ୍ତୁ result;
}

// ggml_conv_transpose_2d_p0

ସ୍ଥିର int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, ପୂର୍ଣ୍ଣସଂଖ୍ୟା s, ପୂର୍ଣ୍ଣସଂଖ୍ୟା p) {
    ଫେରନ୍ତୁ (ins - 1) * s - 2 * p + ks;
}

ସଂରଚନା ggml_tensor * ggml_conv_transpose_2d_p0(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ଅଚଳ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ସଂରଚନା ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_pool_*

ସ୍ଥିର int64_t ggml_calc_pool_output_size(int64_t ins, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ks, ପୂର୍ଣ୍ଣସଂଖ୍ୟା s, ଦଶମିକ p) {
    ଫେରନ୍ତୁ (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ସଂରଚନା ggml_tensor * ggml_pool_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_op_pool     op,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0) {
    ଅଚଳ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_pool_2d

ସଂରଚନା ggml_tensor * ggml_pool_2d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_op_pool     op,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ଦଶମିକ                 p0,
        ଦଶମିକ                 p1) {
    ସଂରଚନା ggml_tensor * result;
    ଅଚଳ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_pool_2d_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * af,
        ଗଣନା ggml_op_pool     op,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   s1,
        ଦଶମିକ                 p0,
        ଦଶମିକ                 p1) {
    ସଂରଚନା ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    ଫେରନ୍ତୁ result;
}

// ggml_upscale / ggml_interpolate

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_interpolate_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_ଗଣନା);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_upscale(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   scale_factor,
        ଗଣନା ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    ଫେରନ୍ତୁ ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

ସଂରଚନା ggml_tensor * ggml_upscale_ext(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   ne0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   ne1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   ne2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   ne3,
        ଗଣନା ggml_scale_mode  mode) {
    ଫେରନ୍ତୁ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

ସଂରଚନା ggml_tensor * ggml_interpolate(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    ଫେରନ୍ତୁ ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

ସଂରଚନା ggml_tensor * ggml_pad(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p3) {
    ଫେରନ୍ତୁ ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ସଂରଚନା ggml_tensor * ggml_pad_circular(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p3) {
    ଫେରନ୍ତୁ ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ସଂରଚନା ggml_tensor * ggml_pad_ext(
            ସଂରଚନା ggml_context * ctx,
            ସଂରଚନା ggml_tensor  * a,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp0,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp0,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp1,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp1,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp2,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp2,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp3,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp3
            ) {
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    ଫେରନ୍ତୁ result;
}

// ggml_pad_ext_circular

ସଂରଚନା ggml_tensor * ggml_pad_ext_circular(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  lp3,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  rp3
        ) {
    ସଂରଚନା ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    ଫେରନ୍ତୁ result;
}

// ggml_pad_reflect_1d

ସଂରଚନା ggml_tensor * ggml_pad_reflect_1d(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_roll

ସଂରଚନା ggml_tensor * ggml_roll(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   shift0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   shift1,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   shift2,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_timestep_embedding

ସଂରଚନା ggml_tensor * ggml_timestep_embedding(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * timesteps,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   dim,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   max_period) {

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    ଫେରନ୍ତୁ result;
}

// ggml_tri

ସଂରଚନା ggml_tensor * ggml_tri(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଗଣନା ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_fill

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_fill_impl(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଦଶମିକ                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_fill(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଦଶମିକ                 c) {
    ଫେରନ୍ତୁ ggml_fill_impl(ctx, a, c, false);
}

ସଂରଚନା ggml_tensor * ggml_fill_inplace(
    ସଂରଚନା ggml_context * ctx,
    ସଂରଚନା ggml_tensor  * a,
    ଦଶମିକ                 c) {
    ଫେରନ୍ତୁ ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ସଂରଚନା ggml_tensor * ggml_argsort(
        ସଂରଚନା ggml_context  * ctx,
        ସଂରଚନା ggml_tensor   * a,
        ଗଣନା ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_argsort_top_k

ସଂରଚନା ggml_tensor * ggml_argsort_ଶୀର୍ଷ_k(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ସଂରଚନା ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    ଫେରନ୍ତୁ result;
}

// ggml_top_k

ସଂରଚନା ggml_tensor * ggml_ଶୀର୍ଷ_k(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_ଶୀର୍ଷ_K;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_arange

ସଂରଚନା ggml_tensor * ggml_arange(
        ସଂରଚନା ggml_context * ctx,
        ଦଶମିକ                 start,
        ଦଶମିକ                 stop,
        ଦଶମିକ                 step) {
    GGML_ASSERT(stop > start);

    ଅଚଳ int64_t steps = (int64_t) ceilf((stop - start) / step);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    ଫେରନ୍ତୁ result;
}

// ggml_flash_attn_ext

ସଂରଚନା ggml_tensor * ggml_flash_attn_ext(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * q,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * mask,
        ଦଶମିକ                 scale,
        ଦଶମିକ                 max_bias,
        ଦଶମିକ                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    ଯଦି (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    ଯଦି (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ଦଶମିକ params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    ଫେରନ୍ତୁ result;
}


ଶୂନ୍ୟ ggml_flash_attn_ext_set_prec(
        ସଂରଚନା ggml_tensor * a,
        ଗଣନା ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ଅଚଳ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

ଗଣନା ggml_prec ggml_flash_attn_ext_get_prec(
        ଅଚଳ ସଂରଚନା ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ଅଚଳ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ଫେରନ୍ତୁ (ଗଣନା ggml_prec) prec_i32;
}

ଶୂନ୍ୟ ggml_flash_attn_ext_set_n_kv_max(
        ସଂରଚନା ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

ଶୂନ୍ୟ ggml_flash_attn_ext_add_sinks(
        ସଂରଚନା ggml_tensor * a,
        ସଂରଚନା ggml_tensor * sinks) {
    ଯଦି (!sinks) {
        a->src[4] = ଖାଲି;
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ଖାଲି);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ସଂରଚନା ggml_tensor * ggml_flash_attn_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * q,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ଅଚଳ int64_t     D = q->ne[0];
    ଅଚଳ int64_t     N = q->ne[1];
    ଅଚଳ int64_t     M = k->ne[1];
    ଅଚଳ int64_t   ne2 = q->ne[2];
    ଅଚଳ int64_t   ne3 = q->ne[3];
    ଅଚଳ int64_t kvne2 = k->ne[2];

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
    ଅଚଳ int64_t elem_q = ggml_nelements(q);
    ଅଚଳ int64_t elem_k = ggml_nelements(k);
    ଅଚଳ int64_t elem_v = ggml_nelements(v);

    ଗଣନା ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ଅଚଳ size_t tsize = ggml_type_size(result_type);

    ଅଚଳ size_t offs_q = 0;
    ଅଚଳ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ଅଚଳ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ଅଚଳ size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ଅଚଳ size_t nelements = (end + tsize - 1)/tsize;

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, ଆକାର(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    ଫେରନ୍ତୁ result;
}

// ggml_ssm_conv

ସଂରଚନା ggml_tensor * ggml_ssm_conv(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * sx,
        ସଂରଚନା ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ଅଚଳ int64_t d_conv  = c->ne[0];
    ଅଚଳ int64_t d_inner = c->ne[1];
    ଅଚଳ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ଅଚଳ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    ଫେରନ୍ତୁ result;
}

// ggml_ssm_scan

ସଂରଚନା ggml_tensor * ggml_ssm_scan(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * s,
        ସଂରଚନା ggml_tensor  * x,
        ସଂରଚନା ggml_tensor  * dt,
        ସଂରଚନା ggml_tensor  * A,
        ସଂରଚନା ggml_tensor  * B,
        ସଂରଚନା ggml_tensor  * C,
        ସଂରଚନା ggml_tensor  * ids,
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
        ଅଚଳ int64_t d_state      = s->ne[0];
        ଅଚଳ int64_t head_dim     = x->ne[0];
        ଅଚଳ int64_t n_head       = x->ne[1];
        ଅଚଳ int64_t n_seq_tokens = x->ne[2];
        ଅଚଳ int64_t n_seqs       = x->ne[3];

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

        ଯଦି (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    ଫେରନ୍ତୁ result;
}

// ggml_win_part

ସଂରଚନା ggml_tensor * ggml_win_part(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା px = (w - a->ne[1]%w)%w;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା py = (w - a->ne[2]%w)%w;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା npx = (px + a->ne[1])/w;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା npy = (py + a->ne[2])/w;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np  = npx*npy;

    ଅଚଳ int64_t ne[4] = { a->ne[0], w, w, np, };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_win_unpart

ସଂରଚନା ggml_tensor * ggml_win_unpart(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   w0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   h0,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ଅଚଳ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, ଆକାର(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_get_rel_pos

ସଂରଚନା ggml_tensor * ggml_get_rel_pos(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   qh,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ଅଚଳ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

// ggml_add_rel_pos

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_add_rel_pos_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * pw,
        ସଂରଚନା ggml_tensor  * ph,
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

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_add_rel_pos(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * pw,
        ସଂରଚନା ggml_tensor  * ph) {
    ଫେରନ୍ତୁ ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

ସଂରଚନା ggml_tensor * ggml_add_rel_pos_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * pw,
        ସଂରଚନା ggml_tensor  * ph) {
    ଫେରନ୍ତୁ ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ସଂରଚନା ggml_tensor * ggml_rwkv_wkv6(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * r,
        ସଂରଚନା ggml_tensor  * tf,
        ସଂରଚନା ggml_tensor  * td,
        ସଂରଚନା ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ଅଚଳ int64_t S = k->ne[0];
    ଅଚଳ int64_t H = k->ne[1];
    ଅଚଳ int64_t n_tokens = k->ne[2];
    ଅଚଳ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ଅଚଳ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    ଫେରନ୍ତୁ result;
}

// ggml_gated_linear_attn

ସଂରଚନା ggml_tensor * ggml_gated_linear_attn(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * q,
        ସଂରଚନା ggml_tensor  * g,
        ସଂରଚନା ggml_tensor  * state,
        ଦଶମିକ scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ଅଚଳ int64_t S = k->ne[0];
    ଅଚଳ int64_t H = k->ne[1];
    ଅଚଳ int64_t n_tokens = k->ne[2];
    ଅଚଳ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ଅଚଳ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    ଫେରନ୍ତୁ result;
}

// ggml_rwkv_wkv7

ସଂରଚନା ggml_tensor * ggml_rwkv_wkv7(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * r,
        ସଂରଚନା ggml_tensor  * w,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ଅଚଳ int64_t S = k->ne[0];
    ଅଚଳ int64_t H = k->ne[1];
    ଅଚଳ int64_t n_tokens = k->ne[2];
    ଅଚଳ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ଅଚଳ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    ଫେରନ୍ତୁ result;
}

// ggml_unary

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_unary_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_unary(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_unary_op    op) {
    ଫେରନ୍ତୁ ggml_unary_impl(ctx, a, op, false);
}

ସଂରଚନା ggml_tensor * ggml_unary_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ଗଣନା ggml_unary_op    op) {
    ଫେରନ୍ତୁ ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_map_custom1_impl(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ଅଚଳ  ggml_custom1_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ସଂରଚନା ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_map_custom1(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ଅଚଳ  ggml_custom1_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ସଂରଚନା ggml_tensor * ggml_map_custom1_inplace(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ଅଚଳ  ggml_custom1_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_map_custom2_impl(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ଅଚଳ  ggml_custom2_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ସଂରଚନା ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_map_custom2(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ଅଚଳ  ggml_custom2_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ସଂରଚନା ggml_tensor * ggml_map_custom2_inplace(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ଅଚଳ  ggml_custom2_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_map_custom3_impl(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ସଂରଚନା ggml_tensor       * c,
        ଅଚଳ  ggml_custom3_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ସଂରଚନା ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ସଂରଚନା ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_map_custom3(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ସଂରଚନା ggml_tensor       * c,
        ଅଚଳ  ggml_custom3_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ସଂରଚନା ggml_tensor * ggml_map_custom3_inplace(
        ସଂରଚନା ggml_context      * ctx,
        ସଂରଚନା ggml_tensor       * a,
        ସଂରଚନା ggml_tensor       * b,
        ସଂରଚନା ggml_tensor       * c,
        ଅଚଳ  ggml_custom3_op_t   fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        n_tasks,
        ଶୂନ୍ୟ                     * userdata) {
    ଫେରନ୍ତୁ ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ସଂରଚନା ggml_tensor * ggml_custom_4d(
        ସଂରଚନା ggml_context * ctx,
        ଗଣନା ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ସଂରଚନା ggml_tensor ** args,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_args,
        ggml_custom_op_t      fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_tasks,
        ଶୂନ୍ୟ                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ସଂରଚନା ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op = GGML_OP_CUSTOM;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_custom_inplace(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor ** args,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_args,
        ggml_custom_op_t      fun,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_tasks,
        ଶୂନ୍ୟ                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);

    ସଂରଚନା ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, ଆକାର(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    ଫେରନ୍ତୁ result;
}
// ggml_cross_entropy_loss

ସଂରଚନା ggml_tensor * ggml_cross_entropy_loss(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_cross_entropy_loss_back

ସଂରଚନା ggml_tensor * ggml_cross_entropy_loss_back(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        ସଂରଚନା ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ସଂରଚନା ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ଫେରନ୍ତୁ result;
}

// opt_step_adamw

ସଂରଚନା ggml_tensor * ggml_opt_step_adamw(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * grad,
        ସଂରଚନା ggml_tensor  * m,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    ଫେରନ୍ତୁ result;
}

// opt_step_sgd

ସଂରଚନା ggml_tensor * ggml_opt_step_sgd(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * grad,
        ସଂରଚନା ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ସଂରଚନା ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    ଫେରନ୍ତୁ result;
}

// solve_tri

ସଂରଚନା ggml_tensor * ggml_solve_tri(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * a,
        ସଂରଚନା ggml_tensor  * b,
        bool                  ବାମ,
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

    GGML_ASSERT(lower && ବାମ && !uni); // TODO: support other variants

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    ଫେରନ୍ତୁ result;
}

// ggml_gated_delta_net

ସଂରଚନା ggml_tensor * ggml_gated_delta_net(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * q,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * v,
        ସଂରଚନା ggml_tensor  * g,
        ସଂରଚନା ggml_tensor  * beta,
        ସଂରଚନା ggml_tensor  * state,
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

    ଅଚଳ int64_t S_v      = v->ne[0];
    ଅଚଳ int64_t H        = v->ne[1];
    ଅଚଳ int64_t n_tokens = v->ne[2];
    ଅଚଳ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ଅଚଳ int64_t state_rows = K * S_v * n_seqs;
    ଅଚଳ int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    ଫେରନ୍ତୁ result;
}

// ggml_lightning_indexer

ସଂରଚନା ggml_tensor * ggml_lightning_indexer(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * q,
        ସଂରଚନା ggml_tensor  * k,
        ସଂରଚନା ggml_tensor  * weights,
        ସଂରଚନା ggml_tensor  * mask) {

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
    ସଂରଚନା ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    ଫେରନ୍ତୁ result;
}

// ggml_dsv4_hc_comb

ସଂରଚନା ggml_tensor * ggml_dsv4_hc_comb(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * mixes,
        ସଂରଚନା ggml_tensor  * scale,
        ସଂରଚନା ggml_tensor  * base,
        ଦଶମିକ                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ଅଚଳ int64_t hc_mix_dim = mixes->ne[0];
    ଅଚଳ int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    ପାଇଁ (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        ଯଦି ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ଭାଙ୍ଗ;
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

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    ଫେରନ୍ତୁ result;
}

// ggml_dsv4_hc_pre

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_dsv4_hc_pre_impl(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * x,
        ସଂରଚନା ggml_tensor  * weights,
        ଦଶମିକ                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ଅଚଳ int64_t n_embd   = x->ne[0];
    ଅଚଳ int64_t hc       = x->ne[1];
    ଅଚଳ int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    ଯଦି (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } ଅନ୍ୟଥା {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_dsv4_hc_pre(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * x,
        ସଂରଚନା ggml_tensor  * weights) {
    ଫେରନ୍ତୁ ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ସଂରଚନା ggml_tensor * ggml_dsv4_hc_pre_gated(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * x,
        ସଂରଚନା ggml_tensor  * gate,
        ଦଶମିକ                 scale) {
    ଫେରନ୍ତୁ ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ସଂରଚନା ggml_tensor * ggml_dsv4_hc_post(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * x,
        ସଂରଚନା ggml_tensor  * residual,
        ସଂରଚନା ggml_tensor  * post,
        ସଂରଚନା ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ଅଚଳ int64_t n_embd   = x->ne[0];
    ଅଚଳ int64_t n_tokens = x->ne[1];
    ଅଚଳ int64_t hc       = residual->ne[1];

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

    ଯଦି (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    ଫେରନ୍ତୁ result;
}

// ggml_top_k_mask

ସଂରଚନା ggml_tensor * ggml_ଶୀର୍ଷ_k_mask(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_tensor  * mask,
        ସଂରଚନା ggml_tensor  * ଶୀର୍ଷ_k,
        ସଂରଚନା ggml_tensor  * score,
        ସଂରଚନା ggml_tensor  * blocks,
        ସଂରଚନା ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(ଶୀର୍ଷ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(ଶୀର୍ଷ_k->ne[1] % mask->ne[1] == 0 && ଶୀର୍ଷ_k->ne[2] % mask->ne[2] == 0 && ଶୀର୍ଷ_k->ne[3] % mask->ne[3] == 0);

    ଯଦି (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == ଶୀର୍ଷ_k->ne[1] && score->ne[2] == ଶୀର୍ଷ_k->ne[2] && score->ne[3] == ଶୀର୍ଷ_k->ne[3]);
    }

    ଯଦି (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    ଯଦି (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == ଶୀର୍ଷ_k->ne[1] && extra->ne[2] == ଶୀର୍ଷ_k->ne[2] && extra->ne[3] == ଶୀର୍ଷ_k->ne[3]);
    }

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], ଶୀର୍ଷ_k->ne[1], ଶୀର୍ଷ_k->ne[2], ଶୀର୍ଷ_k->ne[3]);

    result->op     = GGML_OP_ଶୀର୍ଷ_K_MASK;
    result->src[0] = mask;
    result->src[1] = ଶୀର୍ଷ_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    ଫେରନ୍ତୁ result;
}

////////////////////////////////////////////////////////////////////////////////

ସଂରଚନା ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    ସଂରଚନା ggml_hash_set result;
    result.size = size;
    result.ଚାବିଗୁଡ଼ିକ = GGML_MALLOC(ଆକାର(ସଂରଚନା ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), ଆକାର(ggml_bitset_t));
    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ ggml_hash_set_reset(ସଂରଚନା ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, ଆକାର(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

ଶୂନ୍ୟ ggml_hash_set_free(ସଂରଚନା ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->ଚାବିଗୁଡ଼ିକ);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    ସ୍ଥିର ଅଚଳ size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ସ୍ଥିର ଅଚଳ size_t n_primes = ଆକାର(primes)/ଆକାର(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    ଯେତେବେଳେ (l < r) {
        size_t m = (l + r)/2;
        ଯଦି (primes[m] < min_sz) {
            l = m + 1;
        } ଅନ୍ୟଥା {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    ଫେରନ୍ତୁ sz;
}

ସଂରଚନା hash_map {
    ସଂରଚନା ggml_hash_set set;
    ସଂରଚନା ggml_tensor ** vals;
};

ସ୍ଥିର ସଂରଚନା hash_map * ggml_new_hash_map(size_t size) {
    ସଂରଚନା hash_map * result = GGML_MALLOC(ଆକାର(ସଂରଚନା hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, ଆକାର(ସଂରଚନା ggml_tensor *));
    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hash_map_free(ସଂରଚନା hash_map * map) {
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

ସ୍ଥିର ଶୂନ୍ୟ ggml_add_or_set(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_cgraph  * cgraph,
        size_t                isrc,
        ସଂରଚନା ggml_tensor  * tensor) {
    ସଂରଚନା ggml_tensor * src = cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[isrc];
    GGML_ASSERT(src);
    ଯଦି (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } ଅନ୍ୟଥା {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_acc_or_set(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_cgraph  * cgraph,
        size_t                isrc,
        ସଂରଚନା ggml_tensor  * tensor,
        ଅଚଳ  size_t         nb1,
        ଅଚଳ  size_t         nb2,
        ଅଚଳ  size_t         nb3,
        ଅଚଳ  size_t         offset) {
    ସଂରଚନା ggml_tensor * src = cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[isrc];
    GGML_ASSERT(src);
    ଯଦି (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } ଅନ୍ୟଥା {
        ସଂରଚନା ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_add1_or_set(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_cgraph  * cgraph,
        size_t                isrc,
        ସଂରଚନା ggml_tensor  * tensor) {
    ସଂରଚନା ggml_tensor * src = cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[isrc];
    GGML_ASSERT(src);
    ଯଦି (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ଅନ୍ୟଥା {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sub_or_set(
        ସଂରଚନା ggml_context * ctx,
        ସଂରଚନା ggml_cgraph  * cgraph,
        size_t                isrc,
        ସଂରଚନା ggml_tensor  * tensor) {
    ସଂରଚନା ggml_tensor * src = cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[isrc];
    GGML_ASSERT(src);
    ଯଦି (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } ଅନ୍ୟଥା {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_backward(
        ସଂରଚନା ggml_context * ctx, ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଅଚଳ bool * grads_needed) {
    ସଂରଚନା ggml_tensor * tensor = cgraph->nodes[i];
    ସଂରଚନା ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    ଯଦି (!grad) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା ggml_tensor * src0 = tensor->src[0];
    ସଂରଚନା ggml_tensor * src1 = tensor->src[1];
    ସଂରଚନା ggml_tensor * src2 = tensor->src[2];
    ସଂରଚନା ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ଅଚଳ size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ଅଚଳ size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ଅଚଳ size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ଅଚଳ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ଅଚଳ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ଅଚଳ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    ବାଛ (tensor->op) {
        ସ୍ଥିତି GGML_OP_DUP: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ଯଦି (src1_needs_grads) {
                ସଂରଚନା ggml_tensor * tmp = grad;
                ଯଦି (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD1: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ଯଦି (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ACC: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ଯଦି (src1_needs_grads) {
                ଅଚଳ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ଅଚଳ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ଅଚଳ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ଅଚଳ size_t offset = ((int32_t *) tensor->op_params)[3];

                ସଂରଚନା ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUB: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            ଯଦି (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            ଯଦି (src1_needs_grads) {
                ସଂରଚନା ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                ଯଦି (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIV: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            ଯଦି (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQR: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQRT: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LOG: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SIN: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_COS: {
            ଯଦି (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM: {
            ଯଦି (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM_ROWS: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MEAN: {
            ଯଦି (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT_BACK: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM: {
            ଯଦି (src0_needs_grads) {
                ଦଶମିକ eps;
                memcpy(&eps, tensor->op_params, ଆକାର(ଦଶମିକ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT: {
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

            ଯଦି (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ସଂରଚନା ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                ଯଦି (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ଅଚଳ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ଅଚଳ size_t nb2 = tmp->nb[2] * nr2;
                    ଅଚଳ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            ଯଦି (src1_needs_grads) {
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
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SCALE: {
            ଯଦି (src0_needs_grads) {
                ଦଶମିକ s;
                memcpy(&s, tensor->op_params, ଆକାର(ଦଶମିକ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET: {
            ଅଚଳ size_t nb1    = ((ଅଚଳ int32_t *) tensor->op_params)[0];
            ଅଚଳ size_t nb2    = ((ଅଚଳ int32_t *) tensor->op_params)[1];
            ଅଚଳ size_t nb3    = ((ଅଚଳ int32_t *) tensor->op_params)[2];
            ଅଚଳ size_t offset = ((ଅଚଳ int32_t *) tensor->op_params)[3];

            ସଂରଚନା ggml_tensor * tensor_grad_view = ଖାଲି;

            ଯଦି (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            ଯଦି (src0_needs_grads) {
                ସଂରଚନା ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            ଯଦି (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            ଯଦି (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            ଯଦି (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONT: {
            // same as cpy
            ଯଦି (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RESHAPE: {
            ଯଦି (src0_needs_grads) {
                ସଂରଚନା ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_VIEW: {
            ଯଦି (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, ଆକାର(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                ଯଦି (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PERMUTE: {
            ଯଦି (src0_needs_grads) {
                ଅଚଳ int32_t * axes = (ଅଚଳ int32_t *) tensor->op_params;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis0 = axes[0] & 0x3;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis1 = axes[1] & 0x3;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis2 = axes[2] & 0x3;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis3 = axes[3] & 0x3;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TRANSPOSE: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            ଯଦି (src1_needs_grads) {
                // noop
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF: {
            ଯଦି (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_past = ((ଅଚଳ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_ZERO: {
            ଯଦି (src0_needs_grads) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_past = ((ଅଚଳ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX: {
            ଯଦି (src0_needs_grads) {
                ଦଶମିକ scale    = 1.0f;
                ଦଶମିକ max_bias = 0.0f;

                memcpy(&scale,    (ଅଚଳ ଦଶମିକ *) tensor->op_params + 0, ଆକାର(ଦଶମିକ));
                memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) tensor->op_params + 1, ଆକାର(ଦଶମିକ));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE: {
            ଯଦି (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims     = ((ଅଚଳ int32_t *) tensor->op_params)[1];
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode       = ((ଅଚଳ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig = ((ଅଚଳ int32_t *) tensor->op_params)[4];
                ଦଶମିକ freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ଅଚଳ ଦଶମିକ *) tensor->op_params +  5, ଆକାର(ଦଶମିକ));
                memcpy(&freq_scale,  (ଅଚଳ ଦଶମିକ *) tensor->op_params +  6, ଆକାର(ଦଶମିକ));
                memcpy(&ext_factor,  (ଅଚଳ ଦଶମିକ *) tensor->op_params +  7, ଆକାର(ଦଶମିକ));
                memcpy(&attn_factor, (ଅଚଳ ଦଶମିକ *) tensor->op_params +  8, ଆକାର(ଦଶମିକ));
                memcpy(&beta_fast,   (ଅଚଳ ଦଶମିକ *) tensor->op_params +  9, ଆକାର(ଦଶମିକ));
                memcpy(&beta_slow,   (ଅଚଳ ଦଶମିକ *) tensor->op_params + 10, ଆକାର(ଦଶମିକ));
                memcpy(&sections,                    tensor->op_params + 11, ଆକାର(sections));

                ସଂରଚନା ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL: {
            ଯଦି (src1_needs_grads) {
                ଅଚଳ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ଅଚଳ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ଅଚଳ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ଅଚଳ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ଅଚଳ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ଅଚଳ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ଅଚଳ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_2D: {
            ଯଦି (src0_needs_grads) {
                ଅଚଳ ଗଣନା ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ଅଚଳ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ଅଚଳ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ଅଚଳ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ଅଚଳ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ଅଚଳ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ଅଚଳ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_WIN_PART:
        ସ୍ଥିତି GGML_OP_WIN_UNPART:
        ସ୍ଥିତି GGML_OP_UNARY: {
            ବାଛ (ggml_get_unary_op(tensor)) {
                ସ୍ଥିତି GGML_UNARY_OP_ABS: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SGN: {
                    // noop
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_NEG: {
                    ଯଦି (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_STEP: {
                    // noop
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_RELU: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SILU: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_EXP: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_EXPM1: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS: {
                    ଯଦି (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS: {
            ଯଦି (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GLU: {
            ବାଛ (ggml_get_glu_op(tensor)) {
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU: {
                    ଯଦି (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    ଯଦି (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NONE: {
            // noop
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା:
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ସ୍ଥିର size_t ggml_visit_parents_graph(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, bool compute) {
    ଯଦି (ଗଣ୍ଠି->op != GGML_OP_NONE && compute) {
        ଗଣ୍ଠି->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ଅଚଳ size_t ଗଣ୍ଠି_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି);
    GGML_ASSERT(ଗଣ୍ଠି_hash_pos != GGML_HASHSET_FULL);

    ଯଦି (ggml_bitset_get(cgraph->visited_hash_set.used, ଗଣ୍ଠି_hash_pos)) {
        // already visited

        ଯଦି (compute) {
            // update the compute flag regardless
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; ++i) {
                ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[i];
                ଯଦି (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        ଫେରନ୍ତୁ ଗଣ୍ଠି_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.ଚାବିଗୁଡ଼ିକ[ଗଣ୍ଠି_hash_pos] = ଗଣ୍ଠି;
    ggml_bitset_set(cgraph->visited_hash_set.used, ଗଣ୍ଠି_hash_pos);
    cgraph->use_counts[ଗଣ୍ଠି_hash_pos] = 0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ବାମ_TO_ଡାହାଣ) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ଡାହାଣ_TO_ବାମ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ସଂରଚନା ggml_tensor * src = ଗଣ୍ଠି->src[k];
        ଯଦି (src) {
            ଅଚଳ size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    ଯଦି (ଗଣ୍ଠି->op == GGML_OP_NONE && !(ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        ଯଦି (strlen(ଗଣ୍ଠି->name) == 0) {
            ggml_format_name(ଗଣ୍ଠି, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ଗଣ୍ଠି;
        cgraph->n_leafs++;
    } ଅନ୍ୟଥା {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        ଯଦି (strlen(ଗଣ୍ଠି->name) == 0) {
            ggml_format_name(ଗଣ୍ଠି, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ଗଣ୍ଠି;
        cgraph->n_nodes++;
    }

    ଫେରନ୍ତୁ ଗଣ୍ଠି_hash_pos;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_build_forward_impl(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_tensor * tensor, bool expand, bool compute) {
    ଯଦି (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    ଯଦି (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ସଂରଚନା ggml_tensor * ggml_build_forward_select(
        ସଂରଚନା ggml_cgraph  * cgraph,
        ସଂରଚନା ggml_tensor ** tensors,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   n_tensors,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ଫେରନ୍ତୁ tensors[idx];
}

ଶୂନ୍ୟ ggml_build_forward_expand(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

ଶୂନ୍ୟ ggml_build_forward_order(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

ଶୂନ୍ୟ ggml_build_backward_expand(
        ସଂରଚନା ggml_context *  ctx,
        ସଂରଚନା ggml_cgraph  *  cgraph,
        ସଂରଚନା ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*ଆକାର(ସଂରଚନା ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*ଆକାର(ସଂରଚନା ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, ଆକାର(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_nodes_f; ++i) {
            ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
            any_params = any_params || (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_nodes_f; ++i) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_I32) {
            ଜାରି;
        }

        bool ଗଣ୍ଠି_needs_grad = (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM) || (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ବାଛ (ଗଣ୍ଠି->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            ସ୍ଥିତି GGML_OP_IM2COL:      // only used for its shape
            ସ୍ଥିତି GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_UNARY: {
                ଅଚଳ ଗଣନା ggml_unary_op uop = ggml_get_unary_op(ଗଣ୍ଠି);
                // SGN and STEP unary ops are piecewise constant
                ଯଦି (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } ଭାଙ୍ଗ;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            ସ୍ଥିତି GGML_OP_CPY:           // gradients in CPY target are irrelevant
            ସ୍ଥିତି GGML_OP_GET_ROWS:      // row indices not differentiable
            ସ୍ଥିତି GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            ସ୍ଥିତି GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ଭାଙ୍ଗ;

            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; ++j) {
            ଯଦି (!ଗଣ୍ଠି->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି->src[j])]) {
                ଜାରି;
            }
            GGML_ASSERT(ଗଣ୍ଠି->src[j]->type == GGML_TYPE_F32 || ଗଣ୍ଠି->src[j]->type == GGML_TYPE_F16);
            ଗଣ୍ଠି_needs_grad = true;
            ଭାଙ୍ଗ;
        }
        ଯଦି (!ଗଣ୍ଠି_needs_grad) {
            ଜାରି;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ଗଣ୍ଠି->view_src || ଗଣ୍ଠି->op == GGML_OP_CPY || ଗଣ୍ଠି->op == GGML_OP_SET_ROWS || ଗଣ୍ଠି->op == GGML_OP_VIEW ||
            ଗଣ୍ଠି->op == GGML_OP_RESHAPE || ଗଣ୍ଠି->op == GGML_OP_PERMUTE || ଗଣ୍ଠି->op == GGML_OP_TRANSPOSE);

        ଅଚଳ size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        ଯଦି (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ଗଣ୍ଠି->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    ମୁକ୍ତ(grads_needed);
}

ସ୍ଥିର ଶୂନ୍ୟ * incr_ptr_aligned(ଶୂନ୍ୟ ** p, size_t size, size_t align) {
    ଶୂନ୍ୟ * ptr = *p;
    ptr = (ଶୂନ୍ୟ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) ptr + size);
    ଫେରନ୍ତୁ ptr;
}

ସ୍ଥିର size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    ଶୂନ୍ୟ * p = (ଅକ୍ଷର *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, ଆକାର(ସଂରଚନା ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * ଆକାର(int32_t), ଆକାର(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)); // hash keys
    ଯଦି (grads) {
        incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * ଆକାର(ggml_bitset_t), ଆକାର(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ଫେରନ୍ତୁ nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    ଫେରନ୍ତୁ GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

ସଂରଚନା ggml_cgraph * ggml_new_graph_custom(ସଂରଚନା ggml_context * ctx, size_t size, bool grads) {
    ଅଚଳ size_t obj_size = ggml_graph_nbytes(size, grads);
    ସଂରଚନା ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    ସଂରଚନା ggml_cgraph * cgraph = (ସଂରଚନା ggml_cgraph *) ((ଅକ୍ଷର *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    ଶୂନ୍ୟ * p = cgraph + 1;

    ସଂରଚନା ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *));
    ସଂରଚନା ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * ଆକାର(int32_t), ଆକାର(int32_t));
    ସଂରଚନା ggml_tensor ** hash_ଚାବିଗୁଡ଼ିକ_ptr  =         incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *));
    ସଂରଚନା ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)) : ଖାଲି;
    ସଂରଚନା ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * ଆକାର(ସଂରଚନା ggml_tensor *), ଆକାର(ସଂରଚନା ggml_tensor *)) : ଖାଲି;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * ଆକାର(ggml_bitset_t), ଆକାର(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ପ୍ରତିଜ୍ଞା(obj_size == (size_t)((ଅକ୍ଷର *)p - (ଅକ୍ଷର *)cgraph));

    *cgraph = (ସଂରଚନା ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_ଚାବିଗୁଡ଼ିକ_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_ବାମ_TO_ଡାହାଣ,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    ଯଦି (grads) {
        memset(cgraph->grads,     0, hash_size*ଆକାର(ସଂରଚନା ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*ଆକାର(ସଂରଚନା ggml_tensor *));
    }

    ଫେରନ୍ତୁ cgraph;
}

ସଂରଚନା ggml_cgraph * ggml_new_graph(ସଂରଚନା ggml_context * ctx) {
    ଫେରନ୍ତୁ ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

ସଂରଚନା ggml_cgraph ggml_graph_view(ସଂରଚନା ggml_cgraph * cgraph0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1) {
    ସଂରଚନା ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ଖାଲି, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ଖାଲି,
        /*.leafs            =*/ ଖାଲି,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    ଫେରନ୍ତୁ cgraph;
}

ଶୂନ୍ୟ ggml_graph_cpy(ସଂରଚନା ggml_cgraph * src, ସଂରଚନା ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    ପାଇଁ (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        ଯଦି (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_ଭର୍ତ୍ତି(&dst->visited_hash_set, src->visited_hash_set.ଚାବିଗୁଡ଼ିକ[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    ଯଦି (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*ଆକାର(ସଂରଚନା ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*ଆକାର(ସଂରଚନା ggml_tensor *));
    }
    ଯଦି (src->grads) {
        GGML_ASSERT(dst->grads     != ଖାଲି);
        GGML_ASSERT(dst->grad_accs != ଖାଲି);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src->n_nodes; ++i) {
            ଅଚଳ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ଅଚଳ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

ସଂରଚନା ggml_cgraph * ggml_graph_dup(ସଂରଚନା ggml_context * ctx, ସଂରଚନା ggml_cgraph * cgraph, bool force_grads) {
    ସଂରଚନା ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_set_zero(ସଂରଚନା ggml_tensor * tensor) {
    ଯଦି (ggml_is_empty(tensor)) {
        ଫେରନ୍ତୁ tensor;
    }
    ଯଦି (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } ଅନ୍ୟଥା {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    ଫେରନ୍ତୁ tensor;
}

ଶୂନ୍ୟ ggml_graph_reset(ସଂରଚନା ggml_cgraph * cgraph) {
    ଯଦି (!cgraph) {
        ଫେରନ୍ତୁ;
    }
    GGML_ASSERT(cgraph->grads != ଖାଲି);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି     = cgraph->nodes[i];
        ସଂରଚନା ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, ଗଣ୍ଠି);

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(ଗଣ୍ଠି->src[2]);
            ggml_set_zero(ଗଣ୍ଠି->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        ଯଦି (grad_acc) {
            ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ଅଚଳ ଦଶମିକ onef = 1.0f;
                ଯଦି (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, ଆକାର(ଦଶମିକ));
                } ଅନ୍ୟଥା {
                    GGML_ASSERT(grad_acc->data);
                    *((ଦଶମିକ *) grad_acc->data) = onef;
                }
            } ଅନ୍ୟଥା {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

ଶୂନ୍ୟ ggml_graph_clear(ସଂରଚନା ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_graph_size(ସଂରଚନା ggml_cgraph * cgraph) {
    ଫେରନ୍ତୁ cgraph->size;
}

ସଂରଚନା ggml_tensor * ggml_graph_ଗଣ୍ଠି(ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ଯଦି (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ଫେରନ୍ତୁ cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ଫେରନ୍ତୁ cgraph->nodes[i];
}

ସଂରଚନା ggml_tensor ** ggml_graph_nodes(ସଂରଚନା ggml_cgraph * cgraph) {
    ଫେରନ୍ତୁ cgraph->nodes;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_graph_n_nodes(ସଂରଚନା ggml_cgraph * cgraph) {
    ଫେରନ୍ତୁ cgraph->n_nodes;
}

ଶୂନ୍ୟ ggml_graph_add_ଗଣ୍ଠି(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ସଂରଚନା ggml_tensor * ggml_graph_get_tensor(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ଅକ୍ଷର * name) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * leaf = cgraph->leafs[i];

        ଯଦି (strcmp(leaf->name, name) == 0) {
            ଫେରନ୍ତୁ leaf;
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        ଯଦି (strcmp(ଗଣ୍ଠି->name, name) == 0) {
            ଫେରନ୍ତୁ ଗଣ୍ଠି;
        }
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

ସଂରଚନା ggml_tensor * ggml_graph_get_grad(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    ଅଚଳ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : ଖାଲି;
}

ସଂରଚନା ggml_tensor * ggml_graph_get_grad_acc(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    ଅଚଳ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ଖାଲି;
}

ଶୂନ୍ୟ ggml_graph_print(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1], ଗଣ୍ଠି->ne[2],
                ggml_op_name(ଗଣ୍ଠି->op), (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, ଗଣ୍ଠି) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1],
                ggml_op_name(ଗଣ୍ଠି->op),
                ggml_get_name(ଗଣ୍ଠି));
    }

    GGML_LOG_INFO("========================================\n");
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_ଗଣ୍ଠି_list_find_tensor(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                idxs,
                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଗଣନା_2,
                                      ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଗଣନା_2; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx = idxs[i];

        ଯଦି (ଗଣ୍ଠି_idx >= cgraph->n_nodes) {
            ଫେରନ୍ତୁ -1;
        }
        ଯଦି (cgraph->nodes[ଗଣ୍ଠି_idx] == tensor) {
            ଫେରନ୍ତୁ i;
        }
    }
    ଫେରନ୍ତୁ -1;
}

ସ୍ଥିର bool ggml_is_constant(ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor->buffer != ଖାଲି && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
                                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                ଗଣ୍ଠି_idxs,
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଗଣନା_2,
                                ଅଚଳ ଗଣନା ggml_op *       ops,
                                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                outputs,
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଅଙ୍କ_outputs) {
    GGML_ASSERT(outputs && ଅଙ୍କ_outputs > 0);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଗଣନା_2; ++i) {
        ଯଦି (ଗଣ୍ଠି_idxs[i] >= cgraph->n_nodes) {
            ଫେରନ୍ତୁ false;
        }

        ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idxs[i]];

        ଯଦି (ଗଣ୍ଠି->op != ops[i]) {
            ଫେରନ୍ତୁ false;
        }

        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଫେରନ୍ତୁ false;
        }

        ଯଦି (ggml_ଗଣ୍ଠି_list_find_tensor(cgraph, outputs, ଅଙ୍କ_outputs, ଗଣ୍ଠି) != -1) {
            ଜାରି;
        }

        ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ଫେରନ୍ତୁ false;
        }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା subgraph_uses = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = i + 1; j < ଗଣନା_2; ++j) {
            ଅଚଳ ସଂରଚନା ggml_tensor * other_ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idxs[j]];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ଯଦି (other_ଗଣ୍ଠି->src[src_idx] == ଗଣ୍ଠି) {
                    subgraph_uses++;
                }
            }
        }

        ଯଦି (subgraph_uses != ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, ଗଣ୍ଠି_idxs[i])) {
            ଫେରନ୍ତୁ false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ସଂରଚନା ggml_tensor * view_src = ଗଣ୍ଠି->view_src;
        ଯେତେବେଳେ (view_src) {
            ଯଦି (ggml_ଗଣ୍ଠି_list_find_tensor(cgraph, ଗଣ୍ଠି_idxs, ଗଣନା_2, view_src) == -1 && !ggml_is_constant(view_src)) {
                ଫେରନ୍ତୁ false;
            }
            view_src = view_src->view_src;
        }
    }

    ଫେରନ୍ତୁ true;
}

// check if node is part of the graph
ସ୍ଥିର bool ggml_graph_find(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    ଯଦି (cgraph == ଖାଲି) {
        ଫେରନ୍ତୁ true;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ଯଦି (cgraph->nodes[i] == ଗଣ୍ଠି) {
            ଫେରନ୍ତୁ true;
        }
    }

    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ସଂରଚନା ggml_tensor * ggml_graph_get_parent(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * parent = cgraph->nodes[i];
        ସଂରଚନା ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        ଯଦି (grad == ଗଣ୍ଠି) {
            ଫେରନ୍ତୁ parent;
        }
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_graph_dump_dot_ଗଣ୍ଠି_edge(FILE * fp, ଅଚଳ ସଂରଚନା ggml_cgraph * gb, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ସଂରଚନା ggml_tensor * parent, ଅଚଳ ଅକ୍ଷର * label)  {
    ସଂରଚନା ggml_tensor * gparent = ggml_graph_get_parent(gb, ଗଣ୍ଠି);
    ସଂରଚନା ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (ଶୂନ୍ୟ *) gparent0 : (ଶୂନ୍ୟ *) parent,
            gparent ? (ଶୂନ୍ୟ *) gparent : (ଶୂନ୍ୟ *) ଗଣ୍ଠି,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_graph_dump_dot_leaf_edge(FILE * fp, ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ସଂରଚନା ggml_tensor * parent, ଅଚଳ ଅକ୍ଷର * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (ଶୂନ୍ୟ *) parent,
            (ଶୂନ୍ୟ *) ଗଣ୍ଠି,
            label);
}

ଶୂନ୍ୟ ggml_graph_dump_dot(ଅଚଳ ସଂରଚନା ggml_cgraph * gb, ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ଅକ୍ଷର * filename) {
    ଅକ୍ଷର color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gb->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = gb->nodes[i];
        ସଂରଚନା ggml_tensor * grad = ggml_graph_get_grad(gb, ଗଣ୍ଠି);

        ଯଦି (ggml_graph_get_parent(gb, ଗଣ୍ଠି) != ଖାଲି) {
            ଜାରି;
        }

        ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, ଆକାର(color), "yellow");
        } ଅନ୍ୟଥା ଯଦି (grad) {
            ଯଦି (ggml_graph_find(cgraph, ଗଣ୍ଠି)) {
                snprintf(color, ଆକାର(color), "green");
            } ଅନ୍ୟଥା {
                snprintf(color, ଆକାର(color), "lightblue");
            }
        } ଅନ୍ୟଥା {
            snprintf(color, ଆକାର(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (ଶୂନ୍ୟ *) ଗଣ୍ଠି, color);

        ଯଦି (strlen(ଗଣ୍ଠି->name) > 0) {
            fprintf(fp, "%s (%s)|", ଗଣ୍ଠି->name, ggml_type_name(ଗଣ୍ଠି->type));
        } ଅନ୍ୟଥା {
            fprintf(fp, "(%s)|", ggml_type_name(ଗଣ୍ଠି->type));
        }

        ଯଦି (ggml_is_matrix(ଗଣ୍ଠି)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1], ggml_op_ପ୍ରତୀକ(ଗଣ୍ଠି->op));
        } ଅନ୍ୟଥା {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1], ଗଣ୍ଠି->ne[2], ggml_op_ପ୍ରତୀକ(ଗଣ୍ଠି->op));
        }

        ଯଦି (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_ପ୍ରତୀକ(grad->op));
        } ଅନ୍ୟଥା {
            fprintf(fp, "\"; ]\n");
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gb->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = gb->leafs[i];

        snprintf(color, ଆକାର(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (ଶୂନ୍ୟ *) ଗଣ୍ଠି, color);

        ଯଦି (strlen(ଗଣ୍ଠି->name) > 0) {
            fprintf(fp, "%s (%s)|", ଗଣ୍ଠି->name, ggml_type_name(ଗଣ୍ଠି->type));
        } ଅନ୍ୟଥା {
            fprintf(fp, "(%s)|", ggml_type_name(ଗଣ୍ଠି->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1]);
        ଯଦି (ggml_nelements(ଗଣ୍ଠି) < 5 && ଗଣ୍ଠି->data != ଖାଲି) {
            fprintf(fp, " | (");
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ggml_nelements(ଗଣ୍ଠି); j++) {
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
                ଯଦି (j < ggml_nelements(ଗଣ୍ଠି) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gb->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = gb->nodes[i];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଯଦି (ଗଣ୍ଠି->src[j]) {
                ଅକ୍ଷର label[16];
                snprintf(label, ଆକାର(label), "src %d", j);
                ggml_graph_dump_dot_ଗଣ୍ଠି_edge(fp, gb, ଗଣ୍ଠି, ଗଣ୍ଠି->src[j], label);
            }
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gb->n_leafs; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = gb->leafs[i];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଯଦି (ଗଣ୍ଠି->src[j]) {
                ଅକ୍ଷର label[16];
                snprintf(label, ଆକାର(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, ଗଣ୍ଠି, ଗଣ୍ଠି->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

ଶୂନ୍ୟ ggml_set_ନିବେଶ(ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_ନିବେଶ;
}

ଶୂନ୍ୟ ggml_set_output(ସଂରଚନା ggml_tensor * tensor) {
    ପାଇଁ (ସଂରଚନା ggml_tensor * cur = tensor; cur != ଖାଲି; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

ଶୂନ୍ୟ ggml_set_param(ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

ଶୂନ୍ୟ ggml_set_loss(ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

ଶୂନ୍ୟ ggml_quantize_init(ଗଣନା ggml_type type) {
    ggml_critical_section_start();

    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: // nothing
            ଭାଙ୍ଗ;
    }

    ggml_critical_section_end();
}

ଶୂନ୍ୟ ggml_quantize_free(ଶୂନ୍ୟ) {
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

bool ggml_quantize_requires_imatrix(ଗଣନା ggml_type type) {
    ଫେରନ୍ତୁ
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        ଗଣନା ggml_type   type,
           ଅଚଳ ଦଶମିକ * src,
                  ଶୂନ୍ୟ * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ଅଚଳ ଦଶମିକ * imatrix) {
    ଅଚଳ int64_t n = nrows * n_per_row;

    ଯଦି (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ଖାଲି);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ଅଚଳ size_t start_row = start / n_per_row;
    ଅଚଳ size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (ଅକ୍ଷର *) dst + start_row * row_size, nrows, n_per_row, imatrix); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                size_t elemsize = ଆକାର(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                size_t elemsize = ଆକାର(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                size_t elemsize = ଆକାର(ଦଶମିକ);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ପ୍ରତିଜ୍ଞା(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    ଫେରନ୍ତୁ result;
}

////////////////////////////////////////////////////////////////////////////////

ଶୂନ୍ୟ ggml_log_get(ggml_log_callback * log_callback, ଶୂନ୍ୟ ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

ଶୂନ୍ୟ ggml_log_set(ggml_log_callback log_callback, ଶୂନ୍ୟ * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

ଶୂନ୍ୟ ggml_threadpool_params_init(ସଂରଚନା ggml_threadpool_params * p, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

ସଂରଚନା ggml_threadpool_params ggml_threadpool_params_default(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ସଂରଚନା ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    ଫେରନ୍ତୁ p;
}

bool ggml_threadpool_params_match(ଅଚଳ ସଂରଚନା ggml_threadpool_params * p0, ଅଚଳ ସଂରଚନା ggml_threadpool_params * p1) {
    ଯଦି (p0->n_threads  != p1->n_threads  ) ଫେରନ୍ତୁ false;
    ଯଦି (p0->prio       != p1->prio       ) ଫେରନ୍ତୁ false;
    ଯଦି (p0->poll       != p1->poll       ) ଫେରନ୍ତୁ false;
    ଯଦି (p0->strict_cpu != p1->strict_cpu ) ଫେରନ୍ତୁ false;
    ଫେରନ୍ତୁ memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
