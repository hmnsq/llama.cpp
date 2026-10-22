#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#অন্তর্ভাৱ "ggml-version.h"
#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-threading.h"
#অন্তর্ভাৱ "ggml-cpu.h"
#অন্তর্ভাৱ "ggml.h"

// FIXME: required here for quantization functions
#অন্তর্ভাৱ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#অন্তর্ভাৱ <hbwmalloc.h>
#endif

#যদি defined(_MSC_VER) || defined(__MINGW32__)
#অন্তর্ভাৱ <malloc.h> // using malloc.h with MSC/MINGW
#নোচেদ্যদি !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#অন্তর্ভাৱ <alloca.h>
#endif

#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <errno.h>
#অন্তর্ভাৱ <time.h>
#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <inttypes.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <float.h>
#অন্তর্ভাৱ <limits.h>
#অন্তর্ভাৱ <stdarg.h>
#অন্তর্ভাৱ <signal.h>
#যদি defined(__gnu_linux__)
#অন্তর্ভাৱ <syscall.h>
#endif

#যদি defined(__APPLE__)
#অন্তর্ভাৱ <unistd.h>
#অন্তর্ভাৱ <mach/mach.h>
#অন্তর্ভাৱ <TargetConditionals.h>
#endif

#যদি defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#অন্তর্ভাৱ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(শূন্য) {
#ifdef _MSC_VER
#যদি defined(_WIN32)
    স্থির চঞ্চল LONG counter = 1;
    ঘূৰাই (uint64_t) InterlockedIncrement(&counter) - 1;
#অন্যথা
    স্থির চঞ্চল long long counter = 1;
    ঘূৰাই (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#অন্যথা
    স্থির uint64_t counter = 1;
    ঘূৰাই __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#যদি defined(__AVX512BF16__)
#যদি defined(_MSC_VER)
#define m512i(p) p
#অন্যথা
#অন্তর্ভাৱ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#যদি defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#অন্তর্ভাৱ <unistd.h>
#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <sys/stat.h>
#অন্তর্ভাৱ <sys/wait.h>
#যদি defined(__linux__)
#অন্তর্ভাৱ <sys/prctl.h>
#endif

#যদি defined(__ANDROID__)
#অন্তর্ভাৱ <unwind.h>
#অন্তর্ভাৱ <dlfcn.h>
#অন্তর্ভাৱ <stdio.h>

সংরচনা backtrace_state {
    শূন্য ** current;
    শূন্য ** end;
};

স্থির _Unwind_Reason_Code unwind_callback(সংরচনা _Unwind_Context* context, শূন্য* arg) {
    সংরচনা backtrace_state * state = (সংরচনা backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    যদি (pc) {
        যদি (state->current == state->end) {
            ঘূৰাই URC_END_OF_স্তূপ;
        } অন্যথা {
            *state->current++ = (শূন্য*)pc;
        }
    }
    ঘূৰাই _URC_NO_REASON;
}

স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    ধ্রুৱ পূর্ণাঙ্ক max = 100;
    শূন্য* buffer[max];

    সংরচনা backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    পূর্ণাঙ্ক গুনতি = state.current - buffer;

    বাবে (পূর্ণাঙ্ক idx = 0; idx < গুনতি; ++idx) {
        ধ্রুৱ শূন্য * addr = buffer[idx];
        ধ্রুৱ ৱর্ণ * প্ৰতীক = "";

        Dl_info info;
        যদি (dladdr(addr, &info) && info.dli_sname) {
            প্ৰতীক = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, প্ৰতীক);
    }
}
#নোচেদ্যদি defined(__linux__) && defined(__GLIBC__)
#অন্তর্ভাৱ <execinfo.h>
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    শূন্য * trace[100];
    পূর্ণাঙ্ক nptrs = backtrace(trace, পরিমাণ(trace)/পরিমাণ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#নোচেদ্যদি defined(__APPLE__)
#অন্তর্ভাৱ <execinfo.h>
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    শূন্য * trace[100];
    পূর্ণাঙ্ক nptrs = backtrace(trace, পরিমাণ(trace)/পরিমাণ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#অন্যথা
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    // platform not supported
}
#endif

শূন্য ggml_print_backtrace(শূন্য) {
    ধ্রুৱ ৱর্ণ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    যদি (GGML_NO_BACKTRACE) {
        ঘূৰাই;
    }
#যদি defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ধ্রুৱ ৱর্ণ * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    যদি (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ঘূৰাই;
    }
#endif
#যদি defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    ৱর্ণ * সাৰি = রিক্ত;
    ssize_t length = 0;
    যেতিয়ালৈ ((length = getline(&সাৰি, &size, f)) > 0) {
        যদি (!strncmp(সাৰি, "TracerPid:", পরিমাণ("TracerPid:") - 1) &&
            (length != পরিমাণ("TracerPid:\t0\n") - 1 || সাৰি[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            মুক্ত(সাৰি);
            fclose(f);
            ঘূৰাই;
        }
    }
    মুক্ত(সাৰি);
    fclose(f);
    পূর্ণাঙ্ক lock[2] = { -1, -1 };
    (শূন্য) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ধ্রুৱ পূর্ণাঙ্ক parent_pid = getpid();
    ধ্রুৱ পূর্ণাঙ্ক child_pid = fork();
    যদি (child_pid < 0) { // error
#যদি defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        ঘূৰাই;
    } অন্যথা যদি (child_pid == 0) { // child
        ৱর্ণ attach[32];
        snprintf(attach, পরিমাণ(attach), "attach %d", parent_pid);
#যদি defined(__linux__)
        close(lock[1]);
        (শূন্য) !পঠ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (ৱর্ণ *) রিক্ত);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[পরিমাণ("attach ") - 1],
            (ৱর্ণ *) রিক্ত);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } অন্যথা { // parent
#যদি defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, রিক্ত, 0);
    }
}
#অন্যথা
শূন্য ggml_print_backtrace(শূন্য) {
    // platform not supported
}
#endif

স্থির ggml_abort_callback_t g_abort_callback = রিক্ত;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ঘূৰাই ret_val;
}

শূন্য ggml_abort(ধ্রুৱ ৱর্ণ * file, পূর্ণাঙ্ক সাৰি, ধ্রুৱ ৱর্ণ * fmt, ...) {
    fflush(stdout);

    ৱর্ণ message[2048];
    পূর্ণাঙ্ক offset = snprintf(message, পরিমাণ(message), "%s:%d: ", file, সাৰি);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, পরিমাণ(message) - offset, fmt, args);
    va_end(args);

    যদি (g_abort_callback) {
        g_abort_callback(message);
    } অন্যথা {
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

সংরচনা ggml_logger_state {
    ggml_log_callback log_callback;
    শূন্য * log_callback_user_data;
};
স্থির সংরচনা ggml_logger_state g_logger_state = {ggml_log_callback_default, রিক্ত};

স্থির শূন্য ggml_log_internal_v(গণনা ggml_log_level level, ধ্রুৱ ৱর্ণ * format, va_list args) {
    যদি (format == রিক্ত) {
        ঘূৰাই;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    ৱর্ণ buffer[128];
    পূর্ণাঙ্ক len = vsnprintf(buffer, 128, format, args);
    যদি (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } অন্যথা {
        ৱর্ণ * buffer2 = (ৱর্ণ *) calloc(len + 1, পরিমাণ(ৱর্ণ));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        মুক্ত(buffer2);
    }
    va_end(args_copy);
}

শূন্য ggml_log_internal(গণনা ggml_log_level level, ধ্রুৱ ৱর্ণ * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

শূন্য ggml_log_callback_default(গণনা ggml_log_level level, ধ্রুৱ ৱর্ণ * text, শূন্য * user_data) {
    (শূন্য) level;
    (শূন্য) user_data;
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
#যদি defined(__has_feature)
#যদি __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#অন্যথা
#যদি defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

শূন্য * ggml_aligned_malloc(size_t size) {
#যদি defined(__s390x__)
    ধ্রুৱ পূর্ণাঙ্ক alignment = 256;
#অন্যথা
    ধ্রুৱ পূর্ণাঙ্ক alignment = 64;
#endif

#যদি defined(_MSC_VER) || defined(__MINGW32__)
    ঘূৰাই _aligned_malloc(size, alignment);
#অন্যথা
    যদি (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ঘূৰাই রিক্ত;
    }
    শূন্য * aligned_memory = রিক্ত;
  #ifdef GGML_USE_CPU_HBM
    পূর্ণাঙ্ক result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #নোচেদ্যদি TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    পূর্ণাঙ্ক result = EFAULT;
    ৱিকল্প (alloc_status) {
        পক্ষ KERN_SUCCESS:
            result = 0;
            ভঙ্গ;
        পক্ষ KERN_INVALID_ADDRESS:
            result = EINVAL;
            ভঙ্গ;
        পক্ষ KERN_NO_SPACE:
            result = ENOMEM;
            ভঙ্গ;
        স্ৱভাৱ:
            result = EFAULT;
            ভঙ্গ;
    }
  #অন্যথা
    পূর্ণাঙ্ক result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    যদি (result != 0) {
        // Handle allocation failure
        ধ্রুৱ ৱর্ণ *error_desc = "unknown allocation error";
        ৱিকল্প (result) {
            পক্ষ EINVAL:
                error_desc = "invalid alignment value";
                ভঙ্গ;
            পক্ষ ENOMEM:
                error_desc = "insufficient memory";
                ভঙ্গ;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        ঘূৰাই রিক্ত;
    }
    ঘূৰাই aligned_memory;
#endif
}

শূন্য ggml_aligned_free(শূন্য * ptr, size_t size) {
    GGML_UNUSED(size);
#যদি defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#নোচেদ্যদি GGML_USE_CPU_HBM
    যদি (ptr != রিক্ত) {
        hbw_free(ptr);
    }
#নোচেদ্যদি TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    যদি (ptr != রিক্ত) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#অন্যথা
    মুক্ত(ptr);
#endif
}


অন্তঃস্থ স্থির শূন্য * ggml_malloc(size_t size) {
    যদি (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ঘূৰাই রিক্ত;
    }
    শূন্য * result = malloc(size);
    যদি (result == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ঘূৰাই result;
}

// calloc
অন্তঃস্থ স্থির শূন্য * ggml_calloc(size_t অঙ্ক, size_t size) {
    যদি (অঙ্ক == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ঘূৰাই রিক্ত;
    }
    শূন্য * result = calloc(অঙ্ক, size);
    যদি (result == রিক্ত) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ঘূৰাই result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(অঙ্ক, size) ggml_calloc(অঙ্ক, size)

#define GGML_FREE(ptr) মুক্ত(ptr)

ধ্রুৱ ৱর্ণ * ggml_status_to_string(গণনা ggml_status status) {
    ৱিকল্প (status) {
        পক্ষ GGML_STATUS_ALLOC_FAILED: ঘূৰাই "GGML status: error (failed to allocate memory)";
        পক্ষ GGML_STATUS_FAILED:       ঘূৰাই "GGML status: error (operation failed)";
        পক্ষ GGML_STATUS_SUCCESS:      ঘূৰাই "GGML status: success";
        পক্ষ GGML_STATUS_ABORTED:      ঘূৰাই "GGML status: warning (operation aborted)";
    }

    ঘূৰাই "GGML status: unknown";
}

ভিন্ন ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    ঘূৰাই GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(ভিন্ন x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    ঘূৰাই GGML_FP32_TO_FP16(x);
}

ভিন্ন ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    ঘূৰাই GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(ভিন্ন x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    ঘূৰাই GGML_FP32_TO_BF16(x);
}

শূন্য ggml_fp16_to_fp32_row(ধ্রুৱ ggml_fp16_t * x, ভিন্ন * y, int64_t n) {
    বাবে (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

শূন্য ggml_fp32_to_fp16_row(ধ্রুৱ ভিন্ন * x, ggml_fp16_t * y, int64_t n) {
    পূর্ণাঙ্ক i = 0;
    বাবে (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

শূন্য ggml_bf16_to_fp32_row(ধ্রুৱ ggml_bf16_t * x, ভিন্ন * y, int64_t n) {
    পূর্ণাঙ্ক i = 0;
    বাবে (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

শূন্য ggml_fp32_to_bf16_row_ref(ধ্রুৱ ভিন্ন * x, ggml_bf16_t * y, int64_t n) {
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

শূন্য ggml_fp32_to_bf16_row(ধ্রুৱ ভিন্ন * x, ggml_bf16_t * y, int64_t n) {
  পূর্ণাঙ্ক i = 0;
#যদি defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  বাবে (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    বাবে (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ঘূৰাই memcmp(guid_a, guid_b, পরিমাণ(ggml_guid)) == 0;
}

ধ্রুৱ ৱর্ণ * ggml_version(শূন্য) {
    ঘূৰাই GGML_VERSION;
}

ধ্রুৱ ৱর্ণ * ggml_commit(শূন্য) {
    ঘূৰাই GGML_COMMIT;
}

//
// timing
//

#যদি defined(_MSC_VER) || defined(__MINGW32__)
স্থির int64_t timer_freq, timer_start;
স্থির BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    ঘূৰাই TRUE;
}
শূন্য ggml_time_init(শূন্য) {
    স্থির INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, রিক্ত, রিক্ত);
}
int64_t ggml_time_ms(শূন্য) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ঘূৰাই ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(শূন্য) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    ঘূৰাই ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#অন্যথা
শূন্য ggml_time_init(শূন্য) {}
int64_t ggml_time_ms(শূন্য) {
    সংরচনা timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ঘূৰাই (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(শূন্য) {
    সংরচনা timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ঘূৰাই (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(শূন্য) {
    ঘূৰাই clock();
}

int64_t ggml_cycles_per_ms(শূন্য) {
    ঘূৰাই CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
স্থির wchar_t * ggml_mbstowcs(ধ্রুৱ ৱর্ণ * mbs) {
    পূর্ণাঙ্ক wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, রিক্ত, 0);
    যদি (!wlen) {
        errno = EINVAL;
        ঘূৰাই রিক্ত;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * পরিমাণ(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    যদি (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ঘূৰাই রিক্ত;
    }

    ঘূৰাই wbuf;
}
#endif

FILE * ggml_fopen(ধ্রুৱ ৱর্ণ * fname, ধ্রুৱ ৱর্ণ * mode) {
#ifdef _WIN32
    FILE * file = রিক্ত;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    যদি (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        যদি (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    ঘূৰাই file;
#অন্যথা
    ঘূৰাই fopen(fname, mode);
#endif

}

স্থির ধ্রুৱ সংরচনা ggml_type_traits type_traits[GGML_TYPE_গুনতি] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = পরিমাণ(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = পরিমাণ(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = পরিমাণ(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = পরিমাণ(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = পরিমাণ(দ্ৱিগুণ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = পরিমাণ(ভিন্ন),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = পরিমাণ(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = পরিমাণ(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = পরিমাণ(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = পরিমাণ(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = পরিমাণ(block_q4_1),
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
        .type_size                = পরিমাণ(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = পরিমাণ(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = পরিমাণ(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = পরিমাণ(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = পরিমাণ(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = পরিমাণ(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = রিক্ত,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = রিক্ত,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = রিক্ত,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = রিক্ত,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = পরিমাণ(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = পরিমাণ(ggml_bf16_t),
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
        .type_size                = পরিমাণ(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = পরিমাণ(block_tq2_0),
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

ধ্রুৱ সংরচনা ggml_type_traits * ggml_get_type_traits(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই &type_traits[type];
}

//
// ggml object
//

সংরচনা ggml_object {
    size_t offs;
    size_t size;

    সংরচনা ggml_object * next;

    গণনা ggml_object_type type;

    ৱর্ণ padding[4];
};

স্থির ধ্রুৱ size_t GGML_OBJECT_SIZE = পরিমাণ(সংরচনা ggml_object);

//
// ggml context
//

সংরচনা ggml_context {
    size_t mem_size;
    শূন্য * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    পূর্ণাঙ্ক    n_objects;

    সংরচনা ggml_object * objects_begin;
    সংরচনা ggml_object * objects_end;
};

//
// data types
//

স্থির ধ্রুৱ ৱর্ণ * GGML_OP_NAME[GGML_OP_গুনতি] = {
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

static_assert(GGML_OP_গুনতি == 102, "GGML_OP_COUNT != 102");

স্থির ধ্রুৱ ৱর্ণ * GGML_OP_প্ৰতীক[GGML_OP_গুনতি] = {
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

static_assert(GGML_OP_গুনতি == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_গুনতি == 2, "GGML_OP_POOL_COUNT != 2");

স্থির ধ্রুৱ ৱর্ণ * GGML_UNARY_OP_NAME[GGML_UNARY_OP_গুনতি] = {
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

static_assert(GGML_UNARY_OP_গুনতি == 22, "GGML_UNARY_OP_COUNT != 22");

স্থির ধ্রুৱ ৱর্ণ * GGML_GLU_OP_NAME[GGML_GLU_OP_গুনতি] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_গুনতি == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(পরিমাণ(সংরচনা ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(পরিমাণ(সংরচনা ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_print_object(ধ্রুৱ সংরচনা ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ধ্রুৱ শূন্য *) obj->next);
}

শূন্য ggml_print_objects(ধ্রুৱ সংরচনা ggml_context * ctx) {
    সংরচনা ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ধ্রুৱ শূন্য *) ctx);

    যেতিয়ালৈ (obj != রিক্ত) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] <= 0) {
            ঘূৰাই 0;
        }
    }

    size_t nbytes;
    ধ্রুৱ size_t blck_size = ggml_blck_size(tensor->type);
    যদি (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    অন্যথা {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        বাবে (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ঘূৰাই nbytes;
}

size_t ggml_nbytes_pad(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই type_traits[type].blck_size;
}

size_t ggml_type_size(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই type_traits[type].type_size;
}

size_t ggml_row_size(গণনা ggml_type type, int64_t ne) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    প্রতিজ্ঞা(ne % ggml_blck_size(type) == 0);
    ঘূৰাই ggml_type_size(type)*ne/ggml_blck_size(type);
}

দ্ৱিগুণ ggml_type_sizef(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই ((দ্ৱিগুণ)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ধ্রুৱ ৱর্ণ * ggml_type_name(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই type_traits[type].type_name;
}

bool ggml_is_quantized(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ঘূৰাই type_traits[type].is_quantized;
}

ধ্রুৱ ৱর্ণ * ggml_op_name(গণনা ggml_op op) {
    ঘূৰাই GGML_OP_NAME[op];
}

ধ্রুৱ ৱর্ণ * ggml_op_প্ৰতীক(গণনা ggml_op op) {
    ঘূৰাই GGML_OP_প্ৰতীক[op];
}

ধ্রুৱ ৱর্ণ * ggml_unary_op_name(গণনা ggml_unary_op op) {
    ঘূৰাই GGML_UNARY_OP_NAME[op];
}

ধ্রুৱ ৱর্ণ * ggml_glu_op_name(গণনা ggml_glu_op op) {
    ঘূৰাই GGML_GLU_OP_NAME[op];
}

ধ্রুৱ ৱর্ণ * ggml_op_desc(ধ্রুৱ সংরচনা ggml_tensor * t) {
    যদি (t->op == GGML_OP_UNARY) {
        গণনা ggml_unary_op uop = ggml_get_unary_op(t);
        ঘূৰাই ggml_unary_op_name(uop);
    }
    যদি (t->op == GGML_OP_GLU) {
        গণনা ggml_glu_op gop = ggml_get_glu_op(t);
        ঘূৰাই ggml_glu_op_name(gop);
    }
    ঘূৰাই ggml_op_name(t->op);
}

size_t ggml_element_size(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই tensor->ne[3] == 1;
}

পূর্ণাঙ্ক ggml_n_dims(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        যদি (tensor->ne[i] > 1) {
            ঘূৰাই i + 1;
        }
    }
    ঘূৰাই 1;
}

গণনা ggml_type ggml_ftype_to_ggml_type(গণনা ggml_ftype ftype) {
    গণনা ggml_type wtype = GGML_TYPE_গুনতি;

    ৱিকল্প (ftype) {
        পক্ষ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ভঙ্গ;
        পক্ষ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_গুনতি; ভঙ্গ;
        পক্ষ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_গুনতি; ভঙ্গ;
    }

    GGML_ASSERT(wtype != GGML_TYPE_গুনতি);

    ঘূৰাই wtype;
}

size_t ggml_tensor_overhead(শূন্য) {
    ঘূৰাই GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই tensor->nb[0] > tensor->nb[1];
}

স্থির bool ggml_is_contiguous_m_n(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক m, পূর্ণাঙ্ক n) {
    size_t next_nb = ggml_type_size(tensor->type);
    যদি (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        ঘূৰাই false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    বাবে (পূর্ণাঙ্ক i = 1; i < n; i++) {
        যদি (i > m) {
            যদি (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                ঘূৰাই false;
            }
            next_nb *= tensor->ne[i];
        } অন্যথা {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ঘূৰাই true;
}

bool ggml_is_contiguous(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

স্থির অন্তঃস্থ bool ggml_is_padded_1d(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ঘূৰাই true;
        }
    }
    ঘূৰাই false;
}

bool ggml_are_same_shape(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ধ্রুৱ সংরচনা ggml_tensor * t) {
    ঘূৰাই ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

স্থির অন্তঃস্থ bool ggml_can_repeat_rows(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

সংরচনা ggml_context * ggml_init(সংরচনা ggml_init_params params) {
    স্থির bool is_first_call = true;

    ggml_critical_section_start();

    যদি (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    সংরচনা ggml_context * ctx = GGML_MALLOC(পরিমাণ(সংরচনা ggml_context));

    // allow to call ggml_init with 0 size
    যদি (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ধ্রুৱ size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (সংরচনা ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ রিক্ত,
        /*.objects_end        =*/ রিক্ত,
    };

    GGML_ASSERT(ctx->mem_buffer != রিক্ত);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ঘূৰাই ctx;
}

শূন্য ggml_reset(সংরচনা ggml_context * ctx) {
    যদি (ctx == রিক্ত) {
        ঘূৰাই;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = রিক্ত;
    ctx->objects_end   = রিক্ত;
}

শূন্য ggml_free(সংরচনা ggml_context * ctx) {
    যদি (ctx == রিক্ত) {
        ঘূৰাই;
    }

    যদি (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ধ্রুৱ সংরচনা ggml_context * ctx) {
    ঘূৰাই ctx->objects_end == রিক্ত ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(সংরচনা ggml_context * ctx) {
    ঘূৰাই ctx->no_alloc;
}

শূন্য ggml_set_no_alloc(সংরচনা ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

শূন্য * ggml_get_mem_buffer(ধ্রুৱ সংরচনা ggml_context * ctx) {
    ঘূৰাই ctx->mem_buffer;
}

size_t ggml_get_mem_size(ধ্রুৱ সংরচনা ggml_context * ctx) {
    ঘূৰাই ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ধ্রুৱ সংরচনা ggml_context * ctx) {
    size_t max_size = 0;

    বাবে (সংরচনা ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != রিক্ত; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    ঘূৰাই max_size;
}

////////////////////////////////////////////////////////////////////////////////

স্থির সংরচনা ggml_object * ggml_new_object(সংরচনা ggml_context * ctx, গণনা ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    সংরচনা ggml_object * obj_cur = ctx->objects_end;

    ধ্রুৱ size_t cur_offs = obj_cur == রিক্ত ? 0 : obj_cur->offs;
    ধ্রুৱ size_t cur_size = obj_cur == রিক্ত ? 0 : obj_cur->size;
    ধ্রুৱ size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    ৱর্ণ * ধ্রুৱ mem_buffer = ctx->mem_buffer;
    সংরচনা ggml_object * ধ্রুৱ obj_new = (সংরচনা ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    যদি (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        ঘূৰাই রিক্ত;
    }
    যদি (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        ঘূৰাই রিক্ত;
    }

    যদি (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        ঘূৰাই রিক্ত;
    }

    *obj_new = (সংরচনা ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = রিক্ত,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    যদি (obj_cur != রিক্ত) {
        obj_cur->next = obj_new;
    } অন্যথা {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ঘূৰাই obj_new;
}

স্থির সংরচনা ggml_tensor * ggml_new_tensor_impl(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type      type,
        পূর্ণাঙ্ক                   n_dims,
        ধ্রুৱ int64_t       * ne,
        সংরচনা ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_গুনতি);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    যদি (view_src != রিক্ত && view_src->view_src != রিক্ত) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    বাবে (পূর্ণাঙ্ক i = 1; i < n_dims; i++) {
        যদি (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    বাবে (পূর্ণাঙ্ক i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == রিক্ত || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    শূন্য * data = view_src != রিক্ত ? view_src->data : রিক্ত;
    যদি (data != রিক্ত) {
        data = (ৱর্ণ *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    যদি (view_src == রিক্ত && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    সংরচনা ggml_object * ধ্রুৱ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    সংরচনা ggml_tensor * ধ্রুৱ result = (সংরচনা ggml_tensor *)((ৱর্ণ *)ctx->mem_buffer + obj_new->offs);

    *result = (সংরচনা ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ রিক্ত,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { রিক্ত },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (শূন্য *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ রিক্ত,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    বাবে (পূর্ণাঙ্ক i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    বাবে (পূর্ণাঙ্ক i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_new_tensor(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type      type,
        পূর্ণাঙ্ক                   n_dims,
        ধ্রুৱ int64_t       * ne) {
    ঘূৰাই ggml_new_tensor_impl(ctx, type, n_dims, ne, রিক্ত, 0);
}

সংরচনা ggml_tensor * ggml_new_tensor_1d(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0) {
    ঘূৰাই ggml_new_tensor(ctx, type, 1, &ne0);
}

সংরচনা ggml_tensor * ggml_new_tensor_2d(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ধ্রুৱ int64_t ne[2] = { ne0, ne1 };
    ঘূৰাই ggml_new_tensor(ctx, type, 2, ne);
}

সংরচনা ggml_tensor * ggml_new_tensor_3d(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ধ্রুৱ int64_t ne[3] = { ne0, ne1, ne2 };
    ঘূৰাই ggml_new_tensor(ctx, type, 3, ne);
}

সংরচনা ggml_tensor * ggml_new_tensor_4d(
        সংরচনা ggml_context * ctx,
        গণনা   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ধ্রুৱ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ঘূৰাই ggml_new_tensor(ctx, type, 4, ne);
}

শূন্য * ggml_new_buffer(সংরচনা ggml_context * ctx, size_t nbytes) {
    সংরচনা ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    ঘূৰাই (uint8_t *)ctx->mem_buffer + obj->offs;
}

সংরচনা ggml_tensor * ggml_dup_tensor(সংরচনা ggml_context * ctx, ধ্রুৱ সংরচনা ggml_tensor * src) {
    ঘূৰাই ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

শূন্য ggml_unravel_index(ধ্রুৱ সংরচনা ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ধ্রুৱ int64_t ne2 = tensor->ne[2];
    ধ্রুৱ int64_t ne1 = tensor->ne[1];
    ধ্রুৱ int64_t ne0 = tensor->ne[0];

    ধ্রুৱ int64_t i3_ = (i/(ne2*ne1*ne0));
    ধ্রুৱ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ধ্রুৱ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ধ্রুৱ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    যদি (i0) {
        * i0 = i0_;
    }
    যদি (i1) {
        * i1 = i1_;
    }
    যদি (i2) {
        * i2 = i2_;
    }
    যদি (i3) {
        * i3 = i3_;
    }
}

শূন্য * ggml_get_data(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই tensor->data;
}

ভিন্ন * ggml_get_data_f32(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    প্রতিজ্ঞা(tensor->type == GGML_TYPE_F32);
    ঘূৰাই (ভিন্ন *)(tensor->data);
}

গণনা ggml_unary_op ggml_get_unary_op(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ঘূৰাই (গণনা ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

গণনা ggml_glu_op ggml_get_glu_op(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ঘূৰাই (গণনা ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ধ্রুৱ ৱর্ণ * ggml_get_name(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই tensor->name;
}

সংরচনা ggml_tensor * ggml_set_name(সংরচনা ggml_tensor * tensor, ধ্রুৱ ৱর্ণ * name) {
    size_t i;
    বাবে (i = 0; i < পরিমাণ(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ঘূৰাই tensor;
}

সংরচনা ggml_tensor * ggml_format_name(সংরচনা ggml_tensor * tensor, ধ্রুৱ ৱর্ণ * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, পরিমাণ(tensor->name), fmt, args);
    va_end(args);
    ঘূৰাই tensor;
}

সংরচনা ggml_tensor * ggml_view_tensor(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * src) {
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_get_first_tensor(ধ্রুৱ সংরচনা ggml_context * ctx) {
    সংরচনা ggml_object * obj = ctx->objects_begin;

    ৱর্ণ * ধ্রুৱ mem_buffer = ctx->mem_buffer;

    যেতিয়ালৈ (obj != রিক্ত) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ঘূৰাই (সংরচনা ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ঘূৰাই রিক্ত;
}

সংরচনা ggml_tensor * ggml_get_next_tensor(ধ্রুৱ সংরচনা ggml_context * ctx, সংরচনা ggml_tensor * tensor) {
    সংরচনা ggml_object * obj = (সংরচনা ggml_object *) ((ৱর্ণ *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    ৱর্ণ * ধ্রুৱ mem_buffer = ctx->mem_buffer;

    যেতিয়ালৈ (obj != রিক্ত) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ঘূৰাই (সংরচনা ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    ঘূৰাই রিক্ত;
}

সংরচনা ggml_tensor * ggml_get_tensor(সংরচনা ggml_context * ctx, ধ্রুৱ ৱর্ণ * name) {
    সংরচনা ggml_object * obj = ctx->objects_begin;

    ৱর্ণ * ধ্রুৱ mem_buffer = ctx->mem_buffer;

    যেতিয়ালৈ (obj != রিক্ত) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            সংরচনা ggml_tensor * cur = (সংরচনা ggml_tensor *)(mem_buffer + obj->offs);
            যদি (strcmp(cur->name, name) == 0) {
                ঘূৰাই cur;
            }
        }

        obj = obj->next;
    }

    ঘূৰাই রিক্ত;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

স্থির সংরচনা ggml_tensor * ggml_dup_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_dup(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_dup_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_dup_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_dup_impl(ctx, a, true);
}

// ggml_add

স্থির সংরচনা ggml_tensor * ggml_add_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_add(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_add_impl(ctx, a, b, false);
}

সংরচনা ggml_tensor * ggml_add_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

স্থির সংরচনা ggml_tensor * ggml_add_cast_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        গণনা   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_add_cast(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        গণনা   ggml_type      type) {
    ঘূৰাই ggml_add_cast_impl(ctx, a, b, type);
}

সংরচনা ggml_tensor * ggml_add_id(
            সংরচনা ggml_context * ctx,
            সংরচনা ggml_tensor  * a,
            সংরচনা ggml_tensor  * b,
            সংরচনা ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    ঘূৰাই result;
}

// ggml_add1

স্থির সংরচনা ggml_tensor * ggml_add1_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_add1(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_add1_impl(ctx, a, b, false);
}

সংরচনা ggml_tensor * ggml_add1_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

স্থির সংরচনা ggml_tensor * ggml_acc_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_acc(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ঘূৰাই ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

সংরচনা ggml_tensor * ggml_acc_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ঘূৰাই ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

স্থির সংরচনা ggml_tensor * ggml_sub_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_sub(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_sub_impl(ctx, a, b, false);
}

সংরচনা ggml_tensor * ggml_sub_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

স্থির সংরচনা ggml_tensor * ggml_mul_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_mul(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_mul_impl(ctx, a, b, false);
}

সংরচনা ggml_tensor * ggml_mul_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

স্থির সংরচনা ggml_tensor * ggml_div_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_div(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_div_impl(ctx, a, b, false);
}

সংরচনা ggml_tensor * ggml_div_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

স্থির সংরচনা ggml_tensor * ggml_sqr_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_sqr(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sqr_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_sqr_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

স্থির সংরচনা ggml_tensor * ggml_sqrt_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_sqrt(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sqrt_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_sqrt_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

স্থির সংরচনা ggml_tensor * ggml_log_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_log(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_log_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_log_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_log_impl(ctx, a, true);
}

সংরচনা ggml_tensor * ggml_expm1(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

সংরচনা ggml_tensor * ggml_expm1_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

সংরচনা ggml_tensor * ggml_softplus(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

সংরচনা ggml_tensor * ggml_softplus_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

স্থির সংরচনা ggml_tensor * ggml_sin_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_sin(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sin_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_sin_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_sin_impl(ctx, a, true);
}

// ggml_cos

স্থির সংরচনা ggml_tensor * ggml_cos_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_cos(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_cos_impl(ctx, a, false);
}

সংরচনা ggml_tensor * ggml_cos_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_cos_impl(ctx, a, true);
}

// ggml_sum

সংরচনা ggml_tensor * ggml_sum(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_sum_rows

সংরচনা ggml_tensor * ggml_sum_rows(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    বাবে (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_cumsum

সংরচনা ggml_tensor * ggml_cumsum(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_mean

সংরচনা ggml_tensor * ggml_mean(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_argmax

সংরচনা ggml_tensor * ggml_argmax(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_count_equal

সংরচনা ggml_tensor * ggml_গুনতি_equal(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_গুনতি_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_repeat

সংরচনা ggml_tensor * ggml_repeat(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_repeat_4d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ধ্রুৱ bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_repeat_back

সংরচনা ggml_tensor * ggml_repeat_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_concat

সংরচনা ggml_tensor * ggml_concat(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    সংরচনা ggml_tensor  * b,
    পূর্ণাঙ্ক                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    বাবে (পূর্ণাঙ্ক d = 0; d < GGML_MAX_DIMS; ++d) {
        যদি (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            অনুৱর্ত;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_abs

সংরচনা ggml_tensor * ggml_abs(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

সংরচনা ggml_tensor * ggml_abs_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

সংরচনা ggml_tensor * ggml_sgn(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

সংরচনা ggml_tensor * ggml_sgn_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

সংরচনা ggml_tensor * ggml_neg(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

সংরচনা ggml_tensor * ggml_neg_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

সংরচনা ggml_tensor * ggml_step(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

সংরচনা ggml_tensor * ggml_step_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

সংরচনা ggml_tensor * ggml_tanh(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

সংরচনা ggml_tensor * ggml_tanh_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

সংরচনা ggml_tensor * ggml_elu(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

সংরচনা ggml_tensor * ggml_elu_inplace(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

সংরচনা ggml_tensor * ggml_relu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

সংরচনা ggml_tensor * ggml_relu_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

সংরচনা ggml_tensor * ggml_leaky_relu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 negative_slope,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, পরিমাণ(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_sigmoid

সংরচনা ggml_tensor * ggml_sigmoid(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

সংরচনা ggml_tensor * ggml_sigmoid_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

সংরচনা ggml_tensor * ggml_gelu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

সংরচনা ggml_tensor * ggml_gelu_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

সংরচনা ggml_tensor * ggml_gelu_erf(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

সংরচনা ggml_tensor * ggml_gelu_erf_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

সংরচনা ggml_tensor * ggml_gelu_quick(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

সংরচনা ggml_tensor * ggml_gelu_quick_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

সংরচনা ggml_tensor * ggml_silu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

সংরচনা ggml_tensor * ggml_silu_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

সংরচনা ggml_tensor * ggml_xielu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন alpha_n,
        ভিন্ন alpha_p,
        ভিন্ন beta,
        ভিন্ন eps) {
    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_silu_back

সংরচনা ggml_tensor * ggml_silu_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml hardswish

সংরচনা ggml_tensor * ggml_hardswish(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

সংরচনা ggml_tensor * ggml_hardsigmoid(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

সংরচনা ggml_tensor * ggml_exp(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

সংরচনা ggml_tensor * ggml_exp_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

স্থির সংরচনা ggml_tensor * ggml_glu_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        গণনা ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    যদি (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; বাবে (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, রিক্ত, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_floor

সংরচনা ggml_tensor * ggml_floor(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

সংরচনা ggml_tensor * ggml_floor_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

সংরচনা ggml_tensor * ggml_ceil(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

সংরচনা ggml_tensor * ggml_ceil_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

সংরচনা ggml_tensor * ggml_round(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

সংরচনা ggml_tensor * ggml_round_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

সংরচনা ggml_tensor * ggml_trunc(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

সংরচনা ggml_tensor * ggml_trunc_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

সংরচনা ggml_tensor * ggml_glu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_glu_op      op,
        bool                  swapped) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, op, swapped);
}

সংরচনা ggml_tensor * ggml_glu_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        গণনা ggml_glu_op      op) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

সংরচনা ggml_tensor * ggml_reglu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_REGLU, false);
}

সংরচনা ggml_tensor * ggml_reglu_swapped(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_REGLU, true);
}

সংরচনা ggml_tensor * ggml_reglu_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

সংরচনা ggml_tensor * ggml_geglu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU, false);
}

সংরচনা ggml_tensor * ggml_geglu_swapped(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU, true);
}

সংরচনা ggml_tensor * ggml_geglu_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

সংরচনা ggml_tensor * ggml_swiglu(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_SWIGLU, false);
}

সংরচনা ggml_tensor * ggml_swiglu_swapped(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_SWIGLU, true);
}

সংরচনা ggml_tensor * ggml_swiglu_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

সংরচনা ggml_tensor * ggml_geglu_erf(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU_ERF, false);
}

সংরচনা ggml_tensor * ggml_geglu_erf_swapped(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU_ERF, true);
}

সংরচনা ggml_tensor * ggml_geglu_erf_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

সংরচনা ggml_tensor * ggml_geglu_quick(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU_QUICK, false);
}

সংরচনা ggml_tensor * ggml_geglu_quick_swapped(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_glu_impl(ctx, a, রিক্ত, GGML_GLU_OP_GEGLU_QUICK, true);
}

সংরচনা ggml_tensor * ggml_geglu_quick_split(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

সংরচনা ggml_tensor * ggml_swiglu_oai(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 alpha,
        ভিন্ন                 limit) {
    সংরচনা ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_swiglu_clamp(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 limit) {
    সংরচনা ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    ঘূৰাই result;
}

// ggml_norm

স্থির সংরচনা ggml_tensor * ggml_norm_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, পরিমাণ(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_norm(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_norm_impl(ctx, a, eps, false);
}

সংরচনা ggml_tensor * ggml_norm_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

স্থির সংরচনা ggml_tensor * ggml_rms_norm_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, পরিমাণ(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_rms_norm(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_rms_norm_impl(ctx, a, eps, false);
}

সংরচনা ggml_tensor * ggml_rms_norm_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

সংরচনা ggml_tensor * ggml_rms_norm_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 eps) {
    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, পরিমাণ(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_group_norm

স্থির সংরচনা ggml_tensor * ggml_group_norm_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_groups,
        ভিন্ন                 eps,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_group_norm(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_groups,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

সংরচনা ggml_tensor * ggml_group_norm_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_groups,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

স্থির সংরচনা ggml_tensor * ggml_l2_norm_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_l2_norm(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_l2_norm_impl(ctx, a, eps, false);
}

সংরচনা ggml_tensor * ggml_l2_norm_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 eps) {
    ঘূৰাই ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        সংরচনা ggml_tensor * a,
        গণনা ggml_prec       prec) {
    ৱিকল্প (a->op) {
        পক্ষ GGML_OP_MUL_MAT:
        পক্ষ GGML_OP_MUL_MAT_ID:
            {
                ধ্রুৱ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            {
                ধ্রুৱ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            ভঙ্গ;
        স্ৱভাৱ:
            ঘূৰাই false;
    };

    ঘূৰাই true;
}

bool ggml_prec_set_src(
        সংরচনা ggml_tensor * a,
        গণনা ggml_prec       prec,
        পূর্ণাঙ্ক                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ৱিকল্প (a->op) {
        পক্ষ GGML_OP_MUL_MAT:
        পক্ষ GGML_OP_MUL_MAT_ID:
            {
                যদি (idx != 1) {
                    ঘূৰাই false;
                }

                ধ্রুৱ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            ভঙ্গ;
        স্ৱভাৱ:
            ঘূৰাই false;
    };

    ঘূৰাই true;
}

// ggml_mul_mat

স্থির অন্তঃস্থ bool ggml_can_mul_mat(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

সংরচনা ggml_tensor * ggml_mul_mat(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ধ্রুৱ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

শূন্য ggml_mul_mat_set_prec(
        সংরচনা ggml_tensor * a,
        গণনা ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ধ্রুৱ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

শূন্য ggml_mul_mat_set_hint(
        সংরচনা ggml_tensor * a,
        গণনা ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ধ্রুৱ int32_t hint_i32 = (int32_t) hint;

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
সংরচনা ggml_tensor * ggml_mul_mat_id(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * as,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ধ্রুৱ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    ঘূৰাই result;
}

// ggml_out_prod

স্থির অন্তঃস্থ bool ggml_can_out_prod(ধ্রুৱ সংরচনা ggml_tensor * t0, ধ্রুৱ সংরচনা ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ঘূৰাই (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

সংরচনা ggml_tensor * ggml_out_prod(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ধ্রুৱ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_scale

স্থির সংরচনা ggml_tensor * ggml_scale_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 s,
        ভিন্ন                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ভিন্ন params[2] = { s, b };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_scale(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 s) {
    ঘূৰাই ggml_scale_impl(ctx, a, s, 0.0, false);
}

সংরচনা ggml_tensor * ggml_scale_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 s) {
    ঘূৰাই ggml_scale_impl(ctx, a, s, 0.0, true);
}

সংরচনা ggml_tensor * ggml_scale_bias(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 s,
        ভিন্ন                 b) {
    ঘূৰাই ggml_scale_impl(ctx, a, s, b, false);
}

সংরচনা ggml_tensor * ggml_scale_bias_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 s,
        ভিন্ন                 b) {
    ঘূৰাই ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

স্থির সংরচনা ggml_tensor * ggml_set_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_set(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

সংরচনা ggml_tensor * ggml_set_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

সংরচনা ggml_tensor * ggml_set_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

সংরচনা ggml_tensor * ggml_set_1d_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

সংরচনা ggml_tensor * ggml_set_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

সংরচনা ggml_tensor * ggml_set_2d_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ঘূৰাই ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

স্থির সংরচনা ggml_tensor * ggml_cpy_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, b);
    যদি (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } অন্যথা {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_cpy(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor * a,
        সংরচনা ggml_tensor * b) {
    ঘূৰাই ggml_cpy_impl(ctx, a, b);
}

সংরচনা ggml_tensor * ggml_cast(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা   ggml_type      type) {
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ঘূৰাই result;
}

// ggml_cont

স্থির সংরচনা ggml_tensor * ggml_cont_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_cont(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor * a) {
    ঘূৰাই ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API সংরচনা ggml_tensor * ggml_cont_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0) {
    ঘূৰাই ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API সংরচনা ggml_tensor * ggml_cont_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ঘূৰাই ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API সংরচনা ggml_tensor * ggml_cont_3d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ঘূৰাই ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

সংরচনা ggml_tensor * ggml_cont_4d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_reshape

সংরচনা ggml_tensor * ggml_reshape(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor * a,
        সংরচনা ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_reshape_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ধ্রুৱ int64_t ne[1] = { ne0 };
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_reshape_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ধ্রুৱ int64_t ne[2] = { ne0, ne1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_reshape_3d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ধ্রুৱ int64_t ne[3] = { ne0, ne1, ne2 };
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_reshape_4d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ধ্রুৱ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    ঘূৰাই result;
}

স্থির সংরচনা ggml_tensor * ggml_view_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_dims,
        ধ্রুৱ int64_t       * ne,
        size_t                offset) {
    সংরচনা ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, পরিমাণ(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_view_1d

সংরচনা ggml_tensor * ggml_view_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    সংরচনা ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    ঘূৰাই result;
}

// ggml_view_2d

সংরচনা ggml_tensor * ggml_view_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ধ্রুৱ int64_t ne[2] = { ne0, ne1 };

    সংরচনা ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    ঘূৰাই result;
}

// ggml_view_3d

সংরচনা ggml_tensor * ggml_view_3d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ধ্রুৱ int64_t ne[3] = { ne0, ne1, ne2 };

    সংরচনা ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    ঘূৰাই result;
}

// ggml_view_4d

সংরচনা ggml_tensor * ggml_view_4d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ধ্রুৱ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    সংরচনা ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    ঘূৰাই result;
}

// ggml_permute

সংরচনা ggml_tensor * ggml_permute(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   axis0,
        পূর্ণাঙ্ক                   axis1,
        পূর্ণাঙ্ক                   axis2,
        পূর্ণাঙ্ক                   axis3) {
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

    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, পরিমাণ(params));

    ঘূৰাই result;
}

// ggml_transpose

সংরচনা ggml_tensor * ggml_transpose(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_get_rows

সংরচনা ggml_tensor * ggml_get_rows(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    গণনা ggml_type type = GGML_TYPE_F32;
    যদি (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_get_rows_back

সংরচনা ggml_tensor * ggml_get_rows_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    সংরচনা ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_set_rows

সংরচনা ggml_tensor * ggml_set_rows(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c) {
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

    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ঘূৰাই result;
}

// ggml_diag

সংরচনা ggml_tensor * ggml_diag(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ধ্রুৱ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_diag_mask_inf

স্থির সংরচনা ggml_tensor * ggml_diag_mask_inf_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_diag_mask_inf(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past) {
    ঘূৰাই ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

সংরচনা ggml_tensor * ggml_diag_mask_inf_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past) {
    ঘূৰাই ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

স্থির সংরচনা ggml_tensor * ggml_diag_mask_zero_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_diag_mask_zero(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past) {
    ঘূৰাই ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

সংরচনা ggml_tensor * ggml_diag_mask_zero_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_past) {
    ঘূৰাই ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

স্থির সংরচনা ggml_tensor * ggml_clamp_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        ভিন্ন                 min,
        ভিন্ন                 max,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ভিন্ন params[] = { min, max };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_clamp(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    ভিন্ন                 min,
    ভিন্ন                 max) {
    ঘূৰাই ggml_clamp_impl(ctx, a, min, max, false);
}

সংরচনা ggml_tensor * ggml_clamp_inplace(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    ভিন্ন                 min,
    ভিন্ন                 max) {
    ঘূৰাই ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

স্থির সংরচনা ggml_tensor * ggml_soft_max_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * mask,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    যদি (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    যদি (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ভিন্ন params[] = { scale, max_bias };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_soft_max(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_soft_max_impl(ctx, a, রিক্ত, 1.0f, 0.0f, false);
}

সংরচনা ggml_tensor * ggml_soft_max_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a) {
    ঘূৰাই ggml_soft_max_impl(ctx, a, রিক্ত, 1.0f, 0.0f, true);
}

সংরচনা ggml_tensor * ggml_soft_max_ext(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * mask,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias) {
    ঘূৰাই ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

সংরচনা ggml_tensor * ggml_soft_max_ext_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * mask,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias) {
    ঘূৰাই ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

শূন্য ggml_soft_max_add_sinks(
        সংরচনা ggml_tensor * a,
        সংরচনা ggml_tensor * sinks) {
    যদি (!sinks) {
        a->src[2] = রিক্ত;
        ঘূৰাই;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == রিক্ত);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

স্থির সংরচনা ggml_tensor * ggml_soft_max_ext_back_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias,
        bool                  inplace) {
    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((ভিন্ন *) result->op_params + 0, &scale,    পরিমাণ(ভিন্ন));
    memcpy((ভিন্ন *) result->op_params + 1, &max_bias, পরিমাণ(ভিন্ন));

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_soft_max_ext_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias) {
    ঘূৰাই ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

সংরচনা ggml_tensor * ggml_soft_max_ext_back_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias) {
    ঘূৰাই ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

স্থির সংরচনা ggml_tensor * ggml_rope_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   sections[GGML_MROPE_SECTIONS],
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    যদি (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } অন্যথা {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    যদি (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    পরিমাণ(ভিন্ন));
    memcpy(params +  6, &freq_scale,   পরিমাণ(ভিন্ন));
    memcpy(params +  7, &ext_factor,   পরিমাণ(ভিন্ন));
    memcpy(params +  8, &attn_factor,  পরিমাণ(ভিন্ন));
    memcpy(params +  9, &beta_fast,    পরিমাণ(ভিন্ন));
    memcpy(params + 10, &beta_slow,    পরিমাণ(ভিন্ন));
    যদি (mrope_used && sections) {
        memcpy(params + 11, sections,  পরিমাণ(int32_t) * GGML_MROPE_SECTIONS);
    } অন্যথা {
        memset(params + 11, 0,         পরিমাণ(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_rope(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, রিক্ত, n_dims, রিক্ত, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

সংরচনা ggml_tensor * ggml_rope_multi(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   sections[GGML_MROPE_SECTIONS],
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

সংরচনা ggml_tensor * ggml_rope_multi_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   sections[GGML_MROPE_SECTIONS],
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

সংরচনা ggml_tensor * ggml_rope_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, রিক্ত, n_dims, রিক্ত, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

সংরচনা ggml_tensor * ggml_rope_ext(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, c, n_dims, রিক্ত, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

সংরচনা ggml_tensor * ggml_rope_ext_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, c, n_dims, রিক্ত, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

সংরচনা ggml_tensor * ggml_rope_custom(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, রিক্ত, n_dims, রিক্ত, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

সংরচনা ggml_tensor * ggml_rope_custom_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    ঘূৰাই ggml_rope_impl(
        ctx, a, b, রিক্ত, n_dims, রিক্ত, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
স্থির ভিন্ন ggml_rope_yarn_corr_dim(পূর্ণাঙ্ক n_dims, পূর্ণাঙ্ক n_ctx_orig, ভিন্ন n_rot, ভিন্ন base) {
    ঘূৰাই n_dims * logf(n_ctx_orig / (n_rot * 2 * (ভিন্ন)M_PI)) / (2 * logf(base));
}

শূন্য ggml_rope_yarn_corr_dims(
    পূর্ণাঙ্ক n_dims, পূর্ণাঙ্ক n_ctx_orig, ভিন্ন freq_base, ভিন্ন beta_fast, ভিন্ন beta_slow, ভিন্ন dims[2]
) {
    // start and end correction dims
    ভিন্ন start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ভিন্ন end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

সংরচনা ggml_tensor * ggml_rope_ext_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    সংরচনা ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_rope_multi_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c,
        পূর্ণাঙ্ক                   n_dims,
        পূর্ণাঙ্ক                   sections[4],
        পূর্ণাঙ্ক                   mode,
        পূর্ণাঙ্ক                   n_ctx_orig,
        ভিন্ন                 freq_base,
        ভিন্ন                 freq_scale,
        ভিন্ন                 ext_factor,
        ভিন্ন                 attn_factor,
        ভিন্ন                 beta_fast,
        ভিন্ন                 beta_slow) {
    সংরচনা ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_rope_set_offset(
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ধ্রুৱ int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    ঘূৰাই a;
}

স্থির int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, পূর্ণাঙ্ক s, পূর্ণাঙ্ক p, পূর্ণাঙ্ক d) {
    ঘূৰাই (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
সংরচনা ggml_tensor * ggml_im2col(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   d0,
        পূর্ণাঙ্ক                   d1,
        bool                  is_2D,
        গণনা ggml_type        dst_type) {
    যদি (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } অন্যথা {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ধ্রুৱ int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ধ্রুৱ int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ধ্রুৱ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_im2col_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        int64_t             * ne,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   d0,
        পূর্ণাঙ্ক                   d1,
        bool                  is_2D) {
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_conv_1d

সংরচনা ggml_tensor * ggml_conv_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   d0) {
    সংরচনা ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    সংরচনা ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    ঘূৰাই result;
}

// ggml_conv_1d_ph

সংরচনা ggml_tensor* ggml_conv_1d_ph(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s,
        পূর্ণাঙ্ক                   d) {
    ঘূৰাই ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

সংরচনা ggml_tensor * ggml_conv_1d_dw(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   d0) {
    সংরচনা ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    সংরচনা ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    সংরচনা ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    ঘূৰাই result;
}

// ggml_conv_1d_dw_ph

সংরচনা ggml_tensor * ggml_conv_1d_dw_ph(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   d0) {
    ঘূৰাই ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

সংরচনা ggml_tensor * ggml_col2im_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   oc,
        পূর্ণাঙ্ক                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ধ্রুৱ int64_t K_OC = a->ne[0];
    ধ্রুৱ int64_t T_in = a->ne[1];
    ধ্রুৱ int64_t K = K_OC / oc;
    ধ্রুৱ int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ধ্রুৱ int64_t ne[4] = { T_out, oc, 1, 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_conv_transpose_1d

স্থির int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, পূর্ণাঙ্ক s, পূর্ণাঙ্ক p, পূর্ণাঙ্ক d) {
    ঘূৰাই (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API সংরচনা ggml_tensor * ggml_conv_transpose_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ধ্রুৱ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
সংরচনা ggml_tensor * ggml_conv_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   d0,
        পূর্ণাঙ্ক                   d1) {
    সংরচনা ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    সংরচনা ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ঘূৰাই result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
সংরচনা ggml_tensor * ggml_im2col_3d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        int64_t               IC,
        পূর্ণাঙ্ক                   s0, // stride width
        পূর্ণাঙ্ক                   s1, // stride height
        পূর্ণাঙ্ক                   s2, // stride depth
        পূর্ণাঙ্ক                   p0, // padding width
        পূর্ণাঙ্ক                   p1, // padding height
        পূর্ণাঙ্ক                   p2, // padding depth
        পূর্ণাঙ্ক                   d0, // dilation width
        পূর্ণাঙ্ক                   d1, // dilation height
        পূর্ণাঙ্ক                   d2, // dilation depth
        গণনা ggml_type        dst_type) {
    ধ্রুৱ int64_t N = b->ne[3] / IC;
    ধ্রুৱ int64_t ID = b->ne[2];
    ধ্রুৱ int64_t IH = b->ne[1];
    ধ্রুৱ int64_t IW = b->ne[0];

    ধ্রুৱ int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    ধ্রুৱ int64_t KD = a->ne[2];
    ধ্রুৱ int64_t KH = a->ne[1];
    ধ্রুৱ int64_t KW = a->ne[0];
    ধ্রুৱ int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    ধ্রুৱ int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    ধ্রুৱ int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ধ্রুৱ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
সংরচনা ggml_tensor * ggml_conv_3d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        int64_t               IC,
        পূর্ণাঙ্ক                   s0, // stride width
        পূর্ণাঙ্ক                   s1, // stride height
        পূর্ণাঙ্ক                   s2, // stride depth
        পূর্ণাঙ্ক                   p0, // padding width
        পূর্ণাঙ্ক                   p1, // padding height
        পূর্ণাঙ্ক                   p2, // padding depth
        পূর্ণাঙ্ক                   d0, // dilation width
        পূর্ণাঙ্ক                   d1, // dilation height
        পূর্ণাঙ্ক                   d2  // dilation depth
        ) {
    সংরচনা ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    সংরচনা ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ঘূৰাই result;
}

// ggml_conv_2d_sk_p0

সংরচনা ggml_tensor * ggml_conv_2d_sk_p0(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

সংরচনা ggml_tensor * ggml_conv_2d_s1_ph(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    ঘূৰাই ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

সংরচনা ggml_tensor * ggml_conv_2d_dw(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   d0,
        পূর্ণাঙ্ক                   d1) {
    সংরচনা ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    সংরচনা ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    সংরচনা ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    সংরচনা ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ঘূৰাই result;
}

// ggml_conv_2d_dw_direct

সংরচনা ggml_tensor * ggml_conv_2d_dw_direct(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   stride0,
        পূর্ণাঙ্ক                   stride1,
        পূর্ণাঙ্ক                   pad0,
        পূর্ণাঙ্ক                   pad1,
        পূর্ণাঙ্ক                   dilation0,
        পূর্ণাঙ্ক                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    যদি (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ধ্রুৱ int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    ঘূৰাই result;
}

// ggml_conv_2d_direct

সংরচনা ggml_tensor * ggml_conv_2d_direct(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        সংরচনা ggml_tensor  * b,   // input data [W, H, C, N]
        পূর্ণাঙ্ক                   s0,  // stride dimension 0
        পূর্ণাঙ্ক                   s1,  // stride dimension 1
        পূর্ণাঙ্ক                   p0,  // padding dimension 0
        পূর্ণাঙ্ক                   p1,  // padding dimension 1
        পূর্ণাঙ্ক                   d0,  // dilation dimension 0
        পূর্ণাঙ্ক                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_conv_3d_direct

সংরচনা ggml_tensor * ggml_conv_3d_direct(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        পূর্ণাঙ্ক                   s2,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   p2,
        পূর্ণাঙ্ক                   d0,
        পূর্ণাঙ্ক                   d1,
        পূর্ণাঙ্ক                   d2,
        পূর্ণাঙ্ক                   c,
        পূর্ণাঙ্ক                   n,
        পূর্ণাঙ্ক                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    ঘূৰাই result;
}

// ggml_conv_transpose_2d_p0

স্থির int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, পূর্ণাঙ্ক s, পূর্ণাঙ্ক p) {
    ঘূৰাই (ins - 1) * s - 2 * p + ks;
}

সংরচনা ggml_tensor * ggml_conv_transpose_2d_p0(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        পূর্ণাঙ্ক                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ধ্রুৱ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    সংরচনা ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_pool_*

স্থির int64_t ggml_calc_pool_output_size(int64_t ins, পূর্ণাঙ্ক ks, পূর্ণাঙ্ক s, ভিন্ন p) {
    ঘূৰাই (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

সংরচনা ggml_tensor * ggml_pool_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_op_pool     op,
        পূর্ণাঙ্ক                   k0,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   p0) {
    ধ্রুৱ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_pool_2d

সংরচনা ggml_tensor * ggml_pool_2d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_op_pool     op,
        পূর্ণাঙ্ক                   k0,
        পূর্ণাঙ্ক                   k1,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        ভিন্ন                 p0,
        ভিন্ন                 p1) {
    সংরচনা ggml_tensor * result;
    ধ্রুৱ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_pool_2d_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * af,
        গণনা ggml_op_pool     op,
        পূর্ণাঙ্ক                   k0,
        পূর্ণাঙ্ক                   k1,
        পূর্ণাঙ্ক                   s0,
        পূর্ণাঙ্ক                   s1,
        ভিন্ন                 p0,
        ভিন্ন                 p1) {
    সংরচনা ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    ঘূৰাই result;
}

// ggml_upscale / ggml_interpolate

স্থির সংরচনা ggml_tensor * ggml_interpolate_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_গুনতি);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_upscale(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   scale_factor,
        গণনা ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    ঘূৰাই ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

সংরচনা ggml_tensor * ggml_upscale_ext(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   ne0,
        পূর্ণাঙ্ক                   ne1,
        পূর্ণাঙ্ক                   ne2,
        পূর্ণাঙ্ক                   ne3,
        গণনা ggml_scale_mode  mode) {
    ঘূৰাই ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

সংরচনা ggml_tensor * ggml_interpolate(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    ঘূৰাই ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

সংরচনা ggml_tensor * ggml_pad(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   p2,
        পূর্ণাঙ্ক                   p3) {
    ঘূৰাই ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

সংরচনা ggml_tensor * ggml_pad_circular(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1,
        পূর্ণাঙ্ক                   p2,
        পূর্ণাঙ্ক                   p3) {
    ঘূৰাই ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

সংরচনা ggml_tensor * ggml_pad_ext(
            সংরচনা ggml_context * ctx,
            সংরচনা ggml_tensor  * a,
            পূর্ণাঙ্ক                  lp0,
            পূর্ণাঙ্ক                  rp0,
            পূর্ণাঙ্ক                  lp1,
            পূর্ণাঙ্ক                  rp1,
            পূর্ণাঙ্ক                  lp2,
            পূর্ণাঙ্ক                  rp2,
            পূর্ণাঙ্ক                  lp3,
            পূর্ণাঙ্ক                  rp3
            ) {
    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    ঘূৰাই result;
}

// ggml_pad_ext_circular

সংরচনা ggml_tensor * ggml_pad_ext_circular(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                  lp0,
        পূর্ণাঙ্ক                  rp0,
        পূর্ণাঙ্ক                  lp1,
        পূর্ণাঙ্ক                  rp1,
        পূর্ণাঙ্ক                  lp2,
        পূর্ণাঙ্ক                  rp2,
        পূর্ণাঙ্ক                  lp3,
        পূর্ণাঙ্ক                  rp3
        ) {
    সংরচনা ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    ঘূৰাই result;
}

// ggml_pad_reflect_1d

সংরচনা ggml_tensor * ggml_pad_reflect_1d(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   p0,
        পূর্ণাঙ্ক                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_roll

সংরচনা ggml_tensor * ggml_roll(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   shift0,
        পূর্ণাঙ্ক                   shift1,
        পূর্ণাঙ্ক                   shift2,
        পূর্ণাঙ্ক                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_timestep_embedding

সংরচনা ggml_tensor * ggml_timestep_embedding(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * timesteps,
        পূর্ণাঙ্ক                   dim,
        পূর্ণাঙ্ক                   max_period) {

    সংরচনা ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    ঘূৰাই result;
}

// ggml_tri

সংরচনা ggml_tensor * ggml_tri(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    গণনা ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_fill

স্থির সংরচনা ggml_tensor * ggml_fill_impl(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    ভিন্ন                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_fill(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    ভিন্ন                 c) {
    ঘূৰাই ggml_fill_impl(ctx, a, c, false);
}

সংরচনা ggml_tensor * ggml_fill_inplace(
    সংরচনা ggml_context * ctx,
    সংরচনা ggml_tensor  * a,
    ভিন্ন                 c) {
    ঘূৰাই ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

সংরচনা ggml_tensor * ggml_argsort(
        সংরচনা ggml_context  * ctx,
        সংরচনা ggml_tensor   * a,
        গণনা ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_argsort_top_k

সংরচনা ggml_tensor * ggml_argsort_শীৰ্ষ_k(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    সংরচনা ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    ঘূৰাই result;
}

// ggml_top_k

সংরচনা ggml_tensor * ggml_শীৰ্ষ_k(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_শীৰ্ষ_K;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_arange

সংরচনা ggml_tensor * ggml_arange(
        সংরচনা ggml_context * ctx,
        ভিন্ন                 start,
        ভিন্ন                 stop,
        ভিন্ন                 step) {
    GGML_ASSERT(stop > start);

    ধ্রুৱ int64_t steps = (int64_t) ceilf((stop - start) / step);

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    ঘূৰাই result;
}

// ggml_flash_attn_ext

সংরচনা ggml_tensor * ggml_flash_attn_ext(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * q,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * mask,
        ভিন্ন                 scale,
        ভিন্ন                 max_bias,
        ভিন্ন                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    যদি (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    যদি (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ভিন্ন params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    ঘূৰাই result;
}


শূন্য ggml_flash_attn_ext_set_prec(
        সংরচনা ggml_tensor * a,
        গণনা ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ধ্রুৱ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

গণনা ggml_prec ggml_flash_attn_ext_get_prec(
        ধ্রুৱ সংরচনা ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ধ্রুৱ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ঘূৰাই (গণনা ggml_prec) prec_i32;
}

শূন্য ggml_flash_attn_ext_set_n_kv_max(
        সংরচনা ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

শূন্য ggml_flash_attn_ext_add_sinks(
        সংরচনা ggml_tensor * a,
        সংরচনা ggml_tensor * sinks) {
    যদি (!sinks) {
        a->src[4] = রিক্ত;
        ঘূৰাই;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == রিক্ত);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

সংরচনা ggml_tensor * ggml_flash_attn_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * q,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ধ্রুৱ int64_t     D = q->ne[0];
    ধ্রুৱ int64_t     N = q->ne[1];
    ধ্রুৱ int64_t     M = k->ne[1];
    ধ্রুৱ int64_t   ne2 = q->ne[2];
    ধ্রুৱ int64_t   ne3 = q->ne[3];
    ধ্রুৱ int64_t kvne2 = k->ne[2];

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
    ধ্রুৱ int64_t elem_q = ggml_nelements(q);
    ধ্রুৱ int64_t elem_k = ggml_nelements(k);
    ধ্রুৱ int64_t elem_v = ggml_nelements(v);

    গণনা ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ধ্রুৱ size_t tsize = ggml_type_size(result_type);

    ধ্রুৱ size_t offs_q = 0;
    ধ্রুৱ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ধ্রুৱ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ধ্রুৱ size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ধ্রুৱ size_t nelements = (end + tsize - 1)/tsize;

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, পরিমাণ(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    ঘূৰাই result;
}

// ggml_ssm_conv

সংরচনা ggml_tensor * ggml_ssm_conv(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * sx,
        সংরচনা ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ধ্রুৱ int64_t d_conv  = c->ne[0];
    ধ্রুৱ int64_t d_inner = c->ne[1];
    ধ্রুৱ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ধ্রুৱ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    সংরচনা ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    ঘূৰাই result;
}

// ggml_ssm_scan

সংরচনা ggml_tensor * ggml_ssm_scan(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * s,
        সংরচনা ggml_tensor  * x,
        সংরচনা ggml_tensor  * dt,
        সংরচনা ggml_tensor  * A,
        সংরচনা ggml_tensor  * B,
        সংরচনা ggml_tensor  * C,
        সংরচনা ggml_tensor  * ids,
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
        ধ্রুৱ int64_t d_state      = s->ne[0];
        ধ্রুৱ int64_t head_dim     = x->ne[0];
        ধ্রুৱ int64_t n_head       = x->ne[1];
        ধ্রুৱ int64_t n_seq_tokens = x->ne[2];
        ধ্রুৱ int64_t n_seqs       = x->ne[3];

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

        যদি (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    ঘূৰাই result;
}

// ggml_win_part

সংরচনা ggml_tensor * ggml_win_part(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ধ্রুৱ পূর্ণাঙ্ক px = (w - a->ne[1]%w)%w;
    ধ্রুৱ পূর্ণাঙ্ক py = (w - a->ne[2]%w)%w;

    ধ্রুৱ পূর্ণাঙ্ক npx = (px + a->ne[1])/w;
    ধ্রুৱ পূর্ণাঙ্ক npy = (py + a->ne[2])/w;
    ধ্রুৱ পূর্ণাঙ্ক np  = npx*npy;

    ধ্রুৱ int64_t ne[4] = { a->ne[0], w, w, np, };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_win_unpart

সংরচনা ggml_tensor * ggml_win_unpart(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   w0,
        পূর্ণাঙ্ক                   h0,
        পূর্ণাঙ্ক                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ধ্রুৱ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, পরিমাণ(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_get_rel_pos

সংরচনা ggml_tensor * ggml_get_rel_pos(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        পূর্ণাঙ্ক                   qh,
        পূর্ণাঙ্ক                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ধ্রুৱ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    ঘূৰাই result;
}

// ggml_add_rel_pos

স্থির সংরচনা ggml_tensor * ggml_add_rel_pos_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * pw,
        সংরচনা ggml_tensor  * ph,
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

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_add_rel_pos(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * pw,
        সংরচনা ggml_tensor  * ph) {
    ঘূৰাই ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

সংরচনা ggml_tensor * ggml_add_rel_pos_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * pw,
        সংরচনা ggml_tensor  * ph) {
    ঘূৰাই ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

সংরচনা ggml_tensor * ggml_rwkv_wkv6(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * r,
        সংরচনা ggml_tensor  * tf,
        সংরচনা ggml_tensor  * td,
        সংরচনা ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুৱ int64_t S = k->ne[0];
    ধ্রুৱ int64_t H = k->ne[1];
    ধ্রুৱ int64_t n_tokens = k->ne[2];
    ধ্রুৱ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুৱ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    ঘূৰাই result;
}

// ggml_gated_linear_attn

সংরচনা ggml_tensor * ggml_gated_linear_attn(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * q,
        সংরচনা ggml_tensor  * g,
        সংরচনা ggml_tensor  * state,
        ভিন্ন scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুৱ int64_t S = k->ne[0];
    ধ্রুৱ int64_t H = k->ne[1];
    ধ্রুৱ int64_t n_tokens = k->ne[2];
    ধ্রুৱ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুৱ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    ঘূৰাই result;
}

// ggml_rwkv_wkv7

সংরচনা ggml_tensor * ggml_rwkv_wkv7(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * r,
        সংরচনা ggml_tensor  * w,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুৱ int64_t S = k->ne[0];
    ধ্রুৱ int64_t H = k->ne[1];
    ধ্রুৱ int64_t n_tokens = k->ne[2];
    ধ্রুৱ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুৱ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    ঘূৰাই result;
}

// ggml_unary

স্থির সংরচনা ggml_tensor * ggml_unary_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_unary(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_unary_op    op) {
    ঘূৰাই ggml_unary_impl(ctx, a, op, false);
}

সংরচনা ggml_tensor * ggml_unary_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        গণনা ggml_unary_op    op) {
    ঘূৰাই ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

স্থির সংরচনা ggml_tensor * ggml_map_custom1_impl(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        ধ্রুৱ  ggml_custom1_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    সংরচনা ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_map_custom1(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        ধ্রুৱ  ggml_custom1_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

সংরচনা ggml_tensor * ggml_map_custom1_inplace(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        ধ্রুৱ  ggml_custom1_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

স্থির সংরচনা ggml_tensor * ggml_map_custom2_impl(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        ধ্রুৱ  ggml_custom2_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    সংরচনা ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_map_custom2(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        ধ্রুৱ  ggml_custom2_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

সংরচনা ggml_tensor * ggml_map_custom2_inplace(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        ধ্রুৱ  ggml_custom2_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

স্থির সংরচনা ggml_tensor * ggml_map_custom3_impl(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        সংরচনা ggml_tensor       * c,
        ধ্রুৱ  ggml_custom3_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    সংরচনা ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    সংরচনা ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_map_custom3(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        সংরচনা ggml_tensor       * c,
        ধ্রুৱ  ggml_custom3_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

সংরচনা ggml_tensor * ggml_map_custom3_inplace(
        সংরচনা ggml_context      * ctx,
        সংরচনা ggml_tensor       * a,
        সংরচনা ggml_tensor       * b,
        সংরচনা ggml_tensor       * c,
        ধ্রুৱ  ggml_custom3_op_t   fun,
        পূর্ণাঙ্ক                        n_tasks,
        শূন্য                     * userdata) {
    ঘূৰাই ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

সংরচনা ggml_tensor * ggml_custom_4d(
        সংরচনা ggml_context * ctx,
        গণনা ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        সংরচনা ggml_tensor ** args,
        পূর্ণাঙ্ক                   n_args,
        ggml_custom_op_t      fun,
        পূর্ণাঙ্ক                   n_tasks,
        শূন্য                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    সংরচনা ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op = GGML_OP_CUSTOM;
    বাবে (পূর্ণাঙ্ক i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_custom_inplace(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor ** args,
        পূর্ণাঙ্ক                   n_args,
        ggml_custom_op_t      fun,
        পূর্ণাঙ্ক                   n_tasks,
        শূন্য                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);

    সংরচনা ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, পরিমাণ(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    বাবে (পূর্ণাঙ্ক i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    ঘূৰাই result;
}
// ggml_cross_entropy_loss

সংরচনা ggml_tensor * ggml_cross_entropy_loss(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_cross_entropy_loss_back

সংরচনা ggml_tensor * ggml_cross_entropy_loss_back(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        সংরচনা ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    সংরচনা ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    ঘূৰাই result;
}

// opt_step_adamw

সংরচনা ggml_tensor * ggml_opt_step_adamw(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * grad,
        সংরচনা ggml_tensor  * m,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    ঘূৰাই result;
}

// opt_step_sgd

সংরচনা ggml_tensor * ggml_opt_step_sgd(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * grad,
        সংরচনা ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    সংরচনা ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    ঘূৰাই result;
}

// solve_tri

সংরচনা ggml_tensor * ggml_solve_tri(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * a,
        সংরচনা ggml_tensor  * b,
        bool                  বাম,
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

    GGML_ASSERT(lower && বাম && !uni); // TODO: support other variants

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    ঘূৰাই result;
}

// ggml_gated_delta_net

সংরচনা ggml_tensor * ggml_gated_delta_net(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * q,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * v,
        সংরচনা ggml_tensor  * g,
        সংরচনা ggml_tensor  * beta,
        সংরচনা ggml_tensor  * state,
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

    ধ্রুৱ int64_t S_v      = v->ne[0];
    ধ্রুৱ int64_t H        = v->ne[1];
    ধ্রুৱ int64_t n_tokens = v->ne[2];
    ধ্রুৱ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ধ্রুৱ int64_t state_rows = K * S_v * n_seqs;
    ধ্রুৱ int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    ঘূৰাই result;
}

// ggml_lightning_indexer

সংরচনা ggml_tensor * ggml_lightning_indexer(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * q,
        সংরচনা ggml_tensor  * k,
        সংরচনা ggml_tensor  * weights,
        সংরচনা ggml_tensor  * mask) {

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
    সংরচনা ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    ঘূৰাই result;
}

// ggml_dsv4_hc_comb

সংরচনা ggml_tensor * ggml_dsv4_hc_comb(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * mixes,
        সংরচনা ggml_tensor  * scale,
        সংরচনা ggml_tensor  * base,
        ভিন্ন                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ধ্রুৱ int64_t hc_mix_dim = mixes->ne[0];
    ধ্রুৱ int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    বাবে (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        যদি ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ভঙ্গ;
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

    সংরচনা ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    ঘূৰাই result;
}

// ggml_dsv4_hc_pre

স্থির সংরচনা ggml_tensor * ggml_dsv4_hc_pre_impl(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * x,
        সংরচনা ggml_tensor  * weights,
        ভিন্ন                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ধ্রুৱ int64_t n_embd   = x->ne[0];
    ধ্রুৱ int64_t hc       = x->ne[1];
    ধ্রুৱ int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    যদি (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } অন্যথা {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    সংরচনা ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_dsv4_hc_pre(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * x,
        সংরচনা ggml_tensor  * weights) {
    ঘূৰাই ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

সংরচনা ggml_tensor * ggml_dsv4_hc_pre_gated(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * x,
        সংরচনা ggml_tensor  * gate,
        ভিন্ন                 scale) {
    ঘূৰাই ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

সংরচনা ggml_tensor * ggml_dsv4_hc_post(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * x,
        সংরচনা ggml_tensor  * residual,
        সংরচনা ggml_tensor  * post,
        সংরচনা ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ধ্রুৱ int64_t n_embd   = x->ne[0];
    ধ্রুৱ int64_t n_tokens = x->ne[1];
    ধ্রুৱ int64_t hc       = residual->ne[1];

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

    যদি (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    সংরচনা ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    ঘূৰাই result;
}

// ggml_top_k_mask

সংরচনা ggml_tensor * ggml_শীৰ্ষ_k_mask(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_tensor  * mask,
        সংরচনা ggml_tensor  * শীৰ্ষ_k,
        সংরচনা ggml_tensor  * score,
        সংরচনা ggml_tensor  * blocks,
        সংরচনা ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(শীৰ্ষ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(শীৰ্ষ_k->ne[1] % mask->ne[1] == 0 && শীৰ্ষ_k->ne[2] % mask->ne[2] == 0 && শীৰ্ষ_k->ne[3] % mask->ne[3] == 0);

    যদি (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == শীৰ্ষ_k->ne[1] && score->ne[2] == শীৰ্ষ_k->ne[2] && score->ne[3] == শীৰ্ষ_k->ne[3]);
    }

    যদি (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    যদি (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == শীৰ্ষ_k->ne[1] && extra->ne[2] == শীৰ্ষ_k->ne[2] && extra->ne[3] == শীৰ্ষ_k->ne[3]);
    }

    সংরচনা ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], শীৰ্ষ_k->ne[1], শীৰ্ষ_k->ne[2], শীৰ্ষ_k->ne[3]);

    result->op     = GGML_OP_শীৰ্ষ_K_MASK;
    result->src[0] = mask;
    result->src[1] = শীৰ্ষ_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    ঘূৰাই result;
}

////////////////////////////////////////////////////////////////////////////////

সংরচনা ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    সংরচনা ggml_hash_set result;
    result.size = size;
    result.চাবিগুলি = GGML_MALLOC(পরিমাণ(সংরচনা ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), পরিমাণ(ggml_bitset_t));
    ঘূৰাই result;
}

শূন্য ggml_hash_set_reset(সংরচনা ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, পরিমাণ(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

শূন্য ggml_hash_set_free(সংরচনা ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->চাবিগুলি);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    স্থির ধ্রুৱ size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    স্থির ধ্রুৱ size_t n_primes = পরিমাণ(primes)/পরিমাণ(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    যেতিয়ালৈ (l < r) {
        size_t m = (l + r)/2;
        যদি (primes[m] < min_sz) {
            l = m + 1;
        } অন্যথা {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    ঘূৰাই sz;
}

সংরচনা hash_map {
    সংরচনা ggml_hash_set set;
    সংরচনা ggml_tensor ** vals;
};

স্থির সংরচনা hash_map * ggml_new_hash_map(size_t size) {
    সংরচনা hash_map * result = GGML_MALLOC(পরিমাণ(সংরচনা hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, পরিমাণ(সংরচনা ggml_tensor *));
    ঘূৰাই result;
}

স্থির শূন্য ggml_hash_map_free(সংরচনা hash_map * map) {
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

স্থির শূন্য ggml_add_or_set(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_cgraph  * cgraph,
        size_t                isrc,
        সংরচনা ggml_tensor  * tensor) {
    সংরচনা ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
    GGML_ASSERT(src);
    যদি (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } অন্যথা {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

স্থির শূন্য ggml_acc_or_set(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_cgraph  * cgraph,
        size_t                isrc,
        সংরচনা ggml_tensor  * tensor,
        ধ্রুৱ  size_t         nb1,
        ধ্রুৱ  size_t         nb2,
        ধ্রুৱ  size_t         nb3,
        ধ্রুৱ  size_t         offset) {
    সংরচনা ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
    GGML_ASSERT(src);
    যদি (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } অন্যথা {
        সংরচনা ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.চাবিগুলি[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

স্থির শূন্য ggml_add1_or_set(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_cgraph  * cgraph,
        size_t                isrc,
        সংরচনা ggml_tensor  * tensor) {
    সংরচনা ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
    GGML_ASSERT(src);
    যদি (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } অন্যথা {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

স্থির শূন্য ggml_sub_or_set(
        সংরচনা ggml_context * ctx,
        সংরচনা ggml_cgraph  * cgraph,
        size_t                isrc,
        সংরচনা ggml_tensor  * tensor) {
    সংরচনা ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
    GGML_ASSERT(src);
    যদি (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } অন্যথা {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

স্থির শূন্য ggml_compute_backward(
        সংরচনা ggml_context * ctx, সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক i, ধ্রুৱ bool * grads_needed) {
    সংরচনা ggml_tensor * tensor = cgraph->nodes[i];
    সংরচনা ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    যদি (!grad) {
        ঘূৰাই;
    }

    সংরচনা ggml_tensor * src0 = tensor->src[0];
    সংরচনা ggml_tensor * src1 = tensor->src[1];
    সংরচনা ggml_tensor * src2 = tensor->src[2];
    সংরচনা ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ধ্রুৱ size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ধ্রুৱ size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ধ্রুৱ size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ধ্রুৱ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ধ্রুৱ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ধ্রুৱ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    ৱিকল্প (tensor->op) {
        পক্ষ GGML_OP_DUP: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_ADD: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                সংরচনা ggml_tensor * tmp = grad;
                যদি (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_ADD1: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_ACC: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ধ্রুৱ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ধ্রুৱ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ধ্রুৱ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ধ্রুৱ size_t offset = ((int32_t *) tensor->op_params)[3];

                সংরচনা ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SUB: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_MUL: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            যদি (src1_needs_grads) {
                সংরচনা ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                যদি (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_DIV: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            যদি (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SQR: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SQRT: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_LOG: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SIN: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_COS: {
            যদি (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SUM: {
            যদি (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SUM_ROWS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_MEAN: {
            যদি (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_REPEAT: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_REPEAT_BACK: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM: {
            যদি (src0_needs_grads) {
                ভিন্ন eps;
                memcpy(&eps, tensor->op_params, পরিমাণ(ভিন্ন));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT: {
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

            যদি (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                সংরচনা ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                যদি (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ধ্রুৱ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ধ্রুৱ size_t nb2 = tmp->nb[2] * nr2;
                    ধ্রুৱ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            যদি (src1_needs_grads) {
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
        } ভঙ্গ;
        পক্ষ GGML_OP_SCALE: {
            যদি (src0_needs_grads) {
                ভিন্ন s;
                memcpy(&s, tensor->op_params, পরিমাণ(ভিন্ন));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SET: {
            ধ্রুৱ size_t nb1    = ((ধ্রুৱ int32_t *) tensor->op_params)[0];
            ধ্রুৱ size_t nb2    = ((ধ্রুৱ int32_t *) tensor->op_params)[1];
            ধ্রুৱ size_t nb3    = ((ধ্রুৱ int32_t *) tensor->op_params)[2];
            ধ্রুৱ size_t offset = ((ধ্রুৱ int32_t *) tensor->op_params)[3];

            সংরচনা ggml_tensor * tensor_grad_view = রিক্ত;

            যদি (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            যদি (src0_needs_grads) {
                সংরচনা ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            যদি (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            যদি (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            যদি (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_CONT: {
            // same as cpy
            যদি (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_RESHAPE: {
            যদি (src0_needs_grads) {
                সংরচনা ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_VIEW: {
            যদি (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, পরিমাণ(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                যদি (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } ভঙ্গ;
        পক্ষ GGML_OP_PERMUTE: {
            যদি (src0_needs_grads) {
                ধ্রুৱ int32_t * axes = (ধ্রুৱ int32_t *) tensor->op_params;
                ধ্রুৱ পূর্ণাঙ্ক axis0 = axes[0] & 0x3;
                ধ্রুৱ পূর্ণাঙ্ক axis1 = axes[1] & 0x3;
                ধ্রুৱ পূর্ণাঙ্ক axis2 = axes[2] & 0x3;
                ধ্রুৱ পূর্ণাঙ্ক axis3 = axes[3] & 0x3;
                পূর্ণাঙ্ক axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_TRANSPOSE: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            যদি (src1_needs_grads) {
                // noop
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_INF: {
            যদি (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ধ্রুৱ পূর্ণাঙ্ক n_past = ((ধ্রুৱ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_ZERO: {
            যদি (src0_needs_grads) {
                ধ্রুৱ পূর্ণাঙ্ক n_past = ((ধ্রুৱ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX: {
            যদি (src0_needs_grads) {
                ভিন্ন scale    = 1.0f;
                ভিন্ন max_bias = 0.0f;

                memcpy(&scale,    (ধ্রুৱ ভিন্ন *) tensor->op_params + 0, পরিমাণ(ভিন্ন));
                memcpy(&max_bias, (ধ্রুৱ ভিন্ন *) tensor->op_params + 1, পরিমাণ(ভিন্ন));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ভঙ্গ;
        পক্ষ GGML_OP_ROPE: {
            যদি (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ধ্রুৱ পূর্ণাঙ্ক n_dims     = ((ধ্রুৱ int32_t *) tensor->op_params)[1];
                ধ্রুৱ পূর্ণাঙ্ক mode       = ((ধ্রুৱ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ধ্রুৱ পূর্ণাঙ্ক n_ctx_orig = ((ধ্রুৱ int32_t *) tensor->op_params)[4];
                ভিন্ন freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                পূর্ণাঙ্ক sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ধ্রুৱ ভিন্ন *) tensor->op_params +  5, পরিমাণ(ভিন্ন));
                memcpy(&freq_scale,  (ধ্রুৱ ভিন্ন *) tensor->op_params +  6, পরিমাণ(ভিন্ন));
                memcpy(&ext_factor,  (ধ্রুৱ ভিন্ন *) tensor->op_params +  7, পরিমাণ(ভিন্ন));
                memcpy(&attn_factor, (ধ্রুৱ ভিন্ন *) tensor->op_params +  8, পরিমাণ(ভিন্ন));
                memcpy(&beta_fast,   (ধ্রুৱ ভিন্ন *) tensor->op_params +  9, পরিমাণ(ভিন্ন));
                memcpy(&beta_slow,   (ধ্রুৱ ভিন্ন *) tensor->op_params + 10, পরিমাণ(ভিন্ন));
                memcpy(&sections,                    tensor->op_params + 11, পরিমাণ(sections));

                সংরচনা ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ভঙ্গ;
        পক্ষ GGML_OP_IM2COL: {
            যদি (src1_needs_grads) {
                ধ্রুৱ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ধ্রুৱ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ধ্রুৱ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ধ্রুৱ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ধ্রুৱ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ধ্রুৱ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ধ্রুৱ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_POOL_2D: {
            যদি (src0_needs_grads) {
                ধ্রুৱ গণনা ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ধ্রুৱ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ধ্রুৱ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ধ্রুৱ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ধ্রুৱ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ধ্রুৱ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ধ্রুৱ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_WIN_PART:
        পক্ষ GGML_OP_WIN_UNPART:
        পক্ষ GGML_OP_UNARY: {
            ৱিকল্প (ggml_get_unary_op(tensor)) {
                পক্ষ GGML_UNARY_OP_ABS: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SGN: {
                    // noop
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_NEG: {
                    যদি (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_STEP: {
                    // noop
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_RELU: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SILU: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_EXP: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_EXPM1: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SOFTPLUS: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ভঙ্গ;
                স্ৱভাৱ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ভঙ্গ;
        পক্ষ GGML_OP_GLU: {
            ৱিকল্প (ggml_get_glu_op(tensor)) {
                পক্ষ GGML_GLU_OP_SWIGLU: {
                    যদি (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    যদি (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ভঙ্গ;
                স্ৱভাৱ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ভঙ্গ;
        পক্ষ GGML_OP_NONE: {
            // noop
        } ভঙ্গ;
        পক্ষ GGML_OP_গুনতি:
        স্ৱভাৱ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

স্থির size_t ggml_visit_parents_graph(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_tensor * গ্ৰন্থি, bool compute) {
    যদি (গ্ৰন্থি->op != GGML_OP_NONE && compute) {
        গ্ৰন্থি->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ধ্রুৱ size_t গ্ৰন্থি_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি);
    GGML_ASSERT(গ্ৰন্থি_hash_pos != GGML_HASHSET_FULL);

    যদি (ggml_bitset_get(cgraph->visited_hash_set.used, গ্ৰন্থি_hash_pos)) {
        // already visited

        যদি (compute) {
            // update the compute flag regardless
            বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; ++i) {
                সংরচনা ggml_tensor * src = গ্ৰন্থি->src[i];
                যদি (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        ঘূৰাই গ্ৰন্থি_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.চাবিগুলি[গ্ৰন্থি_hash_pos] = গ্ৰন্থি;
    ggml_bitset_set(cgraph->visited_hash_set.used, গ্ৰন্থি_hash_pos);
    cgraph->use_counts[গ্ৰন্থি_hash_pos] = 0;

    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; ++i) {
        ধ্রুৱ পূর্ণাঙ্ক k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_বাম_TO_ডান) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ডান_TO_বাম) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        সংরচনা ggml_tensor * src = গ্ৰন্থি->src[k];
        যদি (src) {
            ধ্রুৱ size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    যদি (গ্ৰন্থি->op == GGML_OP_NONE && !(গ্ৰন্থি->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        যদি (strlen(গ্ৰন্থি->name) == 0) {
            ggml_format_name(গ্ৰন্থি, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = গ্ৰন্থি;
        cgraph->n_leafs++;
    } অন্যথা {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        যদি (strlen(গ্ৰন্থি->name) == 0) {
            ggml_format_name(গ্ৰন্থি, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = গ্ৰন্থি;
        cgraph->n_nodes++;
    }

    ঘূৰাই গ্ৰন্থি_hash_pos;
}

স্থির শূন্য ggml_build_forward_impl(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_tensor * tensor, bool expand, bool compute) {
    যদি (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ধ্রুৱ পূর্ণাঙ্ক n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ধ্রুৱ পূর্ণাঙ্ক n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    যদি (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

সংরচনা ggml_tensor * ggml_build_forward_select(
        সংরচনা ggml_cgraph  * cgraph,
        সংরচনা ggml_tensor ** tensors,
        পূর্ণাঙ্ক                   n_tensors,
        পূর্ণাঙ্ক                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    বাবে (পূর্ণাঙ্ক i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ঘূৰাই tensors[idx];
}

শূন্য ggml_build_forward_expand(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

শূন্য ggml_build_forward_order(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

শূন্য ggml_build_backward_expand(
        সংরচনা ggml_context *  ctx,
        সংরচনা ggml_cgraph  *  cgraph,
        সংরচনা ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ধ্রুৱ পূর্ণাঙ্ক n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*পরিমাণ(সংরচনা ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*পরিমাণ(সংরচনা ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, পরিমাণ(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        বাবে (পূর্ণাঙ্ক i = 0; i < n_nodes_f; ++i) {
            সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
            any_params = any_params || (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < n_nodes_f; ++i) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        যদি (গ্ৰন্থি->type == GGML_TYPE_I32) {
            অনুৱর্ত;
        }

        bool গ্ৰন্থি_needs_grad = (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_PARAM) || (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ৱিকল্প (গ্ৰন্থি->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            পক্ষ GGML_OP_IM2COL:      // only used for its shape
            পক্ষ GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                ভঙ্গ;
            পক্ষ GGML_OP_UNARY: {
                ধ্রুৱ গণনা ggml_unary_op uop = ggml_get_unary_op(গ্ৰন্থি);
                // SGN and STEP unary ops are piecewise constant
                যদি (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } ভঙ্গ;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            পক্ষ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            পক্ষ GGML_OP_GET_ROWS:      // row indices not differentiable
            পক্ষ GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            পক্ষ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ভঙ্গ;

            স্ৱভাৱ:
                ভঙ্গ;
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; ++j) {
            যদি (!গ্ৰন্থি->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি->src[j])]) {
                অনুৱর্ত;
            }
            GGML_ASSERT(গ্ৰন্থি->src[j]->type == GGML_TYPE_F32 || গ্ৰন্থি->src[j]->type == GGML_TYPE_F16);
            গ্ৰন্থি_needs_grad = true;
            ভঙ্গ;
        }
        যদি (!গ্ৰন্থি_needs_grad) {
            অনুৱর্ত;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!গ্ৰন্থি->view_src || গ্ৰন্থি->op == GGML_OP_CPY || গ্ৰন্থি->op == GGML_OP_SET_ROWS || গ্ৰন্থি->op == GGML_OP_VIEW ||
            গ্ৰন্থি->op == GGML_OP_RESHAPE || গ্ৰন্থি->op == GGML_OP_PERMUTE || গ্ৰন্থি->op == GGML_OP_TRANSPOSE);

        ধ্রুৱ size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        যদি (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } অন্যথা যদি (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, গ্ৰন্থি->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    বাবে (পূর্ণাঙ্ক i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    মুক্ত(grads_needed);
}

স্থির শূন্য * incr_ptr_aligned(শূন্য ** p, size_t size, size_t align) {
    শূন্য * ptr = *p;
    ptr = (শূন্য *) GGML_PAD((uintptr_t) ptr, align);
    *p = (শূন্য *) ((ৱর্ণ *) ptr + size);
    ঘূৰাই ptr;
}

স্থির size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    শূন্য * p = (ৱর্ণ *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, পরিমাণ(সংরচনা ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * পরিমাণ(int32_t), পরিমাণ(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)); // hash keys
    যদি (grads) {
        incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * পরিমাণ(ggml_bitset_t), পরিমাণ(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ঘূৰাই nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    ঘূৰাই GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(শূন্য) {
    ঘূৰাই ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

সংরচনা ggml_cgraph * ggml_new_graph_custom(সংরচনা ggml_context * ctx, size_t size, bool grads) {
    ধ্রুৱ size_t obj_size = ggml_graph_nbytes(size, grads);
    সংরচনা ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    সংরচনা ggml_cgraph * cgraph = (সংরচনা ggml_cgraph *) ((ৱর্ণ *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    শূন্য * p = cgraph + 1;

    সংরচনা ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *));
    সংরচনা ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * পরিমাণ(int32_t), পরিমাণ(int32_t));
    সংরচনা ggml_tensor ** hash_চাবিগুলি_ptr  =         incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *));
    সংরচনা ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)) : রিক্ত;
    সংরচনা ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * পরিমাণ(সংরচনা ggml_tensor *), পরিমাণ(সংরচনা ggml_tensor *)) : রিক্ত;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * পরিমাণ(ggml_bitset_t), পরিমাণ(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    প্রতিজ্ঞা(obj_size == (size_t)((ৱর্ণ *)p - (ৱর্ণ *)cgraph));

    *cgraph = (সংরচনা ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_চাবিগুলি_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_বাম_TO_ডান,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    যদি (grads) {
        memset(cgraph->grads,     0, hash_size*পরিমাণ(সংরচনা ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*পরিমাণ(সংরচনা ggml_tensor *));
    }

    ঘূৰাই cgraph;
}

সংরচনা ggml_cgraph * ggml_new_graph(সংরচনা ggml_context * ctx) {
    ঘূৰাই ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

সংরচনা ggml_cgraph ggml_graph_view(সংরচনা ggml_cgraph * cgraph0, পূর্ণাঙ্ক i0, পূর্ণাঙ্ক i1) {
    সংরচনা ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ রিক্ত, // gradients would need visited_hash_set
        /*.grad_accs        =*/ রিক্ত,
        /*.leafs            =*/ রিক্ত,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    ঘূৰাই cgraph;
}

শূন্য ggml_graph_cpy(সংরচনা ggml_cgraph * src, সংরচনা ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    বাবে (পূর্ণাঙ্ক i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    বাবে (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        যদি (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_ঢোকাও(&dst->visited_hash_set, src->visited_hash_set.চাবিগুলি[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    যদি (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*পরিমাণ(সংরচনা ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*পরিমাণ(সংরচনা ggml_tensor *));
    }
    যদি (src->grads) {
        GGML_ASSERT(dst->grads     != রিক্ত);
        GGML_ASSERT(dst->grad_accs != রিক্ত);
        বাবে (পূর্ণাঙ্ক i = 0; i < src->n_nodes; ++i) {
            ধ্রুৱ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ধ্রুৱ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

সংরচনা ggml_cgraph * ggml_graph_dup(সংরচনা ggml_context * ctx, সংরচনা ggml_cgraph * cgraph, bool force_grads) {
    সংরচনা ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_set_zero(সংরচনা ggml_tensor * tensor) {
    যদি (ggml_is_empty(tensor)) {
        ঘূৰাই tensor;
    }
    যদি (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } অন্যথা {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    ঘূৰাই tensor;
}

শূন্য ggml_graph_reset(সংরচনা ggml_cgraph * cgraph) {
    যদি (!cgraph) {
        ঘূৰাই;
    }
    GGML_ASSERT(cgraph->grads != রিক্ত);

    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি     = cgraph->nodes[i];
        সংরচনা ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, গ্ৰন্থি);

        যদি (গ্ৰন্থি->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(গ্ৰন্থি->src[2]);
            ggml_set_zero(গ্ৰন্থি->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        যদি (grad_acc) {
            যদি (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ধ্রুৱ ভিন্ন onef = 1.0f;
                যদি (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, পরিমাণ(ভিন্ন));
                } অন্যথা {
                    GGML_ASSERT(grad_acc->data);
                    *((ভিন্ন *) grad_acc->data) = onef;
                }
            } অন্যথা {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

শূন্য ggml_graph_clear(সংরচনা ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

পূর্ণাঙ্ক ggml_graph_size(সংরচনা ggml_cgraph * cgraph) {
    ঘূৰাই cgraph->size;
}

সংরচনা ggml_tensor * ggml_graph_গ্ৰন্থি(সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক i) {
    যদি (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ঘূৰাই cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ঘূৰাই cgraph->nodes[i];
}

সংরচনা ggml_tensor ** ggml_graph_nodes(সংরচনা ggml_cgraph * cgraph) {
    ঘূৰাই cgraph->nodes;
}

পূর্ণাঙ্ক ggml_graph_n_nodes(সংরচনা ggml_cgraph * cgraph) {
    ঘূৰাই cgraph->n_nodes;
}

শূন্য ggml_graph_add_গ্ৰন্থি(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

সংরচনা ggml_tensor * ggml_graph_get_tensor(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ ৱর্ণ * name) {
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_leafs; i++) {
        সংরচনা ggml_tensor * leaf = cgraph->leafs[i];

        যদি (strcmp(leaf->name, name) == 0) {
            ঘূৰাই leaf;
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        যদি (strcmp(গ্ৰন্থি->name, name) == 0) {
            ঘূৰাই গ্ৰন্থি;
        }
    }

    ঘূৰাই রিক্ত;
}

সংরচনা ggml_tensor * ggml_graph_get_grad(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ সংরচনা ggml_tensor * গ্ৰন্থি) {
    ধ্রুৱ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি);
    ঘূৰাই igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : রিক্ত;
}

সংরচনা ggml_tensor * ggml_graph_get_grad_acc(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ সংরচনা ggml_tensor * গ্ৰন্থি) {
    ধ্রুৱ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি);
    ঘূৰাই igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : রিক্ত;
}

শূন্য ggml_graph_print(ধ্রুৱ সংরচনা ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                গ্ৰন্থি->ne[0], গ্ৰন্থি->ne[1], গ্ৰন্থি->ne[2],
                ggml_op_name(গ্ৰন্থি->op), (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, গ্ৰন্থি) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_leafs; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                গ্ৰন্থি->ne[0], গ্ৰন্থি->ne[1],
                ggml_op_name(গ্ৰন্থি->op),
                ggml_get_name(গ্ৰন্থি));
    }

    GGML_LOG_INFO("========================================\n");
}

স্থির পূর্ণাঙ্ক ggml_গ্ৰন্থি_list_find_tensor(ধ্রুৱ সংরচনা ggml_cgraph * cgraph,
                                      ধ্রুৱ পূর্ণাঙ্ক *                idxs,
                                      পূর্ণাঙ্ক                        গুনতি,
                                      ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    বাবে (পূর্ণাঙ্ক i = 0; i < গুনতি; ++i) {
        ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_idx = idxs[i];

        যদি (গ্ৰন্থি_idx >= cgraph->n_nodes) {
            ঘূৰাই -1;
        }
        যদি (cgraph->nodes[গ্ৰন্থি_idx] == tensor) {
            ঘূৰাই i;
        }
    }
    ঘূৰাই -1;
}

স্থির bool ggml_is_constant(ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    ঘূৰাই tensor->buffer != রিক্ত && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ধ্রুৱ সংরচনা ggml_cgraph * cgraph,
                                ধ্রুৱ পূর্ণাঙ্ক *                গ্ৰন্থি_idxs,
                                পূর্ণাঙ্ক                        গুনতি,
                                ধ্রুৱ গণনা ggml_op *       ops,
                                ধ্রুৱ পূর্ণাঙ্ক *                outputs,
                                পূর্ণাঙ্ক                        অঙ্ক_outputs) {
    GGML_ASSERT(outputs && অঙ্ক_outputs > 0);

    বাবে (পূর্ণাঙ্ক i = 0; i < গুনতি; ++i) {
        যদি (গ্ৰন্থি_idxs[i] >= cgraph->n_nodes) {
            ঘূৰাই false;
        }

        ধ্রুৱ সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_idxs[i]];

        যদি (গ্ৰন্থি->op != ops[i]) {
            ঘূৰাই false;
        }

        যদি ((গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ঘূৰাই false;
        }

        যদি (ggml_গ্ৰন্থি_list_find_tensor(cgraph, outputs, অঙ্ক_outputs, গ্ৰন্থি) != -1) {
            অনুৱর্ত;
        }

        যদি (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ঘূৰাই false;
        }

        পূর্ণাঙ্ক subgraph_uses = 0;
        বাবে (পূর্ণাঙ্ক j = i + 1; j < গুনতি; ++j) {
            ধ্রুৱ সংরচনা ggml_tensor * other_গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_idxs[j]];
            বাবে (পূর্ণাঙ্ক src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                যদি (other_গ্ৰন্থি->src[src_idx] == গ্ৰন্থি) {
                    subgraph_uses++;
                }
            }
        }

        যদি (subgraph_uses != ggml_গ্ৰন্থি_get_use_গুনতি(cgraph, গ্ৰন্থি_idxs[i])) {
            ঘূৰাই false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        সংরচনা ggml_tensor * view_src = গ্ৰন্থি->view_src;
        যেতিয়ালৈ (view_src) {
            যদি (ggml_গ্ৰন্থি_list_find_tensor(cgraph, গ্ৰন্থি_idxs, গুনতি, view_src) == -1 && !ggml_is_constant(view_src)) {
                ঘূৰাই false;
            }
            view_src = view_src->view_src;
        }
    }

    ঘূৰাই true;
}

// check if node is part of the graph
স্থির bool ggml_graph_find(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ সংরচনা ggml_tensor * গ্ৰন্থি) {
    যদি (cgraph == রিক্ত) {
        ঘূৰাই true;
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        যদি (cgraph->nodes[i] == গ্ৰন্থি) {
            ঘূৰাই true;
        }
    }

    ঘূৰাই false;
}

স্থির সংরচনা ggml_tensor * ggml_graph_get_parent(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ সংরচনা ggml_tensor * গ্ৰন্থি) {
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * parent = cgraph->nodes[i];
        সংরচনা ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        যদি (grad == গ্ৰন্থি) {
            ঘূৰাই parent;
        }
    }

    ঘূৰাই রিক্ত;
}

স্থির শূন্য ggml_graph_dump_dot_গ্ৰন্থি_edge(FILE * fp, ধ্রুৱ সংরচনা ggml_cgraph * gb, সংরচনা ggml_tensor * গ্ৰন্থি, সংরচনা ggml_tensor * parent, ধ্রুৱ ৱর্ণ * label)  {
    সংরচনা ggml_tensor * gparent = ggml_graph_get_parent(gb, গ্ৰন্থি);
    সংরচনা ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (শূন্য *) gparent0 : (শূন্য *) parent,
            gparent ? (শূন্য *) gparent : (শূন্য *) গ্ৰন্থি,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

স্থির শূন্য ggml_graph_dump_dot_leaf_edge(FILE * fp, সংরচনা ggml_tensor * গ্ৰন্থি, সংরচনা ggml_tensor * parent, ধ্রুৱ ৱর্ণ * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (শূন্য *) parent,
            (শূন্য *) গ্ৰন্থি,
            label);
}

শূন্য ggml_graph_dump_dot(ধ্রুৱ সংরচনা ggml_cgraph * gb, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, ধ্রুৱ ৱর্ণ * filename) {
    ৱর্ণ color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    বাবে (পূর্ণাঙ্ক i = 0; i < gb->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = gb->nodes[i];
        সংরচনা ggml_tensor * grad = ggml_graph_get_grad(gb, গ্ৰন্থি);

        যদি (ggml_graph_get_parent(gb, গ্ৰন্থি) != রিক্ত) {
            অনুৱর্ত;
        }

        যদি (গ্ৰন্থি->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, পরিমাণ(color), "yellow");
        } অন্যথা যদি (grad) {
            যদি (ggml_graph_find(cgraph, গ্ৰন্থি)) {
                snprintf(color, পরিমাণ(color), "green");
            } অন্যথা {
                snprintf(color, পরিমাণ(color), "lightblue");
            }
        } অন্যথা {
            snprintf(color, পরিমাণ(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (শূন্য *) গ্ৰন্থি, color);

        যদি (strlen(গ্ৰন্থি->name) > 0) {
            fprintf(fp, "%s (%s)|", গ্ৰন্থি->name, ggml_type_name(গ্ৰন্থি->type));
        } অন্যথা {
            fprintf(fp, "(%s)|", ggml_type_name(গ্ৰন্থি->type));
        }

        যদি (ggml_is_matrix(গ্ৰন্থি)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, গ্ৰন্থি->ne[0], গ্ৰন্থি->ne[1], ggml_op_প্ৰতীক(গ্ৰন্থি->op));
        } অন্যথা {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, গ্ৰন্থি->ne[0], গ্ৰন্থি->ne[1], গ্ৰন্থি->ne[2], ggml_op_প্ৰতীক(গ্ৰন্থি->op));
        }

        যদি (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_প্ৰতীক(grad->op));
        } অন্যথা {
            fprintf(fp, "\"; ]\n");
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < gb->n_leafs; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = gb->leafs[i];

        snprintf(color, পরিমাণ(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (শূন্য *) গ্ৰন্থি, color);

        যদি (strlen(গ্ৰন্থি->name) > 0) {
            fprintf(fp, "%s (%s)|", গ্ৰন্থি->name, ggml_type_name(গ্ৰন্থি->type));
        } অন্যথা {
            fprintf(fp, "(%s)|", ggml_type_name(গ্ৰন্থি->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, গ্ৰন্থি->ne[0], গ্ৰন্থি->ne[1]);
        যদি (ggml_nelements(গ্ৰন্থি) < 5 && গ্ৰন্থি->data != রিক্ত) {
            fprintf(fp, " | (");
            বাবে (পূর্ণাঙ্ক j = 0; j < ggml_nelements(গ্ৰন্থি); j++) {
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
                যদি (j < ggml_nelements(গ্ৰন্থি) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < gb->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = gb->nodes[i];

        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            যদি (গ্ৰন্থি->src[j]) {
                ৱর্ণ label[16];
                snprintf(label, পরিমাণ(label), "src %d", j);
                ggml_graph_dump_dot_গ্ৰন্থি_edge(fp, gb, গ্ৰন্থি, গ্ৰন্থি->src[j], label);
            }
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < gb->n_leafs; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = gb->leafs[i];

        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            যদি (গ্ৰন্থি->src[j]) {
                ৱর্ণ label[16];
                snprintf(label, পরিমাণ(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, গ্ৰন্থি, গ্ৰন্থি->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_set_প্ৰবেশ(সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_প্ৰবেশ;
}

শূন্য ggml_set_output(সংরচনা ggml_tensor * tensor) {
    বাবে (সংরচনা ggml_tensor * cur = tensor; cur != রিক্ত; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

শূন্য ggml_set_param(সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

শূন্য ggml_set_loss(সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_quantize_init(গণনা ggml_type type) {
    ggml_critical_section_start();

    ৱিকল্প (type) {
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ2_S:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ভঙ্গ;
        স্ৱভাৱ: // nothing
            ভঙ্গ;
    }

    ggml_critical_section_end();
}

শূন্য ggml_quantize_free(শূন্য) {
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

bool ggml_quantize_requires_imatrix(গণনা ggml_type type) {
    ঘূৰাই
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        গণনা ggml_type   type,
           ধ্রুৱ ভিন্ন * src,
                  শূন্য * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ধ্রুৱ ভিন্ন * imatrix) {
    ধ্রুৱ int64_t n = nrows * n_per_row;

    যদি (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != রিক্ত);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ধ্রুৱ size_t start_row = start / n_per_row;
    ধ্রুৱ size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    ৱিকল্প (type) {
        পক্ষ GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (ৱর্ণ *) dst + start_row * row_size, nrows, n_per_row, imatrix); ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                size_t elemsize = পরিমাণ(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                size_t elemsize = পরিমাণ(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                size_t elemsize = পরিমাণ(ভিন্ন);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } ভঙ্গ;
        স্ৱভাৱ:
            প্রতিজ্ঞা(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    ঘূৰাই result;
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_log_get(ggml_log_callback * log_callback, শূন্য ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

শূন্য ggml_log_set(ggml_log_callback log_callback, শূন্য * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

শূন্য ggml_threadpool_params_init(সংরচনা ggml_threadpool_params * p, পূর্ণাঙ্ক n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

সংরচনা ggml_threadpool_params ggml_threadpool_params_default(পূর্ণাঙ্ক n_threads) {
    সংরচনা ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    ঘূৰাই p;
}

bool ggml_threadpool_params_match(ধ্রুৱ সংরচনা ggml_threadpool_params * p0, ধ্রুৱ সংরচনা ggml_threadpool_params * p1) {
    যদি (p0->n_threads  != p1->n_threads  ) ঘূৰাই false;
    যদি (p0->prio       != p1->prio       ) ঘূৰাই false;
    যদি (p0->poll       != p1->poll       ) ঘূৰাই false;
    যদি (p0->strict_cpu != p1->strict_cpu ) ঘূৰাই false;
    ঘূৰাই memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
