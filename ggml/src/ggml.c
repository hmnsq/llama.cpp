#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#অন্তর্ভাব "ggml-version.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-threading.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ggml.h"

// FIXME: required here for quantization functions
#অন্তর্ভাব "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#অন্তর্ভাব <hbwmalloc.h>
#endif

#যদি defined(_MSC_VER) || defined(__MINGW32__)
#অন্তর্ভাব <malloc.h> // using malloc.h with MSC/MINGW
#নতুবা_যদি !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#অন্তর্ভাব <alloca.h>
#endif

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <errno.h>
#অন্তর্ভাব <time.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <inttypes.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <float.h>
#অন্তর্ভাব <limits.h>
#অন্তর্ভাব <stdarg.h>
#অন্তর্ভাব <signal.h>
#যদি defined(__gnu_linux__)
#অন্তর্ভাব <syscall.h>
#endif

#যদি defined(__APPLE__)
#অন্তর্ভাব <unistd.h>
#অন্তর্ভাব <mach/mach.h>
#অন্তর্ভাব <TargetConditionals.h>
#endif

#যদি defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#অন্তর্ভাব <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_পরবর্তী_uid(শূন্য) {
#ifdef _MSC_VER
#যদি defined(_WIN32)
    স্থির অস্থির LONG গণক = 1;
    ফেরত (uint64_t) InterlockedIncrement(&গণক) - 1;
#অন্যথা
    স্থির অস্থির long long গণক = 1;
    ফেরত (uint64_t) _InterlockedIncrement64(&গণক) - 1;
#endif
#অন্যথা
    স্থির uint64_t গণক = 1;
    ফেরত __atomic_fetch_add(&গণক, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#যদি defined(__AVX512BF16__)
#যদি defined(_MSC_VER)
#define m512i(p) p
#অন্যথা
#অন্তর্ভাব <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#যদি defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#অন্তর্ভাব <unistd.h>
#অন্তর্ভাব <sys/types.h>
#অন্তর্ভাব <sys/stat.h>
#অন্তর্ভাব <sys/wait.h>
#যদি defined(__linux__)
#অন্তর্ভাব <sys/prctl.h>
#endif

#যদি defined(__ANDROID__)
#অন্তর্ভাব <unwind.h>
#অন্তর্ভাব <dlfcn.h>
#অন্তর্ভাব <stdio.h>

কাঠামো backtrace_state {
    শূন্য ** current;
    শূন্য ** end;
};

স্থির _Unwind_Reason_Code unwind_callback(কাঠামো _Unwind_Context* context, শূন্য* arg) {
    কাঠামো backtrace_state * state = (কাঠামো backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    যদি (pc) {
        যদি (state->current == state->end) {
            ফেরত URC_END_OF_স্তূপ;
        } অন্যথা {
            *state->current++ = (শূন্য*)pc;
        }
    }
    ফেরত _URC_NO_REASON;
}

স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    ধ্রুবক পূর্ণসংখ্যা max = 100;
    শূন্য* বাফার[max];

    কাঠামো backtrace_state state = {বাফার, বাফার + max};
    _Unwind_Backtrace(unwind_callback, &state);

    পূর্ণসংখ্যা গুনতি = state.current - বাফার;

    জন্য (পূর্ণসংখ্যা idx = 0; idx < গুনতি; ++idx) {
        ধ্রুবক শূন্য * addr = বাফার[idx];
        ধ্রুবক অক্ষর * প্রতীক = "";

        Dl_info info;
        যদি (dladdr(addr, &info) && info.dli_sname) {
            প্রতীক = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, প্রতীক);
    }
}
#নতুবা_যদি defined(__linux__) && defined(__GLIBC__)
#অন্তর্ভাব <execinfo.h>
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    শূন্য * trace[100];
    পূর্ণসংখ্যা nptrs = backtrace(trace, আকার(trace)/আকার(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#নতুবা_যদি defined(__APPLE__)
#অন্তর্ভাব <execinfo.h>
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    শূন্য * trace[100];
    পূর্ণসংখ্যা nptrs = backtrace(trace, আকার(trace)/আকার(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#অন্যথা
স্থির শূন্য ggml_print_backtrace_symbols(শূন্য) {
    // platform not supported
}
#endif

শূন্য ggml_print_backtrace(শূন্য) {
    ধ্রুবক অক্ষর * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    যদি (GGML_NO_BACKTRACE) {
        ফেরত;
    }
#যদি defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ধ্রুবক অক্ষর * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    যদি (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        ফেরত;
    }
#endif
#যদি defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t আকার_2 = 0;
    অক্ষর * সারি = ফাঁকা;
    ssize_t দৈর্ঘ্য = 0;
    যখন ((দৈর্ঘ্য = getline(&সারি, &আকার_2, f)) > 0) {
        যদি (!strncmp(সারি, "TracerPid:", আকার("TracerPid:") - 1) &&
            (দৈর্ঘ্য != আকার("TracerPid:\t0\n") - 1 || সারি[দৈর্ঘ্য - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            মুক্ত(সারি);
            fclose(f);
            ফেরত;
        }
    }
    মুক্ত(সারি);
    fclose(f);
    পূর্ণসংখ্যা lock[2] = { -1, -1 };
    (শূন্য) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ধ্রুবক পূর্ণসংখ্যা parent_pid = getpid();
    ধ্রুবক পূর্ণসংখ্যা child_pid = fork();
    যদি (child_pid < 0) { // error
#যদি defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        ফেরত;
    } অন্যথা যদি (child_pid == 0) { // child
        অক্ষর attach[32];
        snprintf(attach, আকার(attach), "attach %d", parent_pid);
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
            (অক্ষর *) ফাঁকা);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[আকার("attach ") - 1],
            (অক্ষর *) ফাঁকা);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } অন্যথা { // parent
#যদি defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, ফাঁকা, 0);
    }
}
#অন্যথা
শূন্য ggml_print_backtrace(শূন্য) {
    // platform not supported
}
#endif

স্থির ggml_abort_callback_t g_abort_callback = ফাঁকা;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    ফেরত ret_val;
}

শূন্য ggml_abort(ধ্রুবক অক্ষর * file, পূর্ণসংখ্যা সারি, ধ্রুবক অক্ষর * fmt, ...) {
    fflush(stdout);

    অক্ষর message[2048];
    পূর্ণসংখ্যা offset = snprintf(message, আকার(message), "%s:%d: ", file, সারি);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, আকার(message) - offset, fmt, args);
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

কাঠামো ggml_logger_state {
    ggml_log_callback log_callback;
    শূন্য * log_callback_user_তথ্য;
};
স্থির কাঠামো ggml_logger_state g_logger_state = {ggml_log_callback_default, ফাঁকা};

স্থির শূন্য ggml_log_internal_v(গণনা ggml_log_level level, ধ্রুবক অক্ষর * format, va_list args) {
    যদি (format == ফাঁকা) {
        ফেরত;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    অক্ষর বাফার[128];
    পূর্ণসংখ্যা len = vsnprintf(বাফার, 128, format, args);
    যদি (len < 128) {
        g_logger_state.log_callback(level, বাফার, g_logger_state.log_callback_user_তথ্য);
    } অন্যথা {
        অক্ষর * buffer2 = (অক্ষর *) calloc(len + 1, আকার(অক্ষর));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_তথ্য);
        মুক্ত(buffer2);
    }
    va_end(args_copy);
}

শূন্য ggml_log_internal(গণনা ggml_log_level level, ধ্রুবক অক্ষর * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

শূন্য ggml_log_callback_default(গণনা ggml_log_level level, ধ্রুবক অক্ষর * text, শূন্য * user_তথ্য) {
    (শূন্য) level;
    (শূন্য) user_তথ্য;
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

শূন্য * ggml_aligned_malloc(size_t আকার_2) {
#যদি defined(__s390x__)
    ধ্রুবক পূর্ণসংখ্যা alignment = 256;
#অন্যথা
    ধ্রুবক পূর্ণসংখ্যা alignment = 64;
#endif

#যদি defined(_MSC_VER) || defined(__MINGW32__)
    ফেরত _aligned_malloc(আকার_2, alignment);
#অন্যথা
    যদি (আকার_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        ফেরত ফাঁকা;
    }
    শূন্য * aligned_memory = ফাঁকা;
  #ifdef GGML_USE_CPU_HBM
    পূর্ণসংখ্যা ফলাফল = hbw_posix_memalign(&aligned_memory, alignment, আকার_2);
  #নতুবা_যদি TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, আকার_2, VM_FLAGS_ANYWHERE);
    পূর্ণসংখ্যা ফলাফল = EFAULT;
    নির্বাচন (alloc_status) {
        ক্ষেত্র KERN_SUCCESS:
            ফলাফল = 0;
            বিরতি;
        ক্ষেত্র KERN_INVALID_ADDRESS:
            ফলাফল = EINVAL;
            বিরতি;
        ক্ষেত্র KERN_NO_SPACE:
            ফলাফল = ENOMEM;
            বিরতি;
        পূর্বনির্ধারিত:
            ফলাফল = EFAULT;
            বিরতি;
    }
  #অন্যথা
    পূর্ণসংখ্যা ফলাফল = posix_memalign(&aligned_memory, alignment, আকার_2);
  #endif
    যদি (ফলাফল != 0) {
        // Handle allocation failure
        ধ্রুবক অক্ষর *error_desc = "unknown allocation error";
        নির্বাচন (ফলাফল) {
            ক্ষেত্র EINVAL:
                error_desc = "invalid alignment value";
                বিরতি;
            ক্ষেত্র ENOMEM:
                error_desc = "insufficient memory";
                বিরতি;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, আকার_2/(1024.0*1024.0));
        ফেরত ফাঁকা;
    }
    ফেরত aligned_memory;
#endif
}

শূন্য ggml_aligned_free(শূন্য * ptr, size_t আকার_2) {
    GGML_UNUSED(আকার_2);
#যদি defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#নতুবা_যদি GGML_USE_CPU_HBM
    যদি (ptr != ফাঁকা) {
        hbw_free(ptr);
    }
#নতুবা_যদি TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    যদি (ptr != ফাঁকা) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, আকার_2);
    }
#অন্যথা
    মুক্ত(ptr);
#endif
}


অন্তঃরেখ স্থির শূন্য * ggml_malloc(size_t আকার_2) {
    যদি (আকার_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        ফেরত ফাঁকা;
    }
    শূন্য * ফলাফল = malloc(আকার_2);
    যদি (ফলাফল == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, আকার_2/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ফেরত ফলাফল;
}

// calloc
অন্তঃরেখ স্থির শূন্য * ggml_calloc(size_t অঙ্ক, size_t আকার_2) {
    যদি (অঙ্ক == 0 || আকার_2 == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        ফেরত ফাঁকা;
    }
    শূন্য * ফলাফল = calloc(অঙ্ক, আকার_2);
    যদি (ফলাফল == ফাঁকা) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, আকার_2/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    ফেরত ফলাফল;
}

#define GGML_MALLOC(আকার_2)      ggml_malloc(আকার_2)
#define GGML_CALLOC(অঙ্ক, আকার_2) ggml_calloc(অঙ্ক, আকার_2)

#define GGML_FREE(ptr) মুক্ত(ptr)

ধ্রুবক অক্ষর * ggml_status_to_string(গণনা ggml_status status) {
    নির্বাচন (status) {
        ক্ষেত্র GGML_STATUS_ALLOC_FAILED: ফেরত "GGML status: error (failed to allocate memory)";
        ক্ষেত্র GGML_STATUS_FAILED:       ফেরত "GGML status: error (operation failed)";
        ক্ষেত্র GGML_STATUS_SUCCESS:      ফেরত "GGML status: success";
        ক্ষেত্র GGML_STATUS_ABORTED:      ফেরত "GGML status: warning (operation aborted)";
    }

    ফেরত "GGML status: unknown";
}

দশমিক ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    ফেরত GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(দশমিক x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    ফেরত GGML_FP32_TO_FP16(x);
}

দশমিক ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    ফেরত GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(দশমিক x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    ফেরত GGML_FP32_TO_BF16(x);
}

শূন্য ggml_fp16_to_fp32_row(ধ্রুবক ggml_fp16_t * x, দশমিক * y, int64_t n) {
    জন্য (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

শূন্য ggml_fp32_to_fp16_row(ধ্রুবক দশমিক * x, ggml_fp16_t * y, int64_t n) {
    পূর্ণসংখ্যা i = 0;
    জন্য (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

শূন্য ggml_bf16_to_fp32_row(ধ্রুবক ggml_bf16_t * x, দশমিক * y, int64_t n) {
    পূর্ণসংখ্যা i = 0;
    জন্য (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

শূন্য ggml_fp32_to_bf16_row_ref(ধ্রুবক দশমিক * x, ggml_bf16_t * y, int64_t n) {
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

শূন্য ggml_fp32_to_bf16_row(ধ্রুবক দশমিক * x, ggml_bf16_t * y, int64_t n) {
  পূর্ণসংখ্যা i = 0;
#যদি defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  জন্য (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    জন্য (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    ফেরত memcmp(guid_a, guid_b, আকার(ggml_guid)) == 0;
}

ধ্রুবক অক্ষর * ggml_version(শূন্য) {
    ফেরত GGML_VERSION;
}

ধ্রুবক অক্ষর * ggml_commit(শূন্য) {
    ফেরত GGML_COMMIT;
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
    Query_Performance_গণক(&t);
    timer_start = t.QuadPart;

    ফেরত TRUE;
}
শূন্য ggml_time_init(শূন্য) {
    স্থির INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, ফাঁকা, ফাঁকা);
}
int64_t ggml_time_ms(শূন্য) {
    LARGE_INTEGER t;
    Query_Performance_গণক(&t);
    ফেরত ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(শূন্য) {
    LARGE_INTEGER t;
    Query_Performance_গণক(&t);
    ফেরত ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#অন্যথা
শূন্য ggml_time_init(শূন্য) {}
int64_t ggml_time_ms(শূন্য) {
    কাঠামো timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ফেরত (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(শূন্য) {
    কাঠামো timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ফেরত (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(শূন্য) {
    ফেরত clock();
}

int64_t ggml_cycles_per_ms(শূন্য) {
    ফেরত CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
স্থির wchar_t * ggml_mbstowcs(ধ্রুবক অক্ষর * mbs) {
    পূর্ণসংখ্যা wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, ফাঁকা, 0);
    যদি (!wlen) {
        errno = EINVAL;
        ফেরত ফাঁকা;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * আকার(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    যদি (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        ফেরত ফাঁকা;
    }

    ফেরত wbuf;
}
#endif

FILE * ggml_fopen(ধ্রুবক অক্ষর * fname, ধ্রুবক অক্ষর * mode) {
#ifdef _WIN32
    FILE * file = ফাঁকা;

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

    ফেরত file;
#অন্যথা
    ফেরত fopen(fname, mode);
#endif

}

স্থির ধ্রুবক কাঠামো ggml_type_traits type_traits[GGML_TYPE_গুনতি] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_আকার                = 1,
        .type_আকার                = আকার(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_আকার                = 1,
        .type_আকার                = আকার(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_আকার                = 1,
        .type_আকার                = আকার(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_আকার                = 1,
        .type_আকার                = আকার(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_আকার                = 1,
        .type_আকার                = আকার(দ্বিগুণ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_আকার                = 1,
        .type_আকার                = আকার(দশমিক),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_আকার                = 1,
        .type_আকার                = আকার(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_আকার                = QK1_0,
        .type_আকার                = আকার(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_আকার                = QK2_0,
        .type_আকার                = আকার(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_আকার                = QK4_0,
        .type_আকার                = আকার(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_আকার                = QK4_1,
        .type_আকার                = আকার(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_আকার                = QK5_0,
        .type_আকার                = আকার(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_আকার                = QK5_1,
        .type_আকার                = আকার(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_আকার                = QK8_0,
        .type_আকার                = আকার(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_আকার                = QK8_1,
        .type_আকার                = আকার(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_আকার                = QK_MXFP4,
        .type_আকার                = আকার(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_আকার                = QK_NVFP4,
        .type_আকার                = আকার(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = ফাঁকা,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = ফাঁকা,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = ফাঁকা,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = ফাঁকা,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_আকার                = QK4_NL,
        .type_আকার                = আকার(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_আকার                = 1,
        .type_আকার                = আকার(ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_bf16_row_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_আকার                = QK_K,
        .type_আকার                = আকার(block_tq2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_আকার                = 0,
        .type_আকার                = 0,
        .is_quantized             = false,
    },
};

ধ্রুবক কাঠামো ggml_type_traits * ggml_get_type_traits(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত &type_traits[type];
}

//
// ggml object
//

কাঠামো ggml_object {
    size_t offs;
    size_t আকার_2;

    কাঠামো ggml_object * পরবর্তী;

    গণনা ggml_object_type type;

    অক্ষর padding[4];
};

স্থির ধ্রুবক size_t GGML_OBJECT_আকার = আকার(কাঠামো ggml_object);

//
// ggml context
//

কাঠামো ggml_context {
    size_t mem_আকার;
    শূন্য * mem_বাফার;
    bool   mem_বাফার_owned;
    bool   no_alloc;

    পূর্ণসংখ্যা    n_objects;

    কাঠামো ggml_object * objects_begin;
    কাঠামো ggml_object * objects_end;
};

//
// data types
//

স্থির ধ্রুবক অক্ষর * GGML_OP_NAME[GGML_OP_গুনতি] = {
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

স্থির ধ্রুবক অক্ষর * GGML_OP_প্রতীক[GGML_OP_গুনতি] = {
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

স্থির ধ্রুবক অক্ষর * GGML_UNARY_OP_NAME[GGML_UNARY_OP_গুনতি] = {
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

স্থির ধ্রুবক অক্ষর * GGML_GLU_OP_NAME[GGML_GLU_OP_গুনতি] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_গুনতি == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(আকার(কাঠামো ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(আকার(কাঠামো ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_print_object(ধ্রুবক কাঠামো ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->আকার_2, (ধ্রুবক শূন্য *) obj->পরবর্তী);
}

শূন্য ggml_print_objects(ধ্রুবক কাঠামো ggml_context * ctx) {
    কাঠামো ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ধ্রুবক শূন্য *) ctx);

    যখন (obj != ফাঁকা) {
        ggml_print_object(obj);
        obj = obj->পরবর্তী;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] <= 0) {
            ফেরত 0;
        }
    }

    size_t nbytes;
    ধ্রুবক size_t blck_আকার = ggml_blck_আকার(tensor->type);
    যদি (blck_আকার == 1) {
        nbytes = ggml_type_আকার(tensor->type);
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    অন্যথা {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_আকার;
        জন্য (পূর্ণসংখ্যা i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    ফেরত nbytes;
}

size_t ggml_nbytes_pad(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_আকার(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত type_traits[type].blck_আকার;
}

size_t ggml_type_আকার(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত type_traits[type].type_আকার;
}

size_t ggml_row_আকার(গণনা ggml_type type, int64_t ne) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    প্রতিজ্ঞা(ne % ggml_blck_আকার(type) == 0);
    ফেরত ggml_type_আকার(type)*ne/ggml_blck_আকার(type);
}

দ্বিগুণ ggml_type_sizef(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত ((দ্বিগুণ)(type_traits[type].type_আকার))/type_traits[type].blck_আকার;
}

ধ্রুবক অক্ষর * ggml_type_name(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত type_traits[type].type_name;
}

bool ggml_is_quantized(গণনা ggml_type type) {
    প্রতিজ্ঞা(type >= 0);
    প্রতিজ্ঞা(type < GGML_TYPE_গুনতি);
    ফেরত type_traits[type].is_quantized;
}

ধ্রুবক অক্ষর * ggml_op_name(গণনা ggml_op op) {
    ফেরত GGML_OP_NAME[op];
}

ধ্রুবক অক্ষর * ggml_op_প্রতীক(গণনা ggml_op op) {
    ফেরত GGML_OP_প্রতীক[op];
}

ধ্রুবক অক্ষর * ggml_unary_op_name(গণনা ggml_unary_op op) {
    ফেরত GGML_UNARY_OP_NAME[op];
}

ধ্রুবক অক্ষর * ggml_glu_op_name(গণনা ggml_glu_op op) {
    ফেরত GGML_GLU_OP_NAME[op];
}

ধ্রুবক অক্ষর * ggml_op_desc(ধ্রুবক কাঠামো ggml_tensor * t) {
    যদি (t->op == GGML_OP_UNARY) {
        গণনা ggml_unary_op uop = ggml_get_unary_op(t);
        ফেরত ggml_unary_op_name(uop);
    }
    যদি (t->op == GGML_OP_GLU) {
        গণনা ggml_glu_op gop = ggml_get_glu_op(t);
        ফেরত ggml_glu_op_name(gop);
    }
    ফেরত ggml_op_name(t->op);
}

size_t ggml_element_আকার(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_type_আকার(tensor->type);
}

bool ggml_is_scalar(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত tensor->ne[3] == 1;
}

পূর্ণসংখ্যা ggml_n_dims(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        যদি (tensor->ne[i] > 1) {
            ফেরত i + 1;
        }
    }
    ফেরত 1;
}

গণনা ggml_type ggml_ftype_to_ggml_type(গণনা ggml_ftype ftype) {
    গণনা ggml_type wtype = GGML_TYPE_গুনতি;

    নির্বাচন (ftype) {
        ক্ষেত্র GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    বিরতি;
        ক্ষেত্র GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_গুনতি; বিরতি;
        ক্ষেত্র GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_গুনতি; বিরতি;
    }

    GGML_ASSERT(wtype != GGML_TYPE_গুনতি);

    ফেরত wtype;
}

size_t ggml_tensor_overhead(শূন্য) {
    ফেরত GGML_OBJECT_আকার + GGML_TENSOR_আকার;
}

bool ggml_is_transposed(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত tensor->nb[0] > tensor->nb[1];
}

স্থির bool ggml_is_contiguous_m_n(ধ্রুবক কাঠামো ggml_tensor * tensor, পূর্ণসংখ্যা m, পূর্ণসংখ্যা n) {
    size_t পরবর্তী_nb = ggml_type_আকার(tensor->type);
    যদি (tensor->ne[0] != ggml_blck_আকার(tensor->type) && tensor->nb[0] != পরবর্তী_nb) {
        ফেরত false;
    }
    পরবর্তী_nb *= tensor->ne[0]/ggml_blck_আকার(tensor->type);
    জন্য (পূর্ণসংখ্যা i = 1; i < n; i++) {
        যদি (i > m) {
            যদি (tensor->ne[i] != 1 && tensor->nb[i] != পরবর্তী_nb) {
                ফেরত false;
            }
            পরবর্তী_nb *= tensor->ne[i];
        } অন্যথা {
            // this dimension does not need to be contiguous
            পরবর্তী_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    ফেরত true;
}

bool ggml_is_contiguous(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_আকার(tensor->type)/ggml_blck_আকার(tensor->type);
}

bool ggml_is_permuted(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_আকার(tensor->type);
}

bool ggml_is_contiguous_rows(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত
        tensor->ne[0] == ggml_blck_আকার(tensor->type) ||
        tensor->nb[0] == ggml_type_আকার(tensor->type);
}

স্থির অন্তঃরেখ bool ggml_is_padded_1d(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত
        tensor->nb[0] == ggml_type_আকার(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            ফেরত true;
        }
    }
    ফেরত false;
}

bool ggml_are_same_shape(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

স্থির অন্তঃরেখ bool ggml_can_repeat_rows(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

কাঠামো ggml_context * ggml_init(কাঠামো ggml_init_params params) {
    স্থির bool is_first_call = true;

    ggml_critical_section_start();

    যদি (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    কাঠামো ggml_context * ctx = GGML_MALLOC(আকার(কাঠামো ggml_context));

    // allow to call ggml_init with 0 size
    যদি (params.mem_আকার == 0) {
        params.mem_আকার = GGML_MEM_ALIGN;
    }

    ধ্রুবক size_t mem_আকার = params.mem_বাফার ? params.mem_আকার : GGML_PAD(params.mem_আকার, GGML_MEM_ALIGN);

    *ctx = (কাঠামো ggml_context) {
        /*.mem_size           =*/ mem_আকার,
        /*.mem_buffer         =*/ params.mem_বাফার ? params.mem_বাফার : ggml_aligned_malloc(mem_আকার),
        /*.mem_buffer_owned   =*/ params.mem_বাফার ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ফাঁকা,
        /*.objects_end        =*/ ফাঁকা,
    };

    GGML_ASSERT(ctx->mem_বাফার != ফাঁকা);

    GGML_ASSERT_ALIGNED(ctx->mem_বাফার);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ফেরত ctx;
}

শূন্য ggml_reset(কাঠামো ggml_context * ctx) {
    যদি (ctx == ফাঁকা) {
        ফেরত;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ফাঁকা;
    ctx->objects_end   = ফাঁকা;
}

শূন্য ggml_free(কাঠামো ggml_context * ctx) {
    যদি (ctx == ফাঁকা) {
        ফেরত;
    }

    যদি (ctx->mem_বাফার_owned) {
        ggml_aligned_free(ctx->mem_বাফার, ctx->mem_আকার);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ধ্রুবক কাঠামো ggml_context * ctx) {
    ফেরত ctx->objects_end == ফাঁকা ? 0 : ctx->objects_end->offs + ctx->objects_end->আকার_2;
}

bool ggml_get_no_alloc(কাঠামো ggml_context * ctx) {
    ফেরত ctx->no_alloc;
}

শূন্য ggml_set_no_alloc(কাঠামো ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

শূন্য * ggml_get_mem_বাফার(ধ্রুবক কাঠামো ggml_context * ctx) {
    ফেরত ctx->mem_বাফার;
}

size_t ggml_get_mem_আকার(ধ্রুবক কাঠামো ggml_context * ctx) {
    ফেরত ctx->mem_আকার;
}

size_t ggml_get_max_tensor_আকার(ধ্রুবক কাঠামো ggml_context * ctx) {
    size_t max_আকার = 0;

    জন্য (কাঠামো ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != ফাঁকা; tensor = ggml_get_পরবর্তী_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_আকার = MAX(max_আকার, bytes);
    }

    ফেরত max_আকার;
}

////////////////////////////////////////////////////////////////////////////////

স্থির কাঠামো ggml_object * ggml_new_object(কাঠামো ggml_context * ctx, গণনা ggml_object_type type, size_t আকার_2) {
    // always insert objects at the end of the context's memory pool
    কাঠামো ggml_object * obj_cur = ctx->objects_end;

    ধ্রুবক size_t cur_offs = obj_cur == ফাঁকা ? 0 : obj_cur->offs;
    ধ্রুবক size_t cur_আকার = obj_cur == ফাঁকা ? 0 : obj_cur->আকার_2;
    ধ্রুবক size_t cur_end  = cur_offs + cur_আকার;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(আকার_2 <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t আকার_needed = GGML_PAD(আকার_2, GGML_MEM_ALIGN);

    অক্ষর * ধ্রুবক mem_বাফার = ctx->mem_বাফার;
    কাঠামো ggml_object * ধ্রুবক obj_new = (কাঠামো ggml_object *)(mem_বাফার + cur_end);

    // integer overflow checks
    যদি (cur_end > SIZE_MAX - আকার_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, আকার_needed);
        ফেরত ফাঁকা;
    }
    যদি (cur_end + আকার_needed > SIZE_MAX - GGML_OBJECT_আকার) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, আকার_needed, (size_t) GGML_OBJECT_আকার);
        ফেরত ফাঁকা;
    }

    যদি (cur_end + আকার_needed + GGML_OBJECT_আকার > ctx->mem_আকার) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + আকার_needed + GGML_OBJECT_আকার, ctx->mem_আকার);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        ফেরত ফাঁকা;
    }

    *obj_new = (কাঠামো ggml_object) {
        .offs = cur_end + GGML_OBJECT_আকার,
        .আকার_2 = আকার_needed,
        .পরবর্তী = ফাঁকা,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_বাফার + obj_new->offs);

    যদি (obj_cur != ফাঁকা) {
        obj_cur->পরবর্তী = obj_new;
    } অন্যথা {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    ফেরত obj_new;
}

স্থির কাঠামো ggml_tensor * ggml_new_tensor_impl(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type      type,
        পূর্ণসংখ্যা                   n_dims,
        ধ্রুবক int64_t       * ne,
        কাঠামো ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_গুনতি);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    যদি (view_src != ফাঁকা && view_src->view_src != ফাঁকা) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    জন্য (পূর্ণসংখ্যা i = 1; i < n_dims; i++) {
        যদি (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t তথ্য_আকার = ggml_row_আকার(type, ne[0]);
    জন্য (পূর্ণসংখ্যা i = 1; i < n_dims; i++) {
        তথ্য_আকার *= ne[i];
    }

    GGML_ASSERT(view_src == ফাঁকা || তথ্য_আকার == 0 || তথ্য_আকার + view_offs <= ggml_nbytes(view_src));

    শূন্য * তথ্য = view_src != ফাঁকা ? view_src->তথ্য : ফাঁকা;
    যদি (তথ্য != ফাঁকা) {
        তথ্য = (অক্ষর *) তথ্য + view_offs;
    }

    size_t obj_alloc_আকার = 0;

    যদি (view_src == ফাঁকা && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_আকার = তথ্য_আকার;
    }

    GGML_ASSERT(GGML_TENSOR_আকার <= SIZE_MAX - obj_alloc_আকার);

    কাঠামো ggml_object * ধ্রুবক obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_আকার + obj_alloc_আকার);
    GGML_ASSERT(obj_new);

    কাঠামো ggml_tensor * ধ্রুবক ফলাফল = (কাঠামো ggml_tensor *)((অক্ষর *)ctx->mem_বাফার + obj_new->offs);

    *ফলাফল = (কাঠামো ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ফাঁকা,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ফাঁকা },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_আকার > 0 ? (শূন্য *)(ফলাফল + 1) : তথ্য,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ফাঁকা,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    জন্য (পূর্ণসংখ্যা i = 0; i < n_dims; i++) {
        ফলাফল->ne[i] = ne[i];
    }

    ফলাফল->nb[0] = ggml_type_আকার(type);
    ফলাফল->nb[1] = ফলাফল->nb[0]*(ফলাফল->ne[0]/ggml_blck_আকার(type));
    জন্য (পূর্ণসংখ্যা i = 2; i < GGML_MAX_DIMS; i++) {
        ফলাফল->nb[i] = ফলাফল->nb[i - 1]*ফলাফল->ne[i - 1];
    }

    ctx->n_objects++;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_new_tensor(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type      type,
        পূর্ণসংখ্যা                   n_dims,
        ধ্রুবক int64_t       * ne) {
    ফেরত ggml_new_tensor_impl(ctx, type, n_dims, ne, ফাঁকা, 0);
}

কাঠামো ggml_tensor * ggml_new_tensor_1d(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0) {
    ফেরত ggml_new_tensor(ctx, type, 1, &ne0);
}

কাঠামো ggml_tensor * ggml_new_tensor_2d(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ধ্রুবক int64_t ne[2] = { ne0, ne1 };
    ফেরত ggml_new_tensor(ctx, type, 2, ne);
}

কাঠামো ggml_tensor * ggml_new_tensor_3d(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ধ্রুবক int64_t ne[3] = { ne0, ne1, ne2 };
    ফেরত ggml_new_tensor(ctx, type, 3, ne);
}

কাঠামো ggml_tensor * ggml_new_tensor_4d(
        কাঠামো ggml_context * ctx,
        গণনা   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ধ্রুবক int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ফেরত ggml_new_tensor(ctx, type, 4, ne);
}

শূন্য * ggml_new_বাফার(কাঠামো ggml_context * ctx, size_t nbytes) {
    কাঠামো ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_বাফার, nbytes);

    ফেরত (uint8_t *)ctx->mem_বাফার + obj->offs;
}

কাঠামো ggml_tensor * ggml_dup_tensor(কাঠামো ggml_context * ctx, ধ্রুবক কাঠামো ggml_tensor * src) {
    ফেরত ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

শূন্য ggml_unravel_সূচক(ধ্রুবক কাঠামো ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ধ্রুবক int64_t ne2 = tensor->ne[2];
    ধ্রুবক int64_t ne1 = tensor->ne[1];
    ধ্রুবক int64_t ne0 = tensor->ne[0];

    ধ্রুবক int64_t i3_ = (i/(ne2*ne1*ne0));
    ধ্রুবক int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ধ্রুবক int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ধ্রুবক int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

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

শূন্য * ggml_get_তথ্য(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত tensor->তথ্য;
}

দশমিক * ggml_get_তথ্য_f32(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    প্রতিজ্ঞা(tensor->type == GGML_TYPE_F32);
    ফেরত (দশমিক *)(tensor->তথ্য);
}

গণনা ggml_unary_op ggml_get_unary_op(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    ফেরত (গণনা ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

গণনা ggml_glu_op ggml_get_glu_op(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    ফেরত (গণনা ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ধ্রুবক অক্ষর * ggml_get_name(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত tensor->name;
}

কাঠামো ggml_tensor * ggml_set_name(কাঠামো ggml_tensor * tensor, ধ্রুবক অক্ষর * name) {
    size_t i;
    জন্য (i = 0; i < আকার(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    ফেরত tensor;
}

কাঠামো ggml_tensor * ggml_format_name(কাঠামো ggml_tensor * tensor, ধ্রুবক অক্ষর * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, আকার(tensor->name), fmt, args);
    va_end(args);
    ফেরত tensor;
}

কাঠামো ggml_tensor * ggml_view_tensor(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * src) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(ফলাফল, "%s (view)", src->name);

    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
        ফলাফল->nb[i] = src->nb[i];
    }

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_get_first_tensor(ধ্রুবক কাঠামো ggml_context * ctx) {
    কাঠামো ggml_object * obj = ctx->objects_begin;

    অক্ষর * ধ্রুবক mem_বাফার = ctx->mem_বাফার;

    যখন (obj != ফাঁকা) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ফেরত (কাঠামো ggml_tensor *)(mem_বাফার + obj->offs);
        }

        obj = obj->পরবর্তী;
    }

    ফেরত ফাঁকা;
}

কাঠামো ggml_tensor * ggml_get_পরবর্তী_tensor(ধ্রুবক কাঠামো ggml_context * ctx, কাঠামো ggml_tensor * tensor) {
    কাঠামো ggml_object * obj = (কাঠামো ggml_object *) ((অক্ষর *)tensor - GGML_OBJECT_আকার);
    obj = obj->পরবর্তী;

    অক্ষর * ধ্রুবক mem_বাফার = ctx->mem_বাফার;

    যখন (obj != ফাঁকা) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ফেরত (কাঠামো ggml_tensor *)(mem_বাফার + obj->offs);
        }

        obj = obj->পরবর্তী;
    }

    ফেরত ফাঁকা;
}

কাঠামো ggml_tensor * ggml_get_tensor(কাঠামো ggml_context * ctx, ধ্রুবক অক্ষর * name) {
    কাঠামো ggml_object * obj = ctx->objects_begin;

    অক্ষর * ধ্রুবক mem_বাফার = ctx->mem_বাফার;

    যখন (obj != ফাঁকা) {
        যদি (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            কাঠামো ggml_tensor * cur = (কাঠামো ggml_tensor *)(mem_বাফার + obj->offs);
            যদি (strcmp(cur->name, name) == 0) {
                ফেরত cur;
            }
        }

        obj = obj->পরবর্তী;
    }

    ফেরত ফাঁকা;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

স্থির কাঠামো ggml_tensor * ggml_dup_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_DUP;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_dup(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_dup_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_dup_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_dup_impl(ctx, a, true);
}

// ggml_add

স্থির কাঠামো ggml_tensor * ggml_add_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_ADD;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_add(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_add_impl(ctx, a, b, false);
}

কাঠামো ggml_tensor * ggml_add_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

স্থির কাঠামো ggml_tensor * ggml_add_cast_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        গণনা   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    ফলাফল->op     = GGML_OP_ADD;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_add_cast(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        গণনা   ggml_type      type) {
    ফেরত ggml_add_cast_impl(ctx, a, b, type);
}

কাঠামো ggml_tensor * ggml_add_id(
            কাঠামো ggml_context * ctx,
            কাঠামো ggml_tensor  * a,
            কাঠামো ggml_tensor  * b,
            কাঠামো ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_ADD_ID;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;
    ফলাফল->src[2] = ids;

    ফেরত ফলাফল;
}

// ggml_add1

স্থির কাঠামো ggml_tensor * ggml_add1_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_ADD1;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_add1(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_add1_impl(ctx, a, b, false);
}

কাঠামো ggml_tensor * ggml_add1_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

স্থির কাঠামো ggml_tensor * ggml_acc_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_ACC;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_acc(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ফেরত ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

কাঠামো ggml_tensor * ggml_acc_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ফেরত ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

স্থির কাঠামো ggml_tensor * ggml_sub_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SUB;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_sub(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_sub_impl(ctx, a, b, false);
}

কাঠামো ggml_tensor * ggml_sub_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

স্থির কাঠামো ggml_tensor * ggml_mul_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_MUL;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_mul(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_mul_impl(ctx, a, b, false);
}

কাঠামো ggml_tensor * ggml_mul_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

স্থির কাঠামো ggml_tensor * ggml_div_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_DIV;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_div(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_div_impl(ctx, a, b, false);
}

কাঠামো ggml_tensor * ggml_div_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

স্থির কাঠামো ggml_tensor * ggml_sqr_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SQR;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_sqr(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sqr_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_sqr_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

স্থির কাঠামো ggml_tensor * ggml_sqrt_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SQRT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_sqrt(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sqrt_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_sqrt_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

স্থির কাঠামো ggml_tensor * ggml_log_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_LOG;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_log(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_log_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_log_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_log_impl(ctx, a, true);
}

কাঠামো ggml_tensor * ggml_expm1(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

কাঠামো ggml_tensor * ggml_expm1_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

কাঠামো ggml_tensor * ggml_softplus(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

কাঠামো ggml_tensor * ggml_softplus_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

স্থির কাঠামো ggml_tensor * ggml_sin_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SIN;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_sin(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sin_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_sin_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_sin_impl(ctx, a, true);
}

// ggml_cos

স্থির কাঠামো ggml_tensor * ggml_cos_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_COS;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_cos(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_cos_impl(ctx, a, false);
}

কাঠামো ggml_tensor * ggml_cos_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_cos_impl(ctx, a, true);
}

// ggml_sum

কাঠামো ggml_tensor * ggml_যোগ(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, a->type, 1);

    ফলাফল->op     = GGML_OP_যোগ;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_sum_rows

কাঠামো ggml_tensor * ggml_যোগ_rows(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    জন্য (পূর্ণসংখ্যা i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ফলাফল->op     = GGML_OP_যোগ_ROWS;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_cumsum

কাঠামো ggml_tensor * ggml_cumsum(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_CUMSUM;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_mean

কাঠামো ggml_tensor * ggml_mean(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_MEAN;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_argmax

কাঠামো ggml_tensor * ggml_argmax(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    ফলাফল->op     = GGML_OP_ARGMAX;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_count_equal

কাঠামো ggml_tensor * ggml_গুনতি_equal(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    ফলাফল->op     = GGML_OP_গুনতি_EQUAL;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_repeat

কাঠামো ggml_tensor * ggml_repeat(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ফলাফল->op     = GGML_OP_REPEAT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_repeat_4d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ধ্রুবক bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ফলাফল->op     = GGML_OP_REPEAT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_repeat_back

কাঠামো ggml_tensor * ggml_repeat_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ফলাফল->op     = GGML_OP_REPEAT_BACK;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_concat

কাঠামো ggml_tensor * ggml_concat(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    কাঠামো ggml_tensor  * b,
    পূর্ণসংখ্যা                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    জন্য (পূর্ণসংখ্যা d = 0; d < GGML_MAX_DIMS; ++d) {
        যদি (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            চালু;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(ফলাফল, 0, dim);

    ফলাফল->op     = GGML_OP_CONCAT;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_abs

কাঠামো ggml_tensor * ggml_abs(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

কাঠামো ggml_tensor * ggml_abs_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

কাঠামো ggml_tensor * ggml_sgn(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

কাঠামো ggml_tensor * ggml_sgn_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

কাঠামো ggml_tensor * ggml_neg(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

কাঠামো ggml_tensor * ggml_neg_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

কাঠামো ggml_tensor * ggml_step(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

কাঠামো ggml_tensor * ggml_step_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

কাঠামো ggml_tensor * ggml_tanh(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

কাঠামো ggml_tensor * ggml_tanh_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

কাঠামো ggml_tensor * ggml_elu(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

কাঠামো ggml_tensor * ggml_elu_inplace(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

কাঠামো ggml_tensor * ggml_relu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

কাঠামো ggml_tensor * ggml_relu_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

কাঠামো ggml_tensor * ggml_leaky_relu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 negative_slope,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(ফলাফল, &negative_slope, আকার(negative_slope));

    ফলাফল->op     = GGML_OP_LEAKY_RELU;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_sigmoid

কাঠামো ggml_tensor * ggml_sigmoid(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

কাঠামো ggml_tensor * ggml_sigmoid_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

কাঠামো ggml_tensor * ggml_gelu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

কাঠামো ggml_tensor * ggml_gelu_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

কাঠামো ggml_tensor * ggml_gelu_erf(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

কাঠামো ggml_tensor * ggml_gelu_erf_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

কাঠামো ggml_tensor * ggml_gelu_quick(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

কাঠামো ggml_tensor * ggml_gelu_quick_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

কাঠামো ggml_tensor * ggml_silu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

কাঠামো ggml_tensor * ggml_silu_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

কাঠামো ggml_tensor * ggml_xielu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক alpha_n,
        দশমিক alpha_p,
        দশমিক beta,
        দশমিক eps) {
    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(ফলাফল, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(ফলাফল, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(ফলাফল, 3, beta);
    ggml_set_op_params_f32(ফলাফল, 4, eps);

    ফলাফল->op     = GGML_OP_UNARY;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_silu_back

কাঠামো ggml_tensor * ggml_silu_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SILU_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml hardswish

কাঠামো ggml_tensor * ggml_hardswish(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

কাঠামো ggml_tensor * ggml_hardsigmoid(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

কাঠামো ggml_tensor * ggml_exp(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

কাঠামো ggml_tensor * ggml_exp_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

স্থির কাঠামো ggml_tensor * ggml_glu_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        গণনা ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    যদি (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; জন্য (পূর্ণসংখ্যা i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ফাঁকা, 0);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) op);
    ggml_set_op_params_i32(ফলাফল, 1, (int32_t) swapped);

    ফলাফল->op     = GGML_OP_GLU;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_floor

কাঠামো ggml_tensor * ggml_floor(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

কাঠামো ggml_tensor * ggml_floor_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

কাঠামো ggml_tensor * ggml_ceil(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

কাঠামো ggml_tensor * ggml_ceil_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

কাঠামো ggml_tensor * ggml_round(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

কাঠামো ggml_tensor * ggml_round_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

কাঠামো ggml_tensor * ggml_trunc(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

কাঠামো ggml_tensor * ggml_trunc_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

কাঠামো ggml_tensor * ggml_glu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_glu_op      op,
        bool                  swapped) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, op, swapped);
}

কাঠামো ggml_tensor * ggml_glu_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        গণনা ggml_glu_op      op) {
    ফেরত ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

কাঠামো ggml_tensor * ggml_reglu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_REGLU, false);
}

কাঠামো ggml_tensor * ggml_reglu_swapped(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_REGLU, true);
}

কাঠামো ggml_tensor * ggml_reglu_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

কাঠামো ggml_tensor * ggml_geglu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU, false);
}

কাঠামো ggml_tensor * ggml_geglu_swapped(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU, true);
}

কাঠামো ggml_tensor * ggml_geglu_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

কাঠামো ggml_tensor * ggml_swiglu(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_SWIGLU, false);
}

কাঠামো ggml_tensor * ggml_swiglu_swapped(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_SWIGLU, true);
}

কাঠামো ggml_tensor * ggml_swiglu_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

কাঠামো ggml_tensor * ggml_geglu_erf(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU_ERF, false);
}

কাঠামো ggml_tensor * ggml_geglu_erf_swapped(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU_ERF, true);
}

কাঠামো ggml_tensor * ggml_geglu_erf_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

কাঠামো ggml_tensor * ggml_geglu_quick(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU_QUICK, false);
}

কাঠামো ggml_tensor * ggml_geglu_quick_swapped(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_glu_impl(ctx, a, ফাঁকা, GGML_GLU_OP_GEGLU_QUICK, true);
}

কাঠামো ggml_tensor * ggml_geglu_quick_split(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

কাঠামো ggml_tensor * ggml_swiglu_oai(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 alpha,
        দশমিক                 limit) {
    কাঠামো ggml_tensor * ফলাফল = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(ফলাফল, 2, alpha);
    ggml_set_op_params_f32(ফলাফল, 3, limit);

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_swiglu_clamp(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 limit) {
    কাঠামো ggml_tensor * ফলাফল = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(ফলাফল, 3, limit);

    ফেরত ফলাফল;
}

// ggml_norm

স্থির কাঠামো ggml_tensor * ggml_norm_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(ফলাফল, &eps, আকার(eps));

    ফলাফল->op     = GGML_OP_NORM;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_norm(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_norm_impl(ctx, a, eps, false);
}

কাঠামো ggml_tensor * ggml_norm_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

স্থির কাঠামো ggml_tensor * ggml_rms_norm_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(ফলাফল, &eps, আকার(eps));

    ফলাফল->op     = GGML_OP_RMS_NORM;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_rms_norm(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_rms_norm_impl(ctx, a, eps, false);
}

কাঠামো ggml_tensor * ggml_rms_norm_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

কাঠামো ggml_tensor * ggml_rms_norm_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 eps) {
    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(ফলাফল, &eps, আকার(eps));

    ফলাফল->op     = GGML_OP_RMS_NORM_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_group_norm

স্থির কাঠামো ggml_tensor * ggml_group_norm_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_groups,
        দশমিক                 eps,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(ফলাফল, 0, n_groups);
    ggml_set_op_params_f32(ফলাফল, 1, eps);

    ফলাফল->op     = GGML_OP_GROUP_NORM;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_group_norm(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_groups,
        দশমিক                 eps) {
    ফেরত ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

কাঠামো ggml_tensor * ggml_group_norm_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_groups,
        দশমিক                 eps) {
    ফেরত ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

স্থির কাঠামো ggml_tensor * ggml_l2_norm_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(ফলাফল, 0, eps);

    ফলাফল->op     = GGML_OP_L2_NORM;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_l2_norm(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_l2_norm_impl(ctx, a, eps, false);
}

কাঠামো ggml_tensor * ggml_l2_norm_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 eps) {
    ফেরত ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        কাঠামো ggml_tensor * a,
        গণনা ggml_prec       prec) {
    নির্বাচন (a->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            {
                ধ্রুবক int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            বিরতি;
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            {
                ধ্রুবক int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত false;
    };

    ফেরত true;
}

bool ggml_prec_set_src(
        কাঠামো ggml_tensor * a,
        গণনা ggml_prec       prec,
        পূর্ণসংখ্যা                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    নির্বাচন (a->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            {
                যদি (idx != 1) {
                    ফেরত false;
                }

                ধ্রুবক int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত false;
    };

    ফেরত true;
}

// ggml_mul_mat

স্থির অন্তঃরেখ bool ggml_can_mul_mat(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

কাঠামো ggml_tensor * ggml_mul_mat(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ধ্রুবক int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_MUL_MAT;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

শূন্য ggml_mul_mat_set_prec(
        কাঠামো ggml_tensor * a,
        গণনা ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ধ্রুবক int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

শূন্য ggml_mul_mat_set_hint(
        কাঠামো ggml_tensor * a,
        গণনা ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ধ্রুবক int32_t hint_i32 = (int32_t) hint;

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
কাঠামো ggml_tensor * ggml_mul_mat_id(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * as,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ধ্রুবক int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_MUL_MAT_ID;
    ফলাফল->src[0] = as;
    ফলাফল->src[1] = b;
    ফলাফল->src[2] = ids;

    ফেরত ফলাফল;
}

// ggml_out_prod

স্থির অন্তঃরেখ bool ggml_can_out_prod(ধ্রুবক কাঠামো ggml_tensor * t0, ধ্রুবক কাঠামো ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ফেরত (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

কাঠামো ggml_tensor * ggml_out_prod(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ধ্রুবক int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_OUT_PROD;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_scale

স্থির কাঠামো ggml_tensor * ggml_scale_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 s,
        দশমিক                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    দশমিক params[2] = { s, b };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op     = GGML_OP_SCALE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_scale(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 s) {
    ফেরত ggml_scale_impl(ctx, a, s, 0.0, false);
}

কাঠামো ggml_tensor * ggml_scale_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 s) {
    ফেরত ggml_scale_impl(ctx, a, s, 0.0, true);
}

কাঠামো ggml_tensor * ggml_scale_bias(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 s,
        দশমিক                 b) {
    ফেরত ggml_scale_impl(ctx, a, s, b, false);
}

কাঠামো ggml_tensor * ggml_scale_bias_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 s,
        দশমিক                 b) {
    ফেরত ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

স্থির কাঠামো ggml_tensor * ggml_set_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_SET;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_set(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

কাঠামো ggml_tensor * ggml_set_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

কাঠামো ggml_tensor * ggml_set_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

কাঠামো ggml_tensor * ggml_set_1d_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

কাঠামো ggml_tensor * ggml_set_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

কাঠামো ggml_tensor * ggml_set_2d_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    ফেরত ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

স্থির কাঠামো ggml_tensor * ggml_cpy_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, b);
    যদি (strlen(b->name) > 0) {
        ggml_format_name(ফলাফল, "%s (copy of %s)", b->name, a->name);
    } অন্যথা {
        ggml_format_name(ফলাফল, "%s (copy)", a->name);
    }

    ফলাফল->op     = GGML_OP_CPY;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_cpy(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor * a,
        কাঠামো ggml_tensor * b) {
    ফেরত ggml_cpy_impl(ctx, a, b);
}

কাঠামো ggml_tensor * ggml_cast(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা   ggml_type      type) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(ফলাফল, "%s (copy)", a->name);

    ফলাফল->op     = GGML_OP_CPY;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = ফলাফল; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    ফেরত ফলাফল;
}

// ggml_cont

স্থির কাঠামো ggml_tensor * ggml_cont_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);
    ggml_format_name(ফলাফল, "%s (cont)", a->name);

    ফলাফল->op     = GGML_OP_CONT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_cont(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor * a) {
    ফেরত ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API কাঠামো ggml_tensor * ggml_cont_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0) {
    ফেরত ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API কাঠামো ggml_tensor * ggml_cont_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    ফেরত ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API কাঠামো ggml_tensor * ggml_cont_3d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    ফেরত ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

কাঠামো ggml_tensor * ggml_cont_4d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(ফলাফল, "%s (cont)", a->name);

    ফলাফল->op     = GGML_OP_CONT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_reshape

কাঠামো ggml_tensor * ggml_reshape(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor * a,
        কাঠামো ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(ফলাফল, "%s (reshaped)", a->name);

    ফলাফল->op     = GGML_OP_RESHAPE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_reshape_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ধ্রুবক int64_t ne[1] = { ne0 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(ফলাফল, "%s (reshaped)", a->name);

    ফলাফল->op     = GGML_OP_RESHAPE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_reshape_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ধ্রুবক int64_t ne[2] = { ne0, ne1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(ফলাফল, "%s (reshaped)", a->name);

    ফলাফল->op     = GGML_OP_RESHAPE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_reshape_3d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ধ্রুবক int64_t ne[3] = { ne0, ne1, ne2 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(ফলাফল, "%s (reshaped)", a->name);

    ফলাফল->op     = GGML_OP_RESHAPE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_reshape_4d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ধ্রুবক int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(ফলাফল, "%s (reshaped)", a->name);

    ফলাফল->op     = GGML_OP_RESHAPE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

স্থির কাঠামো ggml_tensor * ggml_view_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_dims,
        ধ্রুবক int64_t       * ne,
        size_t                offset) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(ফলাফল, "%s (view)", a->name);

    ggml_set_op_params(ফলাফল, &offset, আকার(offset));

    ফলাফল->op     = GGML_OP_VIEW;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_view_1d

কাঠামো ggml_tensor * ggml_view_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    কাঠামো ggml_tensor * ফলাফল = ggml_view_impl(ctx, a, 1, &ne0, offset);

    ফেরত ফলাফল;
}

// ggml_view_2d

কাঠামো ggml_tensor * ggml_view_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ধ্রুবক int64_t ne[2] = { ne0, ne1 };

    কাঠামো ggml_tensor * ফলাফল = ggml_view_impl(ctx, a, 2, ne, offset);

    ফলাফল->nb[1] = nb1;
    ফলাফল->nb[2] = ফলাফল->nb[1]*ne1;
    ফলাফল->nb[3] = ফলাফল->nb[2];

    ফেরত ফলাফল;
}

// ggml_view_3d

কাঠামো ggml_tensor * ggml_view_3d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ধ্রুবক int64_t ne[3] = { ne0, ne1, ne2 };

    কাঠামো ggml_tensor * ফলাফল = ggml_view_impl(ctx, a, 3, ne, offset);

    ফলাফল->nb[1] = nb1;
    ফলাফল->nb[2] = nb2;
    ফলাফল->nb[3] = ফলাফল->nb[2]*ne2;

    ফেরত ফলাফল;
}

// ggml_view_4d

কাঠামো ggml_tensor * ggml_view_4d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ধ্রুবক int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    কাঠামো ggml_tensor * ফলাফল = ggml_view_impl(ctx, a, 4, ne, offset);

    ফলাফল->nb[1] = nb1;
    ফলাফল->nb[2] = nb2;
    ফলাফল->nb[3] = nb3;

    ফেরত ফলাফল;
}

// ggml_permute

কাঠামো ggml_tensor * ggml_permute(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   axis0,
        পূর্ণসংখ্যা                   axis1,
        পূর্ণসংখ্যা                   axis2,
        পূর্ণসংখ্যা                   axis3) {
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

    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);
    ggml_format_name(ফলাফল, "%s (permuted)", a->name);

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

    ফলাফল->ne[0] = ne[0];
    ফলাফল->ne[1] = ne[1];
    ফলাফল->ne[2] = ne[2];
    ফলাফল->ne[3] = ne[3];

    ফলাফল->nb[0] = nb[0];
    ফলাফল->nb[1] = nb[1];
    ফলাফল->nb[2] = nb[2];
    ফলাফল->nb[3] = nb[3];

    ফলাফল->op     = GGML_OP_PERMUTE;
    ফলাফল->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফেরত ফলাফল;
}

// ggml_transpose

কাঠামো ggml_tensor * ggml_transpose(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);
    ggml_format_name(ফলাফল, "%s (transposed)", a->name);

    ফলাফল->ne[0] = a->ne[1];
    ফলাফল->ne[1] = a->ne[0];

    ফলাফল->nb[0] = a->nb[1];
    ফলাফল->nb[1] = a->nb[0];

    ফলাফল->op     = GGML_OP_TRANSPOSE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_get_rows

কাঠামো ggml_tensor * ggml_get_rows(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    গণনা ggml_type type = GGML_TYPE_F32;
    যদি (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    ফলাফল->op     = GGML_OP_GET_ROWS;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_get_rows_back

কাঠামো ggml_tensor * ggml_get_rows_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    ফলাফল->op     = GGML_OP_GET_ROWS_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_set_rows

কাঠামো ggml_tensor * ggml_set_rows(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c) {
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

    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SET_ROWS;
    ফলাফল->src[0] = b;
    ফলাফল->src[1] = c;
    ফলাফল->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    ফেরত ফলাফল;
}

// ggml_diag

কাঠামো ggml_tensor * ggml_diag(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ধ্রুবক int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, 4, ne);

    ফলাফল->op     = GGML_OP_DIAG;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_diag_mask_inf

স্থির কাঠামো ggml_tensor * ggml_diag_mask_inf_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_DIAG_MASK_INF;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_diag_mask_inf(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past) {
    ফেরত ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

কাঠামো ggml_tensor * ggml_diag_mask_inf_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past) {
    ফেরত ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

স্থির কাঠামো ggml_tensor * ggml_diag_mask_zero_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_DIAG_MASK_ZERO;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_diag_mask_zero(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past) {
    ফেরত ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

কাঠামো ggml_tensor * ggml_diag_mask_zero_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_past) {
    ফেরত ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

স্থির কাঠামো ggml_tensor * ggml_clamp_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        দশমিক                 min,
        দশমিক                 max,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    দশমিক params[] = { min, max };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_CLAMP;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_clamp(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    দশমিক                 min,
    দশমিক                 max) {
    ফেরত ggml_clamp_impl(ctx, a, min, max, false);
}

কাঠামো ggml_tensor * ggml_clamp_inplace(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    দশমিক                 min,
    দশমিক                 max) {
    ফেরত ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

স্থির কাঠামো ggml_tensor * ggml_soft_max_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * mask,
        দশমিক                 scale,
        দশমিক                 max_bias,
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

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    দশমিক params[] = { scale, max_bias };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_SOFT_MAX;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = mask;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_soft_max(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_soft_max_impl(ctx, a, ফাঁকা, 1.0f, 0.0f, false);
}

কাঠামো ggml_tensor * ggml_soft_max_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a) {
    ফেরত ggml_soft_max_impl(ctx, a, ফাঁকা, 1.0f, 0.0f, true);
}

কাঠামো ggml_tensor * ggml_soft_max_ext(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * mask,
        দশমিক                 scale,
        দশমিক                 max_bias) {
    ফেরত ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

কাঠামো ggml_tensor * ggml_soft_max_ext_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * mask,
        দশমিক                 scale,
        দশমিক                 max_bias) {
    ফেরত ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

শূন্য ggml_soft_max_add_sinks(
        কাঠামো ggml_tensor * a,
        কাঠামো ggml_tensor * sinks) {
    যদি (!sinks) {
        a->src[2] = ফাঁকা;
        ফেরত;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ফাঁকা);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

স্থির কাঠামো ggml_tensor * ggml_soft_max_ext_back_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 scale,
        দশমিক                 max_bias,
        bool                  inplace) {
    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_SOFT_MAX_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    memcpy((দশমিক *) ফলাফল->op_params + 0, &scale,    আকার(দশমিক));
    memcpy((দশমিক *) ফলাফল->op_params + 1, &max_bias, আকার(দশমিক));

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_soft_max_ext_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 scale,
        দশমিক                 max_bias) {
    ফেরত ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

কাঠামো ggml_tensor * ggml_soft_max_ext_back_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        দশমিক                 scale,
        দশমিক                 max_bias) {
    ফেরত ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

স্থির কাঠামো ggml_tensor * ggml_rope_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   sections[GGML_MROPE_SECTIONS],
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow,
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

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    আকার(দশমিক));
    memcpy(params +  6, &freq_scale,   আকার(দশমিক));
    memcpy(params +  7, &ext_factor,   আকার(দশমিক));
    memcpy(params +  8, &attn_factor,  আকার(দশমিক));
    memcpy(params +  9, &beta_fast,    আকার(দশমিক));
    memcpy(params + 10, &beta_slow,    আকার(দশমিক));
    যদি (mrope_used && sections) {
        memcpy(params + 11, sections,  আকার(int32_t) * GGML_MROPE_SECTIONS);
    } অন্যথা {
        memset(params + 11, 0,         আকার(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_ROPE;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;
    ফলাফল->src[2] = c;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_rope(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode) {
    ফেরত ggml_rope_impl(
        ctx, a, b, ফাঁকা, n_dims, ফাঁকা, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

কাঠামো ggml_tensor * ggml_rope_multi(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   sections[GGML_MROPE_SECTIONS],
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

কাঠামো ggml_tensor * ggml_rope_multi_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   sections[GGML_MROPE_SECTIONS],
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

কাঠামো ggml_tensor * ggml_rope_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode) {
    ফেরত ggml_rope_impl(
        ctx, a, b, ফাঁকা, n_dims, ফাঁকা, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

কাঠামো ggml_tensor * ggml_rope_ext(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, c, n_dims, ফাঁকা, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

কাঠামো ggml_tensor * ggml_rope_ext_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, c, n_dims, ফাঁকা, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

কাঠামো ggml_tensor * ggml_rope_custom(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, ফাঁকা, n_dims, ফাঁকা, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

কাঠামো ggml_tensor * ggml_rope_custom_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    ফেরত ggml_rope_impl(
        ctx, a, b, ফাঁকা, n_dims, ফাঁকা, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
স্থির দশমিক ggml_rope_yarn_corr_dim(পূর্ণসংখ্যা n_dims, পূর্ণসংখ্যা n_ctx_orig, দশমিক n_rot, দশমিক base) {
    ফেরত n_dims * logf(n_ctx_orig / (n_rot * 2 * (দশমিক)M_PI)) / (2 * logf(base));
}

শূন্য ggml_rope_yarn_corr_dims(
    পূর্ণসংখ্যা n_dims, পূর্ণসংখ্যা n_ctx_orig, দশমিক freq_base, দশমিক beta_fast, দশমিক beta_slow, দশমিক dims[2]
) {
    // start and end correction dims
    দশমিক start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    দশমিক end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

কাঠামো ggml_tensor * ggml_rope_ext_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    কাঠামো ggml_tensor * ফলাফল = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ফলাফল->op = GGML_OP_ROPE_BACK;
    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_rope_multi_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c,
        পূর্ণসংখ্যা                   n_dims,
        পূর্ণসংখ্যা                   sections[4],
        পূর্ণসংখ্যা                   mode,
        পূর্ণসংখ্যা                   n_ctx_orig,
        দশমিক                 freq_base,
        দশমিক                 freq_scale,
        দশমিক                 ext_factor,
        দশমিক                 attn_factor,
        দশমিক                 beta_fast,
        দশমিক                 beta_slow) {
    কাঠামো ggml_tensor * ফলাফল = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ফলাফল->op = GGML_OP_ROPE_BACK;
    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_rope_set_offset(
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ধ্রুবক int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    ফেরত a;
}

স্থির int64_t ggml_calc_conv_output_আকার(int64_t ins, int64_t ks, পূর্ণসংখ্যা s, পূর্ণসংখ্যা p, পূর্ণসংখ্যা d) {
    ফেরত (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
কাঠামো ggml_tensor * ggml_im2col(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   d0,
        পূর্ণসংখ্যা                   d1,
        bool                  is_2D,
        গণনা ggml_type        dst_type) {
    যদি (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } অন্যথা {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ধ্রুবক int64_t OH = is_2D ? ggml_calc_conv_output_আকার(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ধ্রুবক int64_t OW =         ggml_calc_conv_output_আকার(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ধ্রুবক int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_IM2COL;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_im2col_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        int64_t             * ne,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   d0,
        পূর্ণসংখ্যা                   d1,
        bool                  is_2D) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_IM2COL_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_conv_1d

কাঠামো ggml_tensor * ggml_conv_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   d0) {
    কাঠামো ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    কাঠামো ggml_tensor * ফলাফল =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    ফলাফল = ggml_reshape_3d(ctx, ফলাফল, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    ফলাফল = ggml_cont(ctx, ggml_permute(ctx, ফলাফল, 0, 2, 1, 3));                // [N, OC, OL]

    ফেরত ফলাফল;
}

// ggml_conv_1d_ph

কাঠামো ggml_tensor* ggml_conv_1d_ph(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s,
        পূর্ণসংখ্যা                   d) {
    ফেরত ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

কাঠামো ggml_tensor * ggml_conv_1d_dw(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   d0) {
    কাঠামো ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    কাঠামো ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    কাঠামো ggml_tensor * ফলাফল = ggml_mul_mat(ctx, im2col, a);

    ফলাফল = ggml_reshape_3d(ctx, ফলাফল, ফলাফল->ne[0], ফলাফল->ne[2], 1);

    ফেরত ফলাফল;
}

// ggml_conv_1d_dw_ph

কাঠামো ggml_tensor * ggml_conv_1d_dw_ph(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   d0) {
    ফেরত ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

কাঠামো ggml_tensor * ggml_col2im_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   oc,
        পূর্ণসংখ্যা                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ধ্রুবক int64_t K_OC = a->ne[0];
    ধ্রুবক int64_t T_in = a->ne[1];
    ধ্রুবক int64_t K = K_OC / oc;
    ধ্রুবক int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ধ্রুবক int64_t ne[4] = { T_out, oc, 1, 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_COL2IM_1D;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_conv_transpose_1d

স্থির int64_t ggml_calc_conv_transpose_1d_output_আকার(int64_t ins, int64_t ks, পূর্ণসংখ্যা s, পূর্ণসংখ্যা p, পূর্ণসংখ্যা d) {
    ফেরত (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API কাঠামো ggml_tensor * ggml_conv_transpose_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ধ্রুবক int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_আকার(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_CONV_TRANSPOSE_1D;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
কাঠামো ggml_tensor * ggml_conv_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   d0,
        পূর্ণসংখ্যা                   d1) {
    কাঠামো ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    কাঠামো ggml_tensor * ফলাফল =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    ফলাফল = ggml_reshape_4d(ctx, ফলাফল, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    ফলাফল = ggml_cont(ctx, ggml_permute(ctx, ফলাফল, 0, 1, 3, 2)); // [N, OC, OH, OW]


    ফেরত ফলাফল;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
কাঠামো ggml_tensor * ggml_im2col_3d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        int64_t               IC,
        পূর্ণসংখ্যা                   s0, // stride width
        পূর্ণসংখ্যা                   s1, // stride height
        পূর্ণসংখ্যা                   s2, // stride depth
        পূর্ণসংখ্যা                   p0, // padding width
        পূর্ণসংখ্যা                   p1, // padding height
        পূর্ণসংখ্যা                   p2, // padding depth
        পূর্ণসংখ্যা                   d0, // dilation width
        পূর্ণসংখ্যা                   d1, // dilation height
        পূর্ণসংখ্যা                   d2, // dilation depth
        গণনা ggml_type        dst_type) {
    ধ্রুবক int64_t N = b->ne[3] / IC;
    ধ্রুবক int64_t ID = b->ne[2];
    ধ্রুবক int64_t IH = b->ne[1];
    ধ্রুবক int64_t IW = b->ne[0];

    ধ্রুবক int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    ধ্রুবক int64_t KD = a->ne[2];
    ধ্রুবক int64_t KH = a->ne[1];
    ধ্রুবক int64_t KW = a->ne[0];
    ধ্রুবক int64_t OD = ggml_calc_conv_output_আকার(ID, KD, s2, p2, d2);
    ধ্রুবক int64_t OH = ggml_calc_conv_output_আকার(IH, KH, s1, p1, d1);
    ধ্রুবক int64_t OW = ggml_calc_conv_output_আকার(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ধ্রুবক int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_IM2COL_3D;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
কাঠামো ggml_tensor * ggml_conv_3d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        int64_t               IC,
        পূর্ণসংখ্যা                   s0, // stride width
        পূর্ণসংখ্যা                   s1, // stride height
        পূর্ণসংখ্যা                   s2, // stride depth
        পূর্ণসংখ্যা                   p0, // padding width
        পূর্ণসংখ্যা                   p1, // padding height
        পূর্ণসংখ্যা                   p2, // padding depth
        পূর্ণসংখ্যা                   d0, // dilation width
        পূর্ণসংখ্যা                   d1, // dilation height
        পূর্ণসংখ্যা                   d2  // dilation depth
        ) {
    কাঠামো ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    কাঠামো ggml_tensor * ফলাফল =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    ফলাফল = ggml_reshape_4d(ctx, ফলাফল, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    ফলাফল = ggml_cont(ctx, ggml_permute(ctx, ফলাফল, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    ফলাফল = ggml_reshape_4d(ctx, ফলাফল, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    ফেরত ফলাফল;
}

// ggml_conv_2d_sk_p0

কাঠামো ggml_tensor * ggml_conv_2d_sk_p0(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

কাঠামো ggml_tensor * ggml_conv_2d_s1_ph(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    ফেরত ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

কাঠামো ggml_tensor * ggml_conv_2d_dw(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   d0,
        পূর্ণসংখ্যা                   d1) {
    কাঠামো ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    কাঠামো ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    কাঠামো ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    কাঠামো ggml_tensor * ফলাফল = ggml_mul_mat(ctx, new_a, new_b);
    ফলাফল = ggml_reshape_4d(ctx, ফলাফল, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    ফেরত ফলাফল;
}

// ggml_conv_2d_dw_direct

কাঠামো ggml_tensor * ggml_conv_2d_dw_direct(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   stride0,
        পূর্ণসংখ্যা                   stride1,
        পূর্ণসংখ্যা                   pad0,
        পূর্ণসংখ্যা                   pad1,
        পূর্ণসংখ্যা                   dilation0,
        পূর্ণসংখ্যা                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_আকার(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_আকার(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, b->type, 4, ne);

    যদি (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ধ্রুবক int64_t type_আকার = ggml_type_আকার(ফলাফল->type);
        GGML_ASSERT(ggml_blck_আকার(ফলাফল->type) == 1);
        ফলাফল->nb[0] = ফলাফল->ne[2] * type_আকার;
        ফলাফল->nb[1] = ফলাফল->ne[0] * ফলাফল->nb[0];
        ফলাফল->nb[2] = type_আকার;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_CONV_2D_DW;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;
    ফেরত ফলাফল;
}

// ggml_conv_2d_direct

কাঠামো ggml_tensor * ggml_conv_2d_direct(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        কাঠামো ggml_tensor  * b,   // input data [W, H, C, N]
        পূর্ণসংখ্যা                   s0,  // stride dimension 0
        পূর্ণসংখ্যা                   s1,  // stride dimension 1
        পূর্ণসংখ্যা                   p0,  // padding dimension 0
        পূর্ণসংখ্যা                   p1,  // padding dimension 1
        পূর্ণসংখ্যা                   d0,  // dilation dimension 0
        পূর্ণসংখ্যা                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_আকার(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_আকার(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(ফলাফল, 0, s0);
    ggml_set_op_params_i32(ফলাফল, 1, s1);
    ggml_set_op_params_i32(ফলাফল, 2, p0);
    ggml_set_op_params_i32(ফলাফল, 3, p1);
    ggml_set_op_params_i32(ফলাফল, 4, d0);
    ggml_set_op_params_i32(ফলাফল, 5, d1);

    ফলাফল->op = GGML_OP_CONV_2D;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_conv_3d_direct

কাঠামো ggml_tensor * ggml_conv_3d_direct(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        পূর্ণসংখ্যা                   s2,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   p2,
        পূর্ণসংখ্যা                   d0,
        পূর্ণসংখ্যা                   d1,
        পূর্ণসংখ্যা                   d2,
        পূর্ণসংখ্যা                   c,
        পূর্ণসংখ্যা                   n,
        পূর্ণসংখ্যা                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_আকার(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_আকার(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_আকার(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(ফলাফল, 0,  s0);
    ggml_set_op_params_i32(ফলাফল, 1,  s1);
    ggml_set_op_params_i32(ফলাফল, 2,  s2);
    ggml_set_op_params_i32(ফলাফল, 3,  p0);
    ggml_set_op_params_i32(ফলাফল, 4,  p1);
    ggml_set_op_params_i32(ফলাফল, 5,  p2);
    ggml_set_op_params_i32(ফলাফল, 6,  d0);
    ggml_set_op_params_i32(ফলাফল, 7,  d1);
    ggml_set_op_params_i32(ফলাফল, 8,  d2);
    ggml_set_op_params_i32(ফলাফল, 9,  c);
    ggml_set_op_params_i32(ফলাফল, 10, n);
    ggml_set_op_params_i32(ফলাফল, 11, oc);

    ফলাফল->op = GGML_OP_CONV_3D;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_conv_transpose_2d_p0

স্থির int64_t ggml_calc_conv_transpose_output_আকার(int64_t ins, int64_t ks, পূর্ণসংখ্যা s, পূর্ণসংখ্যা p) {
    ফেরত (ins - 1) * s - 2 * p + ks;
}

কাঠামো ggml_tensor * ggml_conv_transpose_2d_p0(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        পূর্ণসংখ্যা                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ধ্রুবক int64_t ne[4] = {
        ggml_calc_conv_transpose_output_আকার(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_আকার(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    কাঠামো ggml_tensor* ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(ফলাফল, 0, stride);

    ফলাফল->op     = GGML_OP_CONV_TRANSPOSE_2D;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_pool_*

স্থির int64_t ggml_calc_pool_output_আকার(int64_t ins, পূর্ণসংখ্যা ks, পূর্ণসংখ্যা s, দশমিক p) {
    ফেরত (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

কাঠামো ggml_tensor * ggml_pool_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_op_pool     op,
        পূর্ণসংখ্যা                   k0,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   p0) {
    ধ্রুবক int64_t ne[4] = {
        ggml_calc_pool_output_আকার(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_POOL_1D;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_pool_2d

কাঠামো ggml_tensor * ggml_pool_2d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_op_pool     op,
        পূর্ণসংখ্যা                   k0,
        পূর্ণসংখ্যা                   k1,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        দশমিক                 p0,
        দশমিক                 p1) {
    কাঠামো ggml_tensor * ফলাফল;
    ধ্রুবক int64_t ne[4] = {
        ggml_calc_pool_output_আকার(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_আকার(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_POOL_2D;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_pool_2d_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * af,
        গণনা ggml_op_pool     op,
        পূর্ণসংখ্যা                   k0,
        পূর্ণসংখ্যা                   k1,
        পূর্ণসংখ্যা                   s0,
        পূর্ণসংখ্যা                   s1,
        দশমিক                 p0,
        দশমিক                 p1) {
    কাঠামো ggml_tensor * ফলাফল;
    ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_POOL_2D_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = af;

    ফেরত ফলাফল;
}

// ggml_upscale / ggml_interpolate

স্থির কাঠামো ggml_tensor * ggml_interpolate_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_গুনতি);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t)mode);

    ফলাফল->op     = GGML_OP_UPSCALE;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_upscale(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   scale_factor,
        গণনা ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    ফেরত ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

কাঠামো ggml_tensor * ggml_upscale_ext(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   ne0,
        পূর্ণসংখ্যা                   ne1,
        পূর্ণসংখ্যা                   ne2,
        পূর্ণসংখ্যা                   ne3,
        গণনা ggml_scale_mode  mode) {
    ফেরত ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

কাঠামো ggml_tensor * ggml_interpolate(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    ফেরত ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

কাঠামো ggml_tensor * ggml_pad(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   p2,
        পূর্ণসংখ্যা                   p3) {
    ফেরত ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

কাঠামো ggml_tensor * ggml_pad_circular(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1,
        পূর্ণসংখ্যা                   p2,
        পূর্ণসংখ্যা                   p3) {
    ফেরত ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

কাঠামো ggml_tensor * ggml_pad_ext(
            কাঠামো ggml_context * ctx,
            কাঠামো ggml_tensor  * a,
            পূর্ণসংখ্যা                  lp0,
            পূর্ণসংখ্যা                  rp0,
            পূর্ণসংখ্যা                  lp1,
            পূর্ণসংখ্যা                  rp1,
            পূর্ণসংখ্যা                  lp2,
            পূর্ণসংখ্যা                  rp2,
            পূর্ণসংখ্যা                  lp3,
            পূর্ণসংখ্যা                  rp3
            ) {
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_set_op_params_i32(ফলাফল, 0, lp0);
    ggml_set_op_params_i32(ফলাফল, 1, rp0);
    ggml_set_op_params_i32(ফলাফল, 2, lp1);
    ggml_set_op_params_i32(ফলাফল, 3, rp1);
    ggml_set_op_params_i32(ফলাফল, 4, lp2);
    ggml_set_op_params_i32(ফলাফল, 5, rp2);
    ggml_set_op_params_i32(ফলাফল, 6, lp3);
    ggml_set_op_params_i32(ফলাফল, 7, rp3);
    ggml_set_op_params_i32(ফলাফল, 8, 0); // not circular by default


    ফলাফল->op     = GGML_OP_PAD;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_pad_ext_circular

কাঠামো ggml_tensor * ggml_pad_ext_circular(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                  lp0,
        পূর্ণসংখ্যা                  rp0,
        পূর্ণসংখ্যা                  lp1,
        পূর্ণসংখ্যা                  rp1,
        পূর্ণসংখ্যা                  lp2,
        পূর্ণসংখ্যা                  rp2,
        পূর্ণসংখ্যা                  lp3,
        পূর্ণসংখ্যা                  rp3
        ) {
    কাঠামো ggml_tensor * ফলাফল = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(ফলাফল, 8, 1); // circular
    ফেরত ফলাফল;
}

// ggml_pad_reflect_1d

কাঠামো ggml_tensor * ggml_pad_reflect_1d(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   p0,
        পূর্ণসংখ্যা                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_PAD_REFLECT_1D;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_roll

কাঠামো ggml_tensor * ggml_roll(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   shift0,
        পূর্ণসংখ্যা                   shift1,
        পূর্ণসংখ্যা                   shift2,
        পূর্ণসংখ্যা                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_আকার(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(ফলাফল, 0, shift0);
    ggml_set_op_params_i32(ফলাফল, 1, shift1);
    ggml_set_op_params_i32(ফলাফল, 2, shift2);
    ggml_set_op_params_i32(ফলাফল, 3, shift3);

    ফলাফল->op     = GGML_OP_ROLL;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_timestep_embedding

কাঠামো ggml_tensor * ggml_timestep_embedding(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * timesteps,
        পূর্ণসংখ্যা                   dim,
        পূর্ণসংখ্যা                   max_period) {

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(ফলাফল, 0, dim);
    ggml_set_op_params_i32(ফলাফল, 1, max_period);

    ফলাফল->op     = GGML_OP_TIMESTEP_EMBEDDING;
    ফলাফল->src[0] = timesteps;

    ফেরত ফলাফল;
}

// ggml_tri

কাঠামো ggml_tensor * ggml_tri(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    গণনা ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(ফলাফল, 0, type);

    ফলাফল->op = GGML_OP_TRI;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_fill

স্থির কাঠামো ggml_tensor * ggml_fill_impl(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    দশমিক                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(ফলাফল, 0, c);

    ফলাফল->op = GGML_OP_FILL;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_fill(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    দশমিক                 c) {
    ফেরত ggml_fill_impl(ctx, a, c, false);
}

কাঠামো ggml_tensor * ggml_fill_inplace(
    কাঠামো ggml_context * ctx,
    কাঠামো ggml_tensor  * a,
    দশমিক                 c) {
    ফেরত ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

কাঠামো ggml_tensor * ggml_argsort(
        কাঠামো ggml_context  * ctx,
        কাঠামো ggml_tensor   * a,
        গণনা ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) order);

    ফলাফল->op     = GGML_OP_ARGSORT;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_argsort_top_k

কাঠামো ggml_tensor * ggml_argsort_শীর্ষ_k(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    কাঠামো ggml_tensor * ফলাফল = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    ফলাফল = ggml_view_4d(ctx, ফলাফল,
                k, ফলাফল->ne[1], ফলাফল->ne[2], ফলাফল->ne[3],
                   ফলাফল->nb[1], ফলাফল->nb[2], ফলাফল->nb[3],
                0);

    ফেরত ফলাফল;
}

// ggml_top_k

কাঠামো ggml_tensor * ggml_শীর্ষ_k(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    ফলাফল->op     = GGML_OP_শীর্ষ_K;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_arange

কাঠামো ggml_tensor * ggml_arange(
        কাঠামো ggml_context * ctx,
        দশমিক                 start,
        দশমিক                 stop,
        দশমিক                 step) {
    GGML_ASSERT(stop > start);

    ধ্রুবক int64_t steps = (int64_t) ceilf((stop - start) / step);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(ফলাফল, 0, start);
    ggml_set_op_params_f32(ফলাফল, 1, stop);
    ggml_set_op_params_f32(ফলাফল, 2, step);

    ফলাফল->op = GGML_OP_ARANGE;

    ফেরত ফলাফল;
}

// ggml_flash_attn_ext

কাঠামো ggml_tensor * ggml_flash_attn_ext(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * q,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * mask,
        দশমিক                 scale,
        দশমিক                 max_bias,
        দশমিক                 logit_softcap) {
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
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    দশমিক params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_FLASH_ATTN_EXT;
    ফলাফল->src[0] = q;
    ফলাফল->src[1] = k;
    ফলাফল->src[2] = v;
    ফলাফল->src[3] = mask;

    ফেরত ফলাফল;
}


শূন্য ggml_flash_attn_ext_set_prec(
        কাঠামো ggml_tensor * a,
        গণনা ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ধ্রুবক int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

গণনা ggml_prec ggml_flash_attn_ext_get_prec(
        ধ্রুবক কাঠামো ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ধ্রুবক int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    ফেরত (গণনা ggml_prec) prec_i32;
}

শূন্য ggml_flash_attn_ext_set_n_kv_max(
        কাঠামো ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

শূন্য ggml_flash_attn_ext_add_sinks(
        কাঠামো ggml_tensor * a,
        কাঠামো ggml_tensor * sinks) {
    যদি (!sinks) {
        a->src[4] = ফাঁকা;
        ফেরত;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ফাঁকা);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

কাঠামো ggml_tensor * ggml_flash_attn_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * q,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ধ্রুবক int64_t     D = q->ne[0];
    ধ্রুবক int64_t     N = q->ne[1];
    ধ্রুবক int64_t     M = k->ne[1];
    ধ্রুবক int64_t   ne2 = q->ne[2];
    ধ্রুবক int64_t   ne3 = q->ne[3];
    ধ্রুবক int64_t kvne2 = k->ne[2];

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
    ধ্রুবক int64_t elem_q = ggml_nelements(q);
    ধ্রুবক int64_t elem_k = ggml_nelements(k);
    ধ্রুবক int64_t elem_v = ggml_nelements(v);

    গণনা ggml_type ফলাফল_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_আকার(ফলাফল_type) == 1);
    ধ্রুবক size_t tsize = ggml_type_আকার(ফলাফল_type);

    ধ্রুবক size_t offs_q = 0;
    ধ্রুবক size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ধ্রুবক size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ধ্রুবক size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ধ্রুবক size_t nelements = (end + tsize - 1)/tsize;

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(ফলাফল, &masked_i, আকার(masked_i));

    ফলাফল->op     = GGML_OP_FLASH_ATTN_BACK;
    ফলাফল->src[0] = q;
    ফলাফল->src[1] = k;
    ফলাফল->src[2] = v;
    ফলাফল->src[3] = d;

    ফেরত ফলাফল;
}

// ggml_ssm_conv

কাঠামো ggml_tensor * ggml_ssm_conv(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * sx,
        কাঠামো ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ধ্রুবক int64_t d_conv  = c->ne[0];
    ধ্রুবক int64_t d_inner = c->ne[1];
    ধ্রুবক int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ধ্রুবক int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    ফলাফল->op     = GGML_OP_SSM_CONV;
    ফলাফল->src[0] = sx;
    ফলাফল->src[1] = c;

    ফেরত ফলাফল;
}

// ggml_ssm_scan

কাঠামো ggml_tensor * ggml_ssm_scan(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * s,
        কাঠামো ggml_tensor  * x,
        কাঠামো ggml_tensor  * dt,
        কাঠামো ggml_tensor  * A,
        কাঠামো ggml_tensor  * B,
        কাঠামো ggml_tensor  * C,
        কাঠামো ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_আকার(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_আকার(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_আকার(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ধ্রুবক int64_t d_state      = s->ne[0];
        ধ্রুবক int64_t শীর্ষ_dim     = x->ne[0];
        ধ্রুবক int64_t n_শীর্ষ       = x->ne[1];
        ধ্রুবক int64_t n_seq_tokens = x->ne[2];
        ধ্রুবক int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_শীর্ষ);
        GGML_ASSERT(dt->ne[1] == n_seq_tokens);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == শীর্ষ_dim);
        GGML_ASSERT(s->ne[2] == n_শীর্ষ);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_tokens);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_vector(ids));
        GGML_ASSERT(A->ne[1] == n_শীর্ষ);
        GGML_ASSERT(ggml_is_matrix(A));

        যদি (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    ফলাফল->op   = GGML_OP_SSM_SCAN;
    ফলাফল->src[0] = s;
    ফলাফল->src[1] = x;
    ফলাফল->src[2] = dt;
    ফলাফল->src[3] = A;
    ফলাফল->src[4] = B;
    ফলাফল->src[5] = C;
    ফলাফল->src[6] = ids;

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) K);

    ফেরত ফলাফল;
}

// ggml_win_part

কাঠামো ggml_tensor * ggml_win_part(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ধ্রুবক পূর্ণসংখ্যা px = (w - a->ne[1]%w)%w;
    ধ্রুবক পূর্ণসংখ্যা py = (w - a->ne[2]%w)%w;

    ধ্রুবক পূর্ণসংখ্যা npx = (px + a->ne[1])/w;
    ধ্রুবক পূর্ণসংখ্যা npy = (py + a->ne[2])/w;
    ধ্রুবক পূর্ণসংখ্যা np  = npx*npy;

    ধ্রুবক int64_t ne[4] = { a->ne[0], w, w, np, };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_WIN_PART;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_win_unpart

কাঠামো ggml_tensor * ggml_win_unpart(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   w0,
        পূর্ণসংখ্যা                   h0,
        পূর্ণসংখ্যা                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ধ্রুবক int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(ফলাফল, params, আকার(params));

    ফলাফল->op     = GGML_OP_WIN_UNPART;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_get_rel_pos

কাঠামো ggml_tensor * ggml_get_rel_pos(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        পূর্ণসংখ্যা                   qh,
        পূর্ণসংখ্যা                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ধ্রুবক int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    ফলাফল->op     = GGML_OP_GET_REL_POS;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

// ggml_add_rel_pos

স্থির কাঠামো ggml_tensor * ggml_add_rel_pos_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * pw,
        কাঠামো ggml_tensor  * ph,
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

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(ফলাফল, 0, inplace ? 1 : 0);

    ফলাফল->op     = GGML_OP_ADD_REL_POS;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = pw;
    ফলাফল->src[2] = ph;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_add_rel_pos(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * pw,
        কাঠামো ggml_tensor  * ph) {
    ফেরত ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

কাঠামো ggml_tensor * ggml_add_rel_pos_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * pw,
        কাঠামো ggml_tensor  * ph) {
    ফেরত ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

কাঠামো ggml_tensor * ggml_rwkv_wkv6(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * r,
        কাঠামো ggml_tensor  * tf,
        কাঠামো ggml_tensor  * td,
        কাঠামো ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুবক int64_t S = k->ne[0];
    ধ্রুবক int64_t H = k->ne[1];
    ধ্রুবক int64_t n_tokens = k->ne[2];
    ধ্রুবক int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুবক int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_RWKV_WKV6;
    ফলাফল->src[0] = k;
    ফলাফল->src[1] = v;
    ফলাফল->src[2] = r;
    ফলাফল->src[3] = tf;
    ফলাফল->src[4] = td;
    ফলাফল->src[5] = state;

    ফেরত ফলাফল;
}

// ggml_gated_linear_attn

কাঠামো ggml_tensor * ggml_gated_linear_attn(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * q,
        কাঠামো ggml_tensor  * g,
        কাঠামো ggml_tensor  * state,
        দশমিক scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুবক int64_t S = k->ne[0];
    ধ্রুবক int64_t H = k->ne[1];
    ধ্রুবক int64_t n_tokens = k->ne[2];
    ধ্রুবক int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুবক int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(ফলাফল, 0, scale);

    ফলাফল->op     = GGML_OP_GATED_LINEAR_ATTN;
    ফলাফল->src[0] = k;
    ফলাফল->src[1] = v;
    ফলাফল->src[2] = q;
    ফলাফল->src[3] = g;
    ফলাফল->src[4] = state;

    ফেরত ফলাফল;
}

// ggml_rwkv_wkv7

কাঠামো ggml_tensor * ggml_rwkv_wkv7(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * r,
        কাঠামো ggml_tensor  * w,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ধ্রুবক int64_t S = k->ne[0];
    ধ্রুবক int64_t H = k->ne[1];
    ধ্রুবক int64_t n_tokens = k->ne[2];
    ধ্রুবক int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ধ্রুবক int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op     = GGML_OP_RWKV_WKV7;
    ফলাফল->src[0] = r;
    ফলাফল->src[1] = w;
    ফলাফল->src[2] = k;
    ফলাফল->src[3] = v;
    ফলাফল->src[4] = a;
    ফলাফল->src[5] = b;
    ফলাফল->src[6] = state;

    ফেরত ফলাফল;
}

// ggml_unary

স্থির কাঠামো ggml_tensor * ggml_unary_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) op);

    ফলাফল->op     = GGML_OP_UNARY;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_unary(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_unary_op    op) {
    ফেরত ggml_unary_impl(ctx, a, op, false);
}

কাঠামো ggml_tensor * ggml_unary_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        গণনা ggml_unary_op    op) {
    ফেরত ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

স্থির কাঠামো ggml_tensor * ggml_map_custom1_impl(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        ধ্রুবক  ggml_custom1_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    কাঠামো ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op     = GGML_OP_MAP_CUSTOM1;
    ফলাফল->src[0] = a;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_map_custom1(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        ধ্রুবক  ggml_custom1_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

কাঠামো ggml_tensor * ggml_map_custom1_inplace(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        ধ্রুবক  ggml_custom1_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

স্থির কাঠামো ggml_tensor * ggml_map_custom2_impl(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        ধ্রুবক  ggml_custom2_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    কাঠামো ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op     = GGML_OP_MAP_CUSTOM2;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_map_custom2(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        ধ্রুবক  ggml_custom2_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

কাঠামো ggml_tensor * ggml_map_custom2_inplace(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        ধ্রুবক  ggml_custom2_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

স্থির কাঠামো ggml_tensor * ggml_map_custom3_impl(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        কাঠামো ggml_tensor       * c,
        ধ্রুবক  ggml_custom3_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    কাঠামো ggml_tensor * ফলাফল = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    কাঠামো ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op     = GGML_OP_MAP_CUSTOM3;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;
    ফলাফল->src[2] = c;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_map_custom3(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        কাঠামো ggml_tensor       * c,
        ধ্রুবক  ggml_custom3_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

কাঠামো ggml_tensor * ggml_map_custom3_inplace(
        কাঠামো ggml_context      * ctx,
        কাঠামো ggml_tensor       * a,
        কাঠামো ggml_tensor       * b,
        কাঠামো ggml_tensor       * c,
        ধ্রুবক  ggml_custom3_op_t   fun,
        পূর্ণসংখ্যা                        n_tasks,
        শূন্য                     * userdata) {
    ফেরত ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

কাঠামো ggml_tensor * ggml_custom_4d(
        কাঠামো ggml_context * ctx,
        গণনা ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        কাঠামো ggml_tensor ** args,
        পূর্ণসংখ্যা                   n_args,
        ggml_custom_op_t      fun,
        পূর্ণসংখ্যা                   n_tasks,
        শূন্য                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    কাঠামো ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op = GGML_OP_CUSTOM;
    জন্য (পূর্ণসংখ্যা i = 0; i < n_args; i++) {
        ফলাফল->src[i] = args[i];
    }

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_custom_inplace(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor ** args,
        পূর্ণসংখ্যা                   n_args,
        ggml_custom_op_t      fun,
        পূর্ণসংখ্যা                   n_tasks,
        শূন্য                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);

    কাঠামো ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(ফলাফল, &params, আকার(params));

    ফলাফল->op = GGML_OP_CUSTOM;
    ফলাফল->src[0] = a;
    জন্য (পূর্ণসংখ্যা i = 0; i < n_args; i++) {
        ফলাফল->src[i + 1] = args[i];
    }

    ফেরত ফলাফল;
}
// ggml_cross_entropy_loss

কাঠামো ggml_tensor * ggml_cross_entropy_loss(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_1d(ctx, a->type, 1);

    ফলাফল->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_cross_entropy_loss_back

কাঠামো ggml_tensor * ggml_cross_entropy_loss_back(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
        কাঠামো ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    কাঠামো ggml_tensor * ফলাফল = ggml_dup_tensor(ctx, b);

    ফলাফল->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;
    ফলাফল->src[2] = c;

    ফেরত ফলাফল;
}

// opt_step_adamw

কাঠামো ggml_tensor * ggml_opt_step_adamw(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * grad,
        কাঠামো ggml_tensor  * m,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_OPT_STEP_ADAMW;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = grad;
    ফলাফল->src[2] = m;
    ফলাফল->src[3] = v;
    ফলাফল->src[4] = adamw_params;

    ফেরত ফলাফল;
}

// opt_step_sgd

কাঠামো ggml_tensor * ggml_opt_step_sgd(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * grad,
        কাঠামো ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    কাঠামো ggml_tensor * ফলাফল = ggml_view_tensor(ctx, a);

    ফলাফল->op     = GGML_OP_OPT_STEP_SGD;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = grad;
    ফলাফল->src[2] = params;

    ফেরত ফলাফল;
}

// solve_tri

কাঠামো ggml_tensor * ggml_solve_tri(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * a,
        কাঠামো ggml_tensor  * b,
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

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    ফলাফল->op     = GGML_OP_SOLVE_TRI;
    ফলাফল->src[0] = a;
    ফলাফল->src[1] = b;

    ফেরত ফলাফল;
}

// ggml_gated_delta_net

কাঠামো ggml_tensor * ggml_gated_delta_net(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * q,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * v,
        কাঠামো ggml_tensor  * g,
        কাঠামো ggml_tensor  * beta,
        কাঠামো ggml_tensor  * state,
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

    ধ্রুবক int64_t S_v      = v->ne[0];
    ধ্রুবক int64_t H        = v->ne[1];
    ধ্রুবক int64_t n_tokens = v->ne[2];
    ধ্রুবক int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ধ্রুবক int64_t state_rows = K * S_v * n_seqs;
    ধ্রুবক int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(ফলাফল, 0, (int32_t) K);

    ফলাফল->op     = GGML_OP_GATED_DELTA_NET;
    ফলাফল->src[0] = q;
    ফলাফল->src[1] = k;
    ফলাফল->src[2] = v;
    ফলাফল->src[3] = g;
    ফলাফল->src[4] = beta;
    ফলাফল->src[5] = state;

    ফেরত ফলাফল;
}

// ggml_lightning_indexer

কাঠামো ggml_tensor * ggml_lightning_indexer(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * q,
        কাঠামো ggml_tensor  * k,
        কাঠামো ggml_tensor  * weights,
        কাঠামো ggml_tensor  * mask) {

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
    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ফলাফল->op   = GGML_OP_LIGHTNING_INDEXER;
    ফলাফল->src[0] = q;
    ফলাফল->src[1] = k;
    ফলাফল->src[2] = weights;
    ফলাফল->src[3] = mask;

    ফেরত ফলাফল;
}

// ggml_dsv4_hc_comb

কাঠামো ggml_tensor * ggml_dsv4_hc_comb(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * mixes,
        কাঠামো ggml_tensor  * scale,
        কাঠামো ggml_tensor  * base,
        দশমিক                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ধ্রুবক int64_t hc_mix_dim = mixes->ne[0];
    ধ্রুবক int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    জন্য (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        যদি ((2 + i)*i == hc_mix_dim) {
            hc = i;
            বিরতি;
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

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(ফলাফল, 0, eps);
    ggml_set_op_params_i32(ফলাফল, 1, n_iter);

    ফলাফল->op     = GGML_OP_DSV4_HC_COMB;
    ফলাফল->src[0] = mixes;
    ফলাফল->src[1] = scale;
    ফলাফল->src[2] = base;

    ফেরত ফলাফল;
}

// ggml_dsv4_hc_pre

স্থির কাঠামো ggml_tensor * ggml_dsv4_hc_pre_impl(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * x,
        কাঠামো ggml_tensor  * weights,
        দশমিক                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ধ্রুবক int64_t n_embd   = x->ne[0];
    ধ্রুবক int64_t hc       = x->ne[1];
    ধ্রুবক int64_t n_tokens = x->ne[2];

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

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(ফলাফল, 0, scale);
    ggml_set_op_params_i32(ফলাফল, 1, gated ? 1 : 0);

    ফলাফল->op     = GGML_OP_DSV4_HC_PRE;
    ফলাফল->src[0] = x;
    ফলাফল->src[1] = weights;

    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_dsv4_hc_pre(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * x,
        কাঠামো ggml_tensor  * weights) {
    ফেরত ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

কাঠামো ggml_tensor * ggml_dsv4_hc_pre_gated(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * x,
        কাঠামো ggml_tensor  * gate,
        দশমিক                 scale) {
    ফেরত ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

কাঠামো ggml_tensor * ggml_dsv4_hc_post(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * x,
        কাঠামো ggml_tensor  * residual,
        কাঠামো ggml_tensor  * post,
        কাঠামো ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ধ্রুবক int64_t n_embd   = x->ne[0];
    ধ্রুবক int64_t n_tokens = x->ne[1];
    ধ্রুবক int64_t hc       = residual->ne[1];

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

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    ফলাফল->op     = GGML_OP_DSV4_HC_POST;
    ফলাফল->src[0] = x;
    ফলাফল->src[1] = residual;
    ফলাফল->src[2] = post;
    ফলাফল->src[3] = comb;

    ফেরত ফলাফল;
}

// ggml_top_k_mask

কাঠামো ggml_tensor * ggml_শীর্ষ_k_mask(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_tensor  * mask,
        কাঠামো ggml_tensor  * শীর্ষ_k,
        কাঠামো ggml_tensor  * score,
        কাঠামো ggml_tensor  * blocks,
        কাঠামো ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_আকার(mask->type));
    GGML_ASSERT(শীর্ষ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(শীর্ষ_k->ne[1] % mask->ne[1] == 0 && শীর্ষ_k->ne[2] % mask->ne[2] == 0 && শীর্ষ_k->ne[3] % mask->ne[3] == 0);

    যদি (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == শীর্ষ_k->ne[1] && score->ne[2] == শীর্ষ_k->ne[2] && score->ne[3] == শীর্ষ_k->ne[3]);
    }

    যদি (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    যদি (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == শীর্ষ_k->ne[1] && extra->ne[2] == শীর্ষ_k->ne[2] && extra->ne[3] == শীর্ষ_k->ne[3]);
    }

    কাঠামো ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], শীর্ষ_k->ne[1], শীর্ষ_k->ne[2], শীর্ষ_k->ne[3]);

    ফলাফল->op     = GGML_OP_শীর্ষ_K_MASK;
    ফলাফল->src[0] = mask;
    ফলাফল->src[1] = শীর্ষ_k;
    ফলাফল->src[2] = score;
    ফলাফল->src[3] = blocks;
    ফলাফল->src[4] = extra;

    ফেরত ফলাফল;
}

////////////////////////////////////////////////////////////////////////////////

কাঠামো ggml_hash_set ggml_hash_set_new(size_t আকার_2) {
    আকার_2 = ggml_hash_আকার(আকার_2);
    কাঠামো ggml_hash_set ফলাফল;
    ফলাফল.আকার_2 = আকার_2;
    ফলাফল.চাবিগুলি = GGML_MALLOC(আকার(কাঠামো ggml_tensor *) * আকার_2);
    ফলাফল.used = GGML_CALLOC(ggml_bitset_আকার(আকার_2), আকার(ggml_bitset_t));
    ফেরত ফলাফল;
}

শূন্য ggml_hash_set_reset(কাঠামো ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, আকার(ggml_bitset_t) * ggml_bitset_আকার(hash_set->আকার_2));
}

শূন্য ggml_hash_set_free(কাঠামো ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->চাবিগুলি);
}

size_t ggml_hash_আকার(size_t min_sz) {
    // next primes after powers of two
    স্থির ধ্রুবক size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    স্থির ধ্রুবক size_t n_primes = আকার(primes)/আকার(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    যখন (l < r) {
        size_t m = (l + r)/2;
        যদি (primes[m] < min_sz) {
            l = m + 1;
        } অন্যথা {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    ফেরত sz;
}

কাঠামো hash_map {
    কাঠামো ggml_hash_set set;
    কাঠামো ggml_tensor ** vals;
};

স্থির কাঠামো hash_map * ggml_new_hash_map(size_t আকার_2) {
    কাঠামো hash_map * ফলাফল = GGML_MALLOC(আকার(কাঠামো hash_map));
    ফলাফল->set = ggml_hash_set_new(আকার_2);
    ফলাফল->vals = GGML_CALLOC(ফলাফল->set.আকার_2, আকার(কাঠামো ggml_tensor *));
    ফেরত ফলাফল;
}

স্থির শূন্য ggml_hash_map_free(কাঠামো hash_map * map) {
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
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_cgraph  * cgraph,
        size_t                isrc,
        কাঠামো ggml_tensor  * tensor) {
    কাঠামো ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
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
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_cgraph  * cgraph,
        size_t                isrc,
        কাঠামো ggml_tensor  * tensor,
        ধ্রুবক  size_t         nb1,
        ধ্রুবক  size_t         nb2,
        ধ্রুবক  size_t         nb3,
        ধ্রুবক  size_t         offset) {
    কাঠামো ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
    GGML_ASSERT(src);
    যদি (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } অন্যথা {
        কাঠামো ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.চাবিগুলি[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

স্থির শূন্য ggml_add1_or_set(
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_cgraph  * cgraph,
        size_t                isrc,
        কাঠামো ggml_tensor  * tensor) {
    কাঠামো ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
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
        কাঠামো ggml_context * ctx,
        কাঠামো ggml_cgraph  * cgraph,
        size_t                isrc,
        কাঠামো ggml_tensor  * tensor) {
    কাঠামো ggml_tensor * src = cgraph->visited_hash_set.চাবিগুলি[isrc];
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
        কাঠামো ggml_context * ctx, কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা i, ধ্রুবক bool * grads_needed) {
    কাঠামো ggml_tensor * tensor = cgraph->nodes[i];
    কাঠামো ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    যদি (!grad) {
        ফেরত;
    }

    কাঠামো ggml_tensor * src0 = tensor->src[0];
    কাঠামো ggml_tensor * src1 = tensor->src[1];
    কাঠামো ggml_tensor * src2 = tensor->src[2];
    কাঠামো ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ধ্রুবক size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ধ্রুবক size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ধ্রুবক size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ধ্রুবক bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ধ্রুবক bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ধ্রুবক bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    নির্বাচন (tensor->op) {
        ক্ষেত্র GGML_OP_DUP: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_ADD: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                কাঠামো ggml_tensor * tmp = grad;
                যদি (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_ADD1: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_ACC: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ধ্রুবক size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ধ্রুবক size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ধ্রুবক size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ধ্রুবক size_t offset = ((int32_t *) tensor->op_params)[3];

                কাঠামো ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SUB: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            যদি (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_MUL: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            যদি (src1_needs_grads) {
                কাঠামো ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                যদি (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_DIV: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            যদি (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SQR: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SQRT: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_LOG: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SIN: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_COS: {
            যদি (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_যোগ: {
            যদি (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_যোগ_ROWS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_MEAN: {
            যদি (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_REPEAT: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_REPEAT_BACK: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_RMS_NORM: {
            যদি (src0_needs_grads) {
                দশমিক eps;
                memcpy(&eps, tensor->op_params, আকার(দশমিক));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT: {
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
                কাঠামো ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                যদি (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ধ্রুবক int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ধ্রুবক size_t nb2 = tmp->nb[2] * nr2;
                    ধ্রুবক size_t nb3 = tmp->nb[2];

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
        } বিরতি;
        ক্ষেত্র GGML_OP_SCALE: {
            যদি (src0_needs_grads) {
                দশমিক s;
                memcpy(&s, tensor->op_params, আকার(দশমিক));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SET: {
            ধ্রুবক size_t nb1    = ((ধ্রুবক int32_t *) tensor->op_params)[0];
            ধ্রুবক size_t nb2    = ((ধ্রুবক int32_t *) tensor->op_params)[1];
            ধ্রুবক size_t nb3    = ((ধ্রুবক int32_t *) tensor->op_params)[2];
            ধ্রুবক size_t offset = ((ধ্রুবক int32_t *) tensor->op_params)[3];

            কাঠামো ggml_tensor * tensor_grad_view = ফাঁকা;

            যদি (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            যদি (src0_needs_grads) {
                কাঠামো ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            যদি (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_CPY: {
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
        } বিরতি;
        ক্ষেত্র GGML_OP_CONT: {
            // same as cpy
            যদি (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_RESHAPE: {
            যদি (src0_needs_grads) {
                কাঠামো ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_VIEW: {
            যদি (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, আকার(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                যদি (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_আকার(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_আকার(src0);
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
        } বিরতি;
        ক্ষেত্র GGML_OP_PERMUTE: {
            যদি (src0_needs_grads) {
                ধ্রুবক int32_t * axes = (ধ্রুবক int32_t *) tensor->op_params;
                ধ্রুবক পূর্ণসংখ্যা axis0 = axes[0] & 0x3;
                ধ্রুবক পূর্ণসংখ্যা axis1 = axes[1] & 0x3;
                ধ্রুবক পূর্ণসংখ্যা axis2 = axes[2] & 0x3;
                ধ্রুবক পূর্ণসংখ্যা axis3 = axes[3] & 0x3;
                পূর্ণসংখ্যা axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_TRANSPOSE: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_GET_ROWS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            যদি (src1_needs_grads) {
                // noop
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_DIAG_MASK_INF: {
            যদি (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ধ্রুবক পূর্ণসংখ্যা n_past = ((ধ্রুবক int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_DIAG_MASK_ZERO: {
            যদি (src0_needs_grads) {
                ধ্রুবক পূর্ণসংখ্যা n_past = ((ধ্রুবক int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_SOFT_MAX: {
            যদি (src0_needs_grads) {
                দশমিক scale    = 1.0f;
                দশমিক max_bias = 0.0f;

                memcpy(&scale,    (ধ্রুবক দশমিক *) tensor->op_params + 0, আকার(দশমিক));
                memcpy(&max_bias, (ধ্রুবক দশমিক *) tensor->op_params + 1, আকার(দশমিক));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } বিরতি;
        ক্ষেত্র GGML_OP_ROPE: {
            যদি (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ধ্রুবক পূর্ণসংখ্যা n_dims     = ((ধ্রুবক int32_t *) tensor->op_params)[1];
                ধ্রুবক পূর্ণসংখ্যা mode       = ((ধ্রুবক int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ধ্রুবক পূর্ণসংখ্যা n_ctx_orig = ((ধ্রুবক int32_t *) tensor->op_params)[4];
                দশমিক freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                পূর্ণসংখ্যা sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ধ্রুবক দশমিক *) tensor->op_params +  5, আকার(দশমিক));
                memcpy(&freq_scale,  (ধ্রুবক দশমিক *) tensor->op_params +  6, আকার(দশমিক));
                memcpy(&ext_factor,  (ধ্রুবক দশমিক *) tensor->op_params +  7, আকার(দশমিক));
                memcpy(&attn_factor, (ধ্রুবক দশমিক *) tensor->op_params +  8, আকার(দশমিক));
                memcpy(&beta_fast,   (ধ্রুবক দশমিক *) tensor->op_params +  9, আকার(দশমিক));
                memcpy(&beta_slow,   (ধ্রুবক দশমিক *) tensor->op_params + 10, আকার(দশমিক));
                memcpy(&sections,                    tensor->op_params + 11, আকার(sections));

                কাঠামো ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } বিরতি;
        ক্ষেত্র GGML_OP_IM2COL: {
            যদি (src1_needs_grads) {
                ধ্রুবক int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ধ্রুবক int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ধ্রুবক int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ধ্রুবক int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ধ্রুবক int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ধ্রুবক int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ধ্রুবক bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_POOL_2D: {
            যদি (src0_needs_grads) {
                ধ্রুবক গণনা ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ধ্রুবক      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ধ্রুবক      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ধ্রুবক      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ধ্রুবক      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ধ্রুবক      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ধ্রুবক      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_WIN_PART:
        ক্ষেত্র GGML_OP_WIN_UNPART:
        ক্ষেত্র GGML_OP_UNARY: {
            নির্বাচন (ggml_get_unary_op(tensor)) {
                ক্ষেত্র GGML_UNARY_OP_ABS: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SGN: {
                    // noop
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_NEG: {
                    যদি (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_STEP: {
                    // noop
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_RELU: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SILU: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_EXP: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_EXPM1: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SOFTPLUS: {
                    যদি (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } বিরতি;
                পূর্বনির্ধারিত: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_CROSS_ENTROPY_LOSS: {
            যদি (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } বিরতি;
        ক্ষেত্র GGML_OP_GLU: {
            নির্বাচন (ggml_get_glu_op(tensor)) {
                ক্ষেত্র GGML_GLU_OP_SWIGLU: {
                    যদি (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    যদি (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } বিরতি;
                পূর্বনির্ধারিত: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } বিরতি;
        ক্ষেত্র GGML_OP_NONE: {
            // noop
        } বিরতি;
        ক্ষেত্র GGML_OP_গুনতি:
        পূর্বনির্ধারিত: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

স্থির size_t ggml_visit_parents_graph(কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_tensor * গ্রন্থি, bool compute) {
    যদি (গ্রন্থি->op != GGML_OP_NONE && compute) {
        গ্রন্থি->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ধ্রুবক size_t গ্রন্থি_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি);
    GGML_ASSERT(গ্রন্থি_hash_pos != GGML_HASHSET_FULL);

    যদি (ggml_bitset_get(cgraph->visited_hash_set.used, গ্রন্থি_hash_pos)) {
        // already visited

        যদি (compute) {
            // update the compute flag regardless
            জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; ++i) {
                কাঠামো ggml_tensor * src = গ্রন্থি->src[i];
                যদি (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        ফেরত গ্রন্থি_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.চাবিগুলি[গ্রন্থি_hash_pos] = গ্রন্থি;
    ggml_bitset_set(cgraph->visited_hash_set.used, গ্রন্থি_hash_pos);
    cgraph->use_counts[গ্রন্থি_hash_pos] = 0;

    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; ++i) {
        ধ্রুবক পূর্ণসংখ্যা k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_বাম_TO_ডান) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ডান_TO_বাম) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        কাঠামো ggml_tensor * src = গ্রন্থি->src[k];
        যদি (src) {
            ধ্রুবক size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    যদি (গ্রন্থি->op == GGML_OP_NONE && !(গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->আকার_2);

        যদি (strlen(গ্রন্থি->name) == 0) {
            ggml_format_name(গ্রন্থি, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = গ্রন্থি;
        cgraph->n_leafs++;
    } অন্যথা {
        GGML_ASSERT(cgraph->n_nodes < cgraph->আকার_2);

        যদি (strlen(গ্রন্থি->name) == 0) {
            ggml_format_name(গ্রন্থি, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = গ্রন্থি;
        cgraph->n_nodes++;
    }

    ফেরত গ্রন্থি_hash_pos;
}

স্থির শূন্য ggml_build_forward_impl(কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_tensor * tensor, bool expand, bool compute) {
    যদি (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ধ্রুবক পূর্ণসংখ্যা n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ধ্রুবক পূর্ণসংখ্যা n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    যদি (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

কাঠামো ggml_tensor * ggml_build_forward_select(
        কাঠামো ggml_cgraph  * cgraph,
        কাঠামো ggml_tensor ** tensors,
        পূর্ণসংখ্যা                   n_tensors,
        পূর্ণসংখ্যা                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    জন্য (পূর্ণসংখ্যা i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    ফেরত tensors[idx];
}

শূন্য ggml_build_forward_expand(কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

শূন্য ggml_build_forward_order(কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

শূন্য ggml_build_backward_expand(
        কাঠামো ggml_context *  ctx,
        কাঠামো ggml_cgraph  *  cgraph,
        কাঠামো ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ধ্রুবক পূর্ণসংখ্যা n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.আকার_2*আকার(কাঠামো ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.আকার_2*আকার(কাঠামো ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.আকার_2, আকার(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        জন্য (পূর্ণসংখ্যা i = 0; i < n_nodes_f; ++i) {
            কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];
            any_params = any_params || (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < n_nodes_f; ++i) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        যদি (গ্রন্থি->type == GGML_TYPE_I32) {
            চালু;
        }

        bool গ্রন্থি_needs_grad = (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM) || (গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        নির্বাচন (গ্রন্থি->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            ক্ষেত্র GGML_OP_IM2COL:      // only used for its shape
            ক্ষেত্র GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                বিরতি;
            ক্ষেত্র GGML_OP_UNARY: {
                ধ্রুবক গণনা ggml_unary_op uop = ggml_get_unary_op(গ্রন্থি);
                // SGN and STEP unary ops are piecewise constant
                যদি (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } বিরতি;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            ক্ষেত্র GGML_OP_CPY:           // gradients in CPY target are irrelevant
            ক্ষেত্র GGML_OP_GET_ROWS:      // row indices not differentiable
            ক্ষেত্র GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            ক্ষেত্র GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                বিরতি;

            পূর্বনির্ধারিত:
                বিরতি;
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; ++j) {
            যদি (!গ্রন্থি->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি->src[j])]) {
                চালু;
            }
            GGML_ASSERT(গ্রন্থি->src[j]->type == GGML_TYPE_F32 || গ্রন্থি->src[j]->type == GGML_TYPE_F16);
            গ্রন্থি_needs_grad = true;
            বিরতি;
        }
        যদি (!গ্রন্থি_needs_grad) {
            চালু;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!গ্রন্থি->view_src || গ্রন্থি->op == GGML_OP_CPY || গ্রন্থি->op == GGML_OP_SET_ROWS || গ্রন্থি->op == GGML_OP_VIEW ||
            গ্রন্থি->op == GGML_OP_RESHAPE || গ্রন্থি->op == GGML_OP_PERMUTE || গ্রন্থি->op == GGML_OP_TRANSPOSE);

        ধ্রুবক size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        যদি (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } অন্যথা যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, গ্রন্থি->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    জন্য (পূর্ণসংখ্যা i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    মুক্ত(grads_needed);
}

স্থির শূন্য * incr_ptr_aligned(শূন্য ** p, size_t আকার_2, size_t align) {
    শূন্য * ptr = *p;
    ptr = (শূন্য *) GGML_PAD((uintptr_t) ptr, align);
    *p = (শূন্য *) ((অক্ষর *) ptr + আকার_2);
    ফেরত ptr;
}

স্থির size_t ggml_graph_nbytes(size_t আকার_2, bool grads) {
    size_t hash_আকার = ggml_hash_আকার(আকার_2 * 2);
    শূন্য * p = (অক্ষর *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, আকার(কাঠামো ggml_cgraph), 1);
    incr_ptr_aligned(&p, আকার_2 * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, আকার_2 * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_আকার * আকার(int32_t), আকার(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)); // hash keys
    যদি (grads) {
        incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_আকার(hash_আকার) * আকার(ggml_bitset_t), আকার(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    ফেরত nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t আকার_2, bool grads) {
    ফেরত GGML_OBJECT_আকার + GGML_PAD(ggml_graph_nbytes(আকার_2, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(শূন্য) {
    ফেরত ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_আকার, false);
}

কাঠামো ggml_cgraph * ggml_new_graph_custom(কাঠামো ggml_context * ctx, size_t আকার_2, bool grads) {
    ধ্রুবক size_t obj_আকার = ggml_graph_nbytes(আকার_2, grads);
    কাঠামো ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_আকার);
    কাঠামো ggml_cgraph * cgraph = (কাঠামো ggml_cgraph *) ((অক্ষর *) ctx->mem_বাফার + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_আকার = ggml_hash_আকার(আকার_2 * 2);

    শূন্য * p = cgraph + 1;

    কাঠামো ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, আকার_2      * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *));
    কাঠামো ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, আকার_2      * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_আকার * আকার(int32_t), আকার(int32_t));
    কাঠামো ggml_tensor ** hash_চাবিগুলি_ptr  =         incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *));
    কাঠামো ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)) : ফাঁকা;
    কাঠামো ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_আকার * আকার(কাঠামো ggml_tensor *), আকার(কাঠামো ggml_tensor *)) : ফাঁকা;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_আকার(hash_আকার) * আকার(ggml_bitset_t), আকার(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    প্রতিজ্ঞা(obj_আকার == (size_t)((অক্ষর *)p - (অক্ষর *)cgraph));

    *cgraph = (কাঠামো ggml_cgraph) {
        /*.size         =*/ আকার_2,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_আকার, hash_used, hash_চাবিগুলি_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_বাম_TO_ডান,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    যদি (grads) {
        memset(cgraph->grads,     0, hash_আকার*আকার(কাঠামো ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_আকার*আকার(কাঠামো ggml_tensor *));
    }

    ফেরত cgraph;
}

কাঠামো ggml_cgraph * ggml_new_graph(কাঠামো ggml_context * ctx) {
    ফেরত ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_আকার, false);
}

কাঠামো ggml_cgraph ggml_graph_view(কাঠামো ggml_cgraph * cgraph0, পূর্ণসংখ্যা i0, পূর্ণসংখ্যা i1) {
    কাঠামো ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ফাঁকা, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ফাঁকা,
        /*.leafs            =*/ ফাঁকা,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    ফেরত cgraph;
}

শূন্য ggml_graph_cpy(কাঠামো ggml_cgraph * src, কাঠামো ggml_cgraph * dst) {
    GGML_ASSERT(dst->আকার_2 >= src->n_leafs);
    GGML_ASSERT(dst->আকার_2 >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.আকার_2 >= src->visited_hash_set.আকার_2);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    জন্য (পূর্ণসংখ্যা i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    জন্য (size_t i = 0; i < src->visited_hash_set.আকার_2; ++i) {
        // copy all hashset keys (tensors) that are in use
        যদি (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_ঢোকাও(&dst->visited_hash_set, src->visited_hash_set.চাবিগুলি[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    যদি (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.আকার_2*আকার(কাঠামো ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.আকার_2*আকার(কাঠামো ggml_tensor *));
    }
    যদি (src->grads) {
        GGML_ASSERT(dst->grads     != ফাঁকা);
        GGML_ASSERT(dst->grad_accs != ফাঁকা);
        জন্য (পূর্ণসংখ্যা i = 0; i < src->n_nodes; ++i) {
            ধ্রুবক size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ধ্রুবক size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

কাঠামো ggml_cgraph * ggml_graph_dup(কাঠামো ggml_context * ctx, কাঠামো ggml_cgraph * cgraph, bool force_grads) {
    কাঠামো ggml_cgraph * ফলাফল = ggml_new_graph_custom(ctx, cgraph->আকার_2, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, ফলাফল);
    ফেরত ফলাফল;
}

কাঠামো ggml_tensor * ggml_set_zero(কাঠামো ggml_tensor * tensor) {
    যদি (ggml_is_empty(tensor)) {
        ফেরত tensor;
    }
    যদি (tensor->বাফার) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } অন্যথা {
        GGML_ASSERT(tensor->তথ্য);
        memset(tensor->তথ্য, 0, ggml_nbytes(tensor));
    }
    ফেরত tensor;
}

শূন্য ggml_graph_reset(কাঠামো ggml_cgraph * cgraph) {
    যদি (!cgraph) {
        ফেরত;
    }
    GGML_ASSERT(cgraph->grads != ফাঁকা);

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি     = cgraph->nodes[i];
        কাঠামো ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, গ্রন্থি);

        যদি (গ্রন্থি->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(গ্রন্থি->src[2]);
            ggml_set_zero(গ্রন্থি->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        যদি (grad_acc) {
            যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ধ্রুবক দশমিক onef = 1.0f;
                যদি (grad_acc->বাফার) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, আকার(দশমিক));
                } অন্যথা {
                    GGML_ASSERT(grad_acc->তথ্য);
                    *((দশমিক *) grad_acc->তথ্য) = onef;
                }
            } অন্যথা {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

শূন্য ggml_graph_clear(কাঠামো ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

পূর্ণসংখ্যা ggml_graph_আকার(কাঠামো ggml_cgraph * cgraph) {
    ফেরত cgraph->আকার_2;
}

কাঠামো ggml_tensor * ggml_graph_গ্রন্থি(কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা i) {
    যদি (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        ফেরত cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    ফেরত cgraph->nodes[i];
}

কাঠামো ggml_tensor ** ggml_graph_nodes(কাঠামো ggml_cgraph * cgraph) {
    ফেরত cgraph->nodes;
}

পূর্ণসংখ্যা ggml_graph_n_nodes(কাঠামো ggml_cgraph * cgraph) {
    ফেরত cgraph->n_nodes;
}

শূন্য ggml_graph_add_গ্রন্থি(কাঠামো ggml_cgraph * cgraph, কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->আকার_2 > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

কাঠামো ggml_tensor * ggml_graph_get_tensor(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক অক্ষর * name) {
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_leafs; i++) {
        কাঠামো ggml_tensor * leaf = cgraph->leafs[i];

        যদি (strcmp(leaf->name, name) == 0) {
            ফেরত leaf;
        }
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        যদি (strcmp(গ্রন্থি->name, name) == 0) {
            ফেরত গ্রন্থি;
        }
    }

    ফেরত ফাঁকা;
}

কাঠামো ggml_tensor * ggml_graph_get_grad(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি) {
    ধ্রুবক size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি);
    ফেরত igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : ফাঁকা;
}

কাঠামো ggml_tensor * ggml_graph_get_grad_acc(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি) {
    ধ্রুবক size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি);
    ফেরত igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ফাঁকা;
}

শূন্য ggml_graph_print(ধ্রুবক কাঠামো ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                গ্রন্থি->ne[0], গ্রন্থি->ne[1], গ্রন্থি->ne[2],
                ggml_op_name(গ্রন্থি->op), (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, গ্রন্থি) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_leafs; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                গ্রন্থি->ne[0], গ্রন্থি->ne[1],
                ggml_op_name(গ্রন্থি->op),
                ggml_get_name(গ্রন্থি));
    }

    GGML_LOG_INFO("========================================\n");
}

স্থির পূর্ণসংখ্যা ggml_গ্রন্থি_list_find_tensor(ধ্রুবক কাঠামো ggml_cgraph * cgraph,
                                      ধ্রুবক পূর্ণসংখ্যা *                idxs,
                                      পূর্ণসংখ্যা                        গুনতি,
                                      ধ্রুবক কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    জন্য (পূর্ণসংখ্যা i = 0; i < গুনতি; ++i) {
        ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_idx = idxs[i];

        যদি (গ্রন্থি_idx >= cgraph->n_nodes) {
            ফেরত -1;
        }
        যদি (cgraph->nodes[গ্রন্থি_idx] == tensor) {
            ফেরত i;
        }
    }
    ফেরত -1;
}

স্থির bool ggml_is_constant(ধ্রুবক কাঠামো ggml_tensor * tensor) {
    ফেরত tensor->বাফার != ফাঁকা && ggml_backend_বাফার_get_usage(tensor->বাফার) == GGML_BACKEND_বাফার_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ধ্রুবক কাঠামো ggml_cgraph * cgraph,
                                ধ্রুবক পূর্ণসংখ্যা *                গ্রন্থি_idxs,
                                পূর্ণসংখ্যা                        গুনতি,
                                ধ্রুবক গণনা ggml_op *       ops,
                                ধ্রুবক পূর্ণসংখ্যা *                outputs,
                                পূর্ণসংখ্যা                        অঙ্ক_outputs) {
    GGML_ASSERT(outputs && অঙ্ক_outputs > 0);

    জন্য (পূর্ণসংখ্যা i = 0; i < গুনতি; ++i) {
        যদি (গ্রন্থি_idxs[i] >= cgraph->n_nodes) {
            ফেরত false;
        }

        ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[গ্রন্থি_idxs[i]];

        যদি (গ্রন্থি->op != ops[i]) {
            ফেরত false;
        }

        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ফেরত false;
        }

        যদি (ggml_গ্রন্থি_list_find_tensor(cgraph, outputs, অঙ্ক_outputs, গ্রন্থি) != -1) {
            চালু;
        }

        যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ফেরত false;
        }

        পূর্ণসংখ্যা subgraph_uses = 0;
        জন্য (পূর্ণসংখ্যা j = i + 1; j < গুনতি; ++j) {
            ধ্রুবক কাঠামো ggml_tensor * other_গ্রন্থি = cgraph->nodes[গ্রন্থি_idxs[j]];
            জন্য (পূর্ণসংখ্যা src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                যদি (other_গ্রন্থি->src[src_idx] == গ্রন্থি) {
                    subgraph_uses++;
                }
            }
        }

        যদি (subgraph_uses != ggml_গ্রন্থি_get_use_গুনতি(cgraph, গ্রন্থি_idxs[i])) {
            ফেরত false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        কাঠামো ggml_tensor * view_src = গ্রন্থি->view_src;
        যখন (view_src) {
            যদি (ggml_গ্রন্থি_list_find_tensor(cgraph, গ্রন্থি_idxs, গুনতি, view_src) == -1 && !ggml_is_constant(view_src)) {
                ফেরত false;
            }
            view_src = view_src->view_src;
        }
    }

    ফেরত true;
}

// check if node is part of the graph
স্থির bool ggml_graph_find(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি) {
    যদি (cgraph == ফাঁকা) {
        ফেরত true;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        যদি (cgraph->nodes[i] == গ্রন্থি) {
            ফেরত true;
        }
    }

    ফেরত false;
}

স্থির কাঠামো ggml_tensor * ggml_graph_get_parent(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি) {
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * parent = cgraph->nodes[i];
        কাঠামো ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        যদি (grad == গ্রন্থি) {
            ফেরত parent;
        }
    }

    ফেরত ফাঁকা;
}

স্থির শূন্য ggml_graph_dump_dot_গ্রন্থি_edge(FILE * fp, ধ্রুবক কাঠামো ggml_cgraph * gb, কাঠামো ggml_tensor * গ্রন্থি, কাঠামো ggml_tensor * parent, ধ্রুবক অক্ষর * label)  {
    কাঠামো ggml_tensor * gparent = ggml_graph_get_parent(gb, গ্রন্থি);
    কাঠামো ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (শূন্য *) gparent0 : (শূন্য *) parent,
            gparent ? (শূন্য *) gparent : (শূন্য *) গ্রন্থি,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

স্থির শূন্য ggml_graph_dump_dot_leaf_edge(FILE * fp, কাঠামো ggml_tensor * গ্রন্থি, কাঠামো ggml_tensor * parent, ধ্রুবক অক্ষর * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (শূন্য *) parent,
            (শূন্য *) গ্রন্থি,
            label);
}

শূন্য ggml_graph_dump_dot(ধ্রুবক কাঠামো ggml_cgraph * gb, ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক অক্ষর * filename) {
    অক্ষর color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    জন্য (পূর্ণসংখ্যা i = 0; i < gb->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = gb->nodes[i];
        কাঠামো ggml_tensor * grad = ggml_graph_get_grad(gb, গ্রন্থি);

        যদি (ggml_graph_get_parent(gb, গ্রন্থি) != ফাঁকা) {
            চালু;
        }

        যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, আকার(color), "yellow");
        } অন্যথা যদি (grad) {
            যদি (ggml_graph_find(cgraph, গ্রন্থি)) {
                snprintf(color, আকার(color), "green");
            } অন্যথা {
                snprintf(color, আকার(color), "lightblue");
            }
        } অন্যথা {
            snprintf(color, আকার(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (শূন্য *) গ্রন্থি, color);

        যদি (strlen(গ্রন্থি->name) > 0) {
            fprintf(fp, "%s (%s)|", গ্রন্থি->name, ggml_type_name(গ্রন্থি->type));
        } অন্যথা {
            fprintf(fp, "(%s)|", ggml_type_name(গ্রন্থি->type));
        }

        যদি (ggml_is_matrix(গ্রন্থি)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, গ্রন্থি->ne[0], গ্রন্থি->ne[1], ggml_op_প্রতীক(গ্রন্থি->op));
        } অন্যথা {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, গ্রন্থি->ne[0], গ্রন্থি->ne[1], গ্রন্থি->ne[2], ggml_op_প্রতীক(গ্রন্থি->op));
        }

        যদি (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_প্রতীক(grad->op));
        } অন্যথা {
            fprintf(fp, "\"; ]\n");
        }
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < gb->n_leafs; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = gb->leafs[i];

        snprintf(color, আকার(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (শূন্য *) গ্রন্থি, color);

        যদি (strlen(গ্রন্থি->name) > 0) {
            fprintf(fp, "%s (%s)|", গ্রন্থি->name, ggml_type_name(গ্রন্থি->type));
        } অন্যথা {
            fprintf(fp, "(%s)|", ggml_type_name(গ্রন্থি->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, গ্রন্থি->ne[0], গ্রন্থি->ne[1]);
        যদি (ggml_nelements(গ্রন্থি) < 5 && গ্রন্থি->তথ্য != ফাঁকা) {
            fprintf(fp, " | (");
            জন্য (পূর্ণসংখ্যা j = 0; j < ggml_nelements(গ্রন্থি); j++) {
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
                যদি (j < ggml_nelements(গ্রন্থি) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < gb->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = gb->nodes[i];

        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            যদি (গ্রন্থি->src[j]) {
                অক্ষর label[16];
                snprintf(label, আকার(label), "src %d", j);
                ggml_graph_dump_dot_গ্রন্থি_edge(fp, gb, গ্রন্থি, গ্রন্থি->src[j], label);
            }
        }
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < gb->n_leafs; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = gb->leafs[i];

        জন্য (পূর্ণসংখ্যা j = 0; j < GGML_MAX_SRC; j++) {
            যদি (গ্রন্থি->src[j]) {
                অক্ষর label[16];
                snprintf(label, আকার(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, গ্রন্থি, গ্রন্থি->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_set_প্রবেশ(কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_প্রবেশ;
}

শূন্য ggml_set_output(কাঠামো ggml_tensor * tensor) {
    জন্য (কাঠামো ggml_tensor * cur = tensor; cur != ফাঁকা; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

শূন্য ggml_set_param(কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

শূন্য ggml_set_loss(কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_quantize_init(গণনা ggml_type type) {
    ggml_critical_section_start();

    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
        ক্ষেত্র GGML_TYPE_IQ2_XS:
        ক্ষেত্র GGML_TYPE_IQ2_S:
        ক্ষেত্র GGML_TYPE_IQ1_S:
        ক্ষেত্র GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); বিরতি;
        পূর্বনির্ধারিত: // nothing
            বিরতি;
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
    ফেরত
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        গণনা ggml_type   type,
           ধ্রুবক দশমিক * src,
                  শূন্য * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ধ্রুবক দশমিক * imatrix) {
    ধ্রুবক int64_t n = nrows * n_per_row;

    যদি (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ফাঁকা);
    }

    GGML_ASSERT(start % type_traits[type].blck_আকার == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ধ্রুবক size_t start_row = start / n_per_row;
    ধ্রুবক size_t row_আকার  = ggml_row_আকার(type, n_per_row);

    size_t ফলাফল = 0;

    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_Q1_0:    ফলাফল = quantize_q1_0   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:    ফলাফল = quantize_q2_0   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:    ফলাফল = quantize_q4_0   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:    ফলাফল = quantize_q4_1   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:    ফলাফল = quantize_q5_0   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:    ফলাফল = quantize_q5_1   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:    ফলাফল = quantize_q8_0   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:   ফলাফল = quantize_mxfp4  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_NVFP4:   ফলাফল = quantize_nvfp4  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:    ফলাফল = quantize_q2_K   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:    ফলাফল = quantize_q3_K   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:    ফলাফল = quantize_q4_K   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:    ফলাফল = quantize_q5_K   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:    ফলাফল = quantize_q6_K   (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_TQ1_0:   ফলাফল = quantize_tq1_0  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_TQ2_0:   ফলাফল = quantize_tq2_0  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XXS: ফলাফল = quantize_iq2_xxs(src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XS:  ফলাফল = quantize_iq2_xs (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_XXS: ফলাফল = quantize_iq3_xxs(src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_S:   ফলাফল = quantize_iq3_s  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_S:   ফলাফল = quantize_iq2_s  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_S:   ফলাফল = quantize_iq1_s  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_M:   ফলাফল = quantize_iq1_m  (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_NL:  ফলাফল = quantize_iq4_nl (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_XS:  ফলাফল = quantize_iq4_xs (src + start, (অক্ষর *) dst + start_row * row_আকার, nrows, n_per_row, imatrix); বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            {
                size_t elemsize = আকার(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                ফলাফল = n * elemsize;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_BF16:
            {
                size_t elemsize = আকার(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                ফলাফল = n * elemsize;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_F32:
            {
                size_t elemsize = আকার(দশমিক);
                ফলাফল = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, ফলাফল);
            } বিরতি;
        পূর্বনির্ধারিত:
            প্রতিজ্ঞা(false);
    }

    GGML_ASSERT(ফলাফল == nrows * row_আকার);

    ফেরত ফলাফল;
}

////////////////////////////////////////////////////////////////////////////////

শূন্য ggml_log_get(ggml_log_callback * log_callback, শূন্য ** user_তথ্য) {
    *log_callback = g_logger_state.log_callback;
    *user_তথ্য    = g_logger_state.log_callback_user_তথ্য;
}

শূন্য ggml_log_set(ggml_log_callback log_callback, শূন্য * user_তথ্য) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_তথ্য = user_তথ্য;
}

শূন্য ggml_threadpool_params_init(কাঠামো ggml_threadpool_params * p, পূর্ণসংখ্যা n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

কাঠামো ggml_threadpool_params ggml_threadpool_params_default(পূর্ণসংখ্যা n_threads) {
    কাঠামো ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    ফেরত p;
}

bool ggml_threadpool_params_match(ধ্রুবক কাঠামো ggml_threadpool_params * p0, ধ্রুবক কাঠামো ggml_threadpool_params * p1) {
    যদি (p0->n_threads  != p1->n_threads  ) ফেরত false;
    যদি (p0->prio       != p1->prio       ) ফেরত false;
    যদি (p0->poll       != p1->poll       ) ফেরত false;
    যদি (p0->strict_cpu != p1->strict_cpu ) ফেরত false;
    ফেরত memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
