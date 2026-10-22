#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#અન્તર્ભાવ "ggml-version.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-threading.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml.h"

// FIXME: required here for quantization functions
#અન્તર્ભાવ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#અન્તર્ભાવ <hbwmalloc.h>
#endif

#જો defined(_MSC_VER) || defined(__MINGW32__)
#અન્તર્ભાવ <malloc.h> // using malloc.h with MSC/MINGW
#અથવાજો !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#અન્તર્ભાવ <alloca.h>
#endif

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <errno.h>
#અન્તર્ભાવ <time.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <inttypes.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <float.h>
#અન્તર્ભાવ <limits.h>
#અન્તર્ભાવ <stdarg.h>
#અન્તર્ભાવ <signal.h>
#જો defined(__gnu_linux__)
#અન્તર્ભાવ <syscall.h>
#endif

#જો defined(__APPLE__)
#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ <mach/mach.h>
#અન્તર્ભાવ <TargetConditionals.h>
#endif

#જો defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#અન્તર્ભાવ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(શૂન્ય) {
#ifdef _MSC_VER
#જો defined(_WIN32)
    સ્થિર અસ્થિર LONG counter = 1;
    પાછા (uint64_t) InterlockedIncrement(&counter) - 1;
#નહીંતર
    સ્થિર અસ્થિર long long counter = 1;
    પાછા (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#નહીંતર
    સ્થિર uint64_t counter = 1;
    પાછા __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#જો defined(__AVX512BF16__)
#જો defined(_MSC_VER)
#define m512i(p) p
#નહીંતર
#અન્તર્ભાવ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#જો defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <sys/wait.h>
#જો defined(__linux__)
#અન્તર્ભાવ <sys/prctl.h>
#endif

#જો defined(__ANDROID__)
#અન્તર્ભાવ <unwind.h>
#અન્તર્ભાવ <dlfcn.h>
#અન્તર્ભાવ <stdio.h>

સંરચના backtrace_state {
    શૂન્ય ** current;
    શૂન્ય ** end;
};

સ્થિર _Unwind_Reason_Code unwind_callback(સંરચના _Unwind_Context* context, શૂન્ય* arg) {
    સંરચના backtrace_state * state = (સંરચના backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    જો (pc) {
        જો (state->current == state->end) {
            પાછા URC_END_OF_ઢગલો;
        } નહીંતર {
            *state->current++ = (શૂન્ય*)pc;
        }
    }
    પાછા _URC_NO_REASON;
}

સ્થિર શૂન્ય ggml_print_backtrace_symbols(શૂન્ય) {
    અચલ પૂર્ણાંક max = 100;
    શૂન્ય* buffer[max];

    સંરચના backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    પૂર્ણાંક ગણતરી = state.current - buffer;

    માટે (પૂર્ણાંક idx = 0; idx < ગણતરી; ++idx) {
        અચલ શૂન્ય * addr = buffer[idx];
        અચલ અક્ષર * પ્રતીક = "";

        Dl_info info;
        જો (dladdr(addr, &info) && info.dli_sname) {
            પ્રતીક = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, પ્રતીક);
    }
}
#અથવાજો defined(__linux__) && defined(__GLIBC__)
#અન્તર્ભાવ <execinfo.h>
સ્થિર શૂન્ય ggml_print_backtrace_symbols(શૂન્ય) {
    શૂન્ય * trace[100];
    પૂર્ણાંક nptrs = backtrace(trace, કદ(trace)/કદ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#અથવાજો defined(__APPLE__)
#અન્તર્ભાવ <execinfo.h>
સ્થિર શૂન્ય ggml_print_backtrace_symbols(શૂન્ય) {
    શૂન્ય * trace[100];
    પૂર્ણાંક nptrs = backtrace(trace, કદ(trace)/કદ(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#નહીંતર
સ્થિર શૂન્ય ggml_print_backtrace_symbols(શૂન્ય) {
    // platform not supported
}
#endif

શૂન્ય ggml_print_backtrace(શૂન્ય) {
    અચલ અક્ષર * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    જો (GGML_NO_BACKTRACE) {
        પાછા;
    }
#જો defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    અચલ અક્ષર * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    જો (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        પાછા;
    }
#endif
#જો defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    અક્ષર * લીટી = ખાલી;
    ssize_t length = 0;
    જ્યારે ((length = getline(&લીટી, &size, f)) > 0) {
        જો (!strncmp(લીટી, "TracerPid:", કદ("TracerPid:") - 1) &&
            (length != કદ("TracerPid:\t0\n") - 1 || લીટી[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            મુક્ત(લીટી);
            fclose(f);
            પાછા;
        }
    }
    મુક્ત(લીટી);
    fclose(f);
    પૂર્ણાંક lock[2] = { -1, -1 };
    (શૂન્ય) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    અચલ પૂર્ણાંક parent_pid = getpid();
    અચલ પૂર્ણાંક child_pid = fork();
    જો (child_pid < 0) { // error
#જો defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        પાછા;
    } નહીંતર જો (child_pid == 0) { // child
        અક્ષર attach[32];
        snprintf(attach, કદ(attach), "attach %d", parent_pid);
#જો defined(__linux__)
        close(lock[1]);
        (શૂન્ય) !પઠ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (અક્ષર *) ખાલી);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[કદ("attach ") - 1],
            (અક્ષર *) ખાલી);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } નહીંતર { // parent
#જો defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, ખાલી, 0);
    }
}
#નહીંતર
શૂન્ય ggml_print_backtrace(શૂન્ય) {
    // platform not supported
}
#endif

સ્થિર ggml_abort_callback_t g_abort_callback = ખાલી;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    પાછા ret_val;
}

શૂન્ય ggml_abort(અચલ અક્ષર * file, પૂર્ણાંક લીટી, અચલ અક્ષર * fmt, ...) {
    fflush(stdout);

    અક્ષર message[2048];
    પૂર્ણાંક offset = snprintf(message, કદ(message), "%s:%d: ", file, લીટી);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, કદ(message) - offset, fmt, args);
    va_end(args);

    જો (g_abort_callback) {
        g_abort_callback(message);
    } નહીંતર {
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

સંરચના ggml_logger_state {
    ggml_log_callback log_callback;
    શૂન્ય * log_callback_user_data;
};
સ્થિર સંરચના ggml_logger_state g_logger_state = {ggml_log_callback_default, ખાલી};

સ્થિર શૂન્ય ggml_log_internal_v(ગણના ggml_log_level level, અચલ અક્ષર * format, va_list args) {
    જો (format == ખાલી) {
        પાછા;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    અક્ષર buffer[128];
    પૂર્ણાંક len = vsnprintf(buffer, 128, format, args);
    જો (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } નહીંતર {
        અક્ષર * buffer2 = (અક્ષર *) calloc(len + 1, કદ(અક્ષર));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        મુક્ત(buffer2);
    }
    va_end(args_copy);
}

શૂન્ય ggml_log_internal(ગણના ggml_log_level level, અચલ અક્ષર * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

શૂન્ય ggml_log_callback_default(ગણના ggml_log_level level, અચલ અક્ષર * text, શૂન્ય * user_data) {
    (શૂન્ય) level;
    (શૂન્ય) user_data;
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
#જો defined(__has_feature)
#જો __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#નહીંતર
#જો defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

શૂન્ય * ggml_aligned_malloc(size_t size) {
#જો defined(__s390x__)
    અચલ પૂર્ણાંક alignment = 256;
#નહીંતર
    અચલ પૂર્ણાંક alignment = 64;
#endif

#જો defined(_MSC_VER) || defined(__MINGW32__)
    પાછા _aligned_malloc(size, alignment);
#નહીંતર
    જો (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        પાછા ખાલી;
    }
    શૂન્ય * aligned_memory = ખાલી;
  #ifdef GGML_USE_CPU_HBM
    પૂર્ણાંક result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #અથવાજો TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    પૂર્ણાંક result = EFAULT;
    પસંદગી (alloc_status) {
        સ્થિતિ KERN_SUCCESS:
            result = 0;
            તોડો;
        સ્થિતિ KERN_INVALID_ADDRESS:
            result = EINVAL;
            તોડો;
        સ્થિતિ KERN_NO_SPACE:
            result = ENOMEM;
            તોડો;
        મૂળભૂત:
            result = EFAULT;
            તોડો;
    }
  #નહીંતર
    પૂર્ણાંક result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    જો (result != 0) {
        // Handle allocation failure
        અચલ અક્ષર *error_desc = "unknown allocation error";
        પસંદગી (result) {
            સ્થિતિ EINVAL:
                error_desc = "invalid alignment value";
                તોડો;
            સ્થિતિ ENOMEM:
                error_desc = "insufficient memory";
                તોડો;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        પાછા ખાલી;
    }
    પાછા aligned_memory;
#endif
}

શૂન્ય ggml_aligned_free(શૂન્ય * ptr, size_t size) {
    GGML_UNUSED(size);
#જો defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#અથવાજો GGML_USE_CPU_HBM
    જો (ptr != ખાલી) {
        hbw_free(ptr);
    }
#અથવાજો TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    જો (ptr != ખાલી) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#નહીંતર
    મુક્ત(ptr);
#endif
}


અંતઃરેખ સ્થિર શૂન્ય * ggml_malloc(size_t size) {
    જો (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        પાછા ખાલી;
    }
    શૂન્ય * result = malloc(size);
    જો (result == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    પાછા result;
}

// calloc
અંતઃરેખ સ્થિર શૂન્ય * ggml_calloc(size_t અંક, size_t size) {
    જો (અંક == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        પાછા ખાલી;
    }
    શૂન્ય * result = calloc(અંક, size);
    જો (result == ખાલી) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    પાછા result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(અંક, size) ggml_calloc(અંક, size)

#define GGML_FREE(ptr) મુક્ત(ptr)

અચલ અક્ષર * ggml_status_to_string(ગણના ggml_status status) {
    પસંદગી (status) {
        સ્થિતિ GGML_STATUS_ALLOC_FAILED: પાછા "GGML status: error (failed to allocate memory)";
        સ્થિતિ GGML_STATUS_FAILED:       પાછા "GGML status: error (operation failed)";
        સ્થિતિ GGML_STATUS_SUCCESS:      પાછા "GGML status: success";
        સ્થિતિ GGML_STATUS_ABORTED:      પાછા "GGML status: warning (operation aborted)";
    }

    પાછા "GGML status: unknown";
}

દશાંશ ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    પાછા GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(દશાંશ x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    પાછા GGML_FP32_TO_FP16(x);
}

દશાંશ ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    પાછા GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(દશાંશ x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    પાછા GGML_FP32_TO_BF16(x);
}

શૂન્ય ggml_fp16_to_fp32_row(અચલ ggml_fp16_t * x, દશાંશ * y, int64_t n) {
    માટે (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

શૂન્ય ggml_fp32_to_fp16_row(અચલ દશાંશ * x, ggml_fp16_t * y, int64_t n) {
    પૂર્ણાંક i = 0;
    માટે (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

શૂન્ય ggml_bf16_to_fp32_row(અચલ ggml_bf16_t * x, દશાંશ * y, int64_t n) {
    પૂર્ણાંક i = 0;
    માટે (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

શૂન્ય ggml_fp32_to_bf16_row_ref(અચલ દશાંશ * x, ggml_bf16_t * y, int64_t n) {
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

શૂન્ય ggml_fp32_to_bf16_row(અચલ દશાંશ * x, ggml_bf16_t * y, int64_t n) {
  પૂર્ણાંક i = 0;
#જો defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  માટે (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    માટે (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    પાછા memcmp(guid_a, guid_b, કદ(ggml_guid)) == 0;
}

અચલ અક્ષર * ggml_version(શૂન્ય) {
    પાછા GGML_VERSION;
}

અચલ અક્ષર * ggml_commit(શૂન્ય) {
    પાછા GGML_COMMIT;
}

//
// timing
//

#જો defined(_MSC_VER) || defined(__MINGW32__)
સ્થિર int64_t timer_freq, timer_start;
સ્થિર BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    પાછા TRUE;
}
શૂન્ય ggml_time_init(શૂન્ય) {
    સ્થિર INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, ખાલી, ખાલી);
}
int64_t ggml_time_ms(શૂન્ય) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    પાછા ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(શૂન્ય) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    પાછા ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#નહીંતર
શૂન્ય ggml_time_init(શૂન્ય) {}
int64_t ggml_time_ms(શૂન્ય) {
    સંરચના timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    પાછા (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(શૂન્ય) {
    સંરચના timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    પાછા (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(શૂન્ય) {
    પાછા clock();
}

int64_t ggml_cycles_per_ms(શૂન્ય) {
    પાછા CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
સ્થિર wchar_t * ggml_mbstowcs(અચલ અક્ષર * mbs) {
    પૂર્ણાંક wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, ખાલી, 0);
    જો (!wlen) {
        errno = EINVAL;
        પાછા ખાલી;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * કદ(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    જો (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        પાછા ખાલી;
    }

    પાછા wbuf;
}
#endif

FILE * ggml_fopen(અચલ અક્ષર * fname, અચલ અક્ષર * mode) {
#ifdef _WIN32
    FILE * file = ખાલી;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    જો (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        જો (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    પાછા file;
#નહીંતર
    પાછા fopen(fname, mode);
#endif

}

સ્થિર અચલ સંરચના ggml_type_traits type_traits[GGML_TYPE_ગણતરી] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = કદ(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = કદ(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = કદ(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = કદ(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = કદ(દ્વિગુણ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = કદ(દશાંશ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = કદ(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = કદ(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = કદ(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = કદ(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = કદ(block_q4_1),
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
        .type_size                = કદ(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = કદ(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = કદ(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = કદ(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = કદ(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = કદ(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = ખાલી,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = ખાલી,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = ખાલી,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = ખાલી,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = કદ(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = કદ(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = કદ(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = કદ(ggml_bf16_t),
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
        .type_size                = કદ(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = કદ(block_tq2_0),
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

અચલ સંરચના ggml_type_traits * ggml_get_type_traits(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા &type_traits[type];
}

//
// ggml object
//

સંરચના ggml_object {
    size_t offs;
    size_t size;

    સંરચના ggml_object * next;

    ગણના ggml_object_type type;

    અક્ષર padding[4];
};

સ્થિર અચલ size_t GGML_OBJECT_SIZE = કદ(સંરચના ggml_object);

//
// ggml context
//

સંરચના ggml_context {
    size_t mem_size;
    શૂન્ય * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    પૂર્ણાંક    n_objects;

    સંરચના ggml_object * objects_begin;
    સંરચના ggml_object * objects_end;
};

//
// data types
//

સ્થિર અચલ અક્ષર * GGML_OP_NAME[GGML_OP_ગણતરી] = {
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

static_assert(GGML_OP_ગણતરી == 102, "GGML_OP_COUNT != 102");

સ્થિર અચલ અક્ષર * GGML_OP_પ્રતીક[GGML_OP_ગણતરી] = {
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

static_assert(GGML_OP_ગણતરી == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_ગણતરી == 2, "GGML_OP_POOL_COUNT != 2");

સ્થિર અચલ અક્ષર * GGML_UNARY_OP_NAME[GGML_UNARY_OP_ગણતરી] = {
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

static_assert(GGML_UNARY_OP_ગણતરી == 22, "GGML_UNARY_OP_COUNT != 22");

સ્થિર અચલ અક્ષર * GGML_GLU_OP_NAME[GGML_GLU_OP_ગણતરી] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_ગણતરી == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(કદ(સંરચના ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(કદ(સંરચના ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

શૂન્ય ggml_print_object(અચલ સંરચના ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (અચલ શૂન્ય *) obj->next);
}

શૂન્ય ggml_print_objects(અચલ સંરચના ggml_context * ctx) {
    સંરચના ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (અચલ શૂન્ય *) ctx);

    જ્યારે (obj != ખાલી) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(અચલ સંરચના ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
        જો (tensor->ne[i] <= 0) {
            પાછા 0;
        }
    }

    size_t nbytes;
    અચલ size_t blck_size = ggml_blck_size(tensor->type);
    જો (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    નહીંતર {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    પાછા nbytes;
}

size_t ggml_nbytes_pad(અચલ સંરચના ggml_tensor * tensor) {
    પાછા GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા type_traits[type].blck_size;
}

size_t ggml_type_size(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા type_traits[type].type_size;
}

size_t ggml_row_size(ગણના ggml_type type, int64_t ne) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પ્રતિજ્ઞા(ne % ggml_blck_size(type) == 0);
    પાછા ggml_type_size(type)*ne/ggml_blck_size(type);
}

દ્વિગુણ ggml_type_sizef(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા ((દ્વિગુણ)(type_traits[type].type_size))/type_traits[type].blck_size;
}

અચલ અક્ષર * ggml_type_name(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા type_traits[type].type_name;
}

bool ggml_is_quantized(ગણના ggml_type type) {
    પ્રતિજ્ઞા(type >= 0);
    પ્રતિજ્ઞા(type < GGML_TYPE_ગણતરી);
    પાછા type_traits[type].is_quantized;
}

અચલ અક્ષર * ggml_op_name(ગણના ggml_op op) {
    પાછા GGML_OP_NAME[op];
}

અચલ અક્ષર * ggml_op_પ્રતીક(ગણના ggml_op op) {
    પાછા GGML_OP_પ્રતીક[op];
}

અચલ અક્ષર * ggml_unary_op_name(ગણના ggml_unary_op op) {
    પાછા GGML_UNARY_OP_NAME[op];
}

અચલ અક્ષર * ggml_glu_op_name(ગણના ggml_glu_op op) {
    પાછા GGML_GLU_OP_NAME[op];
}

અચલ અક્ષર * ggml_op_desc(અચલ સંરચના ggml_tensor * t) {
    જો (t->op == GGML_OP_UNARY) {
        ગણના ggml_unary_op uop = ggml_get_unary_op(t);
        પાછા ggml_unary_op_name(uop);
    }
    જો (t->op == GGML_OP_GLU) {
        ગણના ggml_glu_op gop = ggml_get_glu_op(t);
        પાછા ggml_glu_op_name(gop);
    }
    પાછા ggml_op_name(t->op);
}

size_t ggml_element_size(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_type_size(tensor->type);
}

bool ggml_is_scalar(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(અચલ સંરચના ggml_tensor * tensor) {
    પાછા tensor->ne[3] == 1;
}

પૂર્ણાંક ggml_n_dims(અચલ સંરચના ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        જો (tensor->ne[i] > 1) {
            પાછા i + 1;
        }
    }
    પાછા 1;
}

ગણના ggml_type ggml_ftype_to_ggml_type(ગણના ggml_ftype ftype) {
    ગણના ggml_type wtype = GGML_TYPE_ગણતરી;

    પસંદગી (ftype) {
        સ્થિતિ GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    તોડો;
        સ્થિતિ GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_ગણતરી; તોડો;
        સ્થિતિ GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_ગણતરી; તોડો;
    }

    GGML_ASSERT(wtype != GGML_TYPE_ગણતરી);

    પાછા wtype;
}

size_t ggml_tensor_overhead(શૂન્ય) {
    પાછા GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(અચલ સંરચના ggml_tensor * tensor) {
    પાછા tensor->nb[0] > tensor->nb[1];
}

સ્થિર bool ggml_is_contiguous_m_n(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક m, પૂર્ણાંક n) {
    size_t next_nb = ggml_type_size(tensor->type);
    જો (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        પાછા false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    માટે (પૂર્ણાંક i = 1; i < n; i++) {
        જો (i > m) {
            જો (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                પાછા false;
            }
            next_nb *= tensor->ne[i];
        } નહીંતર {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    પાછા true;
}

bool ggml_is_contiguous(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(અચલ સંરચના ggml_tensor * tensor) {
    પાછા ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(અચલ સંરચના ggml_tensor * tensor) {
    પાછા
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(અચલ સંરચના ggml_tensor * tensor) {
    પાછા
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

સ્થિર અંતઃરેખ bool ggml_is_padded_1d(અચલ સંરચના ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(અચલ સંરચના ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
        જો (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            પાછા true;
        }
    }
    પાછા false;
}

bool ggml_are_same_shape(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(અચલ સંરચના ggml_tensor * t) {
    પાછા ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

સ્થિર અંતઃરેખ bool ggml_can_repeat_rows(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

સંરચના ggml_context * ggml_init(સંરચના ggml_init_params params) {
    સ્થિર bool is_first_call = true;

    ggml_critical_section_start();

    જો (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    સંરચના ggml_context * ctx = GGML_MALLOC(કદ(સંરચના ggml_context));

    // allow to call ggml_init with 0 size
    જો (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    અચલ size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (સંરચના ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ખાલી,
        /*.objects_end        =*/ ખાલી,
    };

    GGML_ASSERT(ctx->mem_buffer != ખાલી);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    પાછા ctx;
}

શૂન્ય ggml_reset(સંરચના ggml_context * ctx) {
    જો (ctx == ખાલી) {
        પાછા;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ખાલી;
    ctx->objects_end   = ખાલી;
}

શૂન્ય ggml_free(સંરચના ggml_context * ctx) {
    જો (ctx == ખાલી) {
        પાછા;
    }

    જો (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(અચલ સંરચના ggml_context * ctx) {
    પાછા ctx->objects_end == ખાલી ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(સંરચના ggml_context * ctx) {
    પાછા ctx->no_alloc;
}

શૂન્ય ggml_set_no_alloc(સંરચના ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

શૂન્ય * ggml_get_mem_buffer(અચલ સંરચના ggml_context * ctx) {
    પાછા ctx->mem_buffer;
}

size_t ggml_get_mem_size(અચલ સંરચના ggml_context * ctx) {
    પાછા ctx->mem_size;
}

size_t ggml_get_max_tensor_size(અચલ સંરચના ggml_context * ctx) {
    size_t max_size = 0;

    માટે (સંરચના ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != ખાલી; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    પાછા max_size;
}

////////////////////////////////////////////////////////////////////////////////

સ્થિર સંરચના ggml_object * ggml_new_object(સંરચના ggml_context * ctx, ગણના ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    સંરચના ggml_object * obj_cur = ctx->objects_end;

    અચલ size_t cur_offs = obj_cur == ખાલી ? 0 : obj_cur->offs;
    અચલ size_t cur_size = obj_cur == ખાલી ? 0 : obj_cur->size;
    અચલ size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    અક્ષર * અચલ mem_buffer = ctx->mem_buffer;
    સંરચના ggml_object * અચલ obj_new = (સંરચના ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    જો (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        પાછા ખાલી;
    }
    જો (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        પાછા ખાલી;
    }

    જો (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        પાછા ખાલી;
    }

    *obj_new = (સંરચના ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = ખાલી,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    જો (obj_cur != ખાલી) {
        obj_cur->next = obj_new;
    } નહીંતર {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    પાછા obj_new;
}

સ્થિર સંરચના ggml_tensor * ggml_new_tensor_impl(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type      type,
        પૂર્ણાંક                   n_dims,
        અચલ int64_t       * ne,
        સંરચના ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_ગણતરી);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    જો (view_src != ખાલી && view_src->view_src != ખાલી) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    માટે (પૂર્ણાંક i = 1; i < n_dims; i++) {
        જો (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    માટે (પૂર્ણાંક i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == ખાલી || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    શૂન્ય * data = view_src != ખાલી ? view_src->data : ખાલી;
    જો (data != ખાલી) {
        data = (અક્ષર *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    જો (view_src == ખાલી && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    સંરચના ggml_object * અચલ obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    સંરચના ggml_tensor * અચલ result = (સંરચના ggml_tensor *)((અક્ષર *)ctx->mem_buffer + obj_new->offs);

    *result = (સંરચના ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ખાલી,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ખાલી },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (શૂન્ય *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ખાલી,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    માટે (પૂર્ણાંક i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    માટે (પૂર્ણાંક i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    પાછા result;
}

સંરચના ggml_tensor * ggml_new_tensor(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type      type,
        પૂર્ણાંક                   n_dims,
        અચલ int64_t       * ne) {
    પાછા ggml_new_tensor_impl(ctx, type, n_dims, ne, ખાલી, 0);
}

સંરચના ggml_tensor * ggml_new_tensor_1d(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type      type,
        int64_t ne0) {
    પાછા ggml_new_tensor(ctx, type, 1, &ne0);
}

સંરચના ggml_tensor * ggml_new_tensor_2d(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    અચલ int64_t ne[2] = { ne0, ne1 };
    પાછા ggml_new_tensor(ctx, type, 2, ne);
}

સંરચના ggml_tensor * ggml_new_tensor_3d(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    અચલ int64_t ne[3] = { ne0, ne1, ne2 };
    પાછા ggml_new_tensor(ctx, type, 3, ne);
}

સંરચના ggml_tensor * ggml_new_tensor_4d(
        સંરચના ggml_context * ctx,
        ગણના   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    અચલ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    પાછા ggml_new_tensor(ctx, type, 4, ne);
}

શૂન્ય * ggml_new_buffer(સંરચના ggml_context * ctx, size_t nbytes) {
    સંરચના ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    પાછા (uint8_t *)ctx->mem_buffer + obj->offs;
}

સંરચના ggml_tensor * ggml_dup_tensor(સંરચના ggml_context * ctx, અચલ સંરચના ggml_tensor * src) {
    પાછા ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

શૂન્ય ggml_unravel_index(અચલ સંરચના ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    અચલ int64_t ne2 = tensor->ne[2];
    અચલ int64_t ne1 = tensor->ne[1];
    અચલ int64_t ne0 = tensor->ne[0];

    અચલ int64_t i3_ = (i/(ne2*ne1*ne0));
    અચલ int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    અચલ int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    અચલ int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    જો (i0) {
        * i0 = i0_;
    }
    જો (i1) {
        * i1 = i1_;
    }
    જો (i2) {
        * i2 = i2_;
    }
    જો (i3) {
        * i3 = i3_;
    }
}

શૂન્ય * ggml_get_data(અચલ સંરચના ggml_tensor * tensor) {
    પાછા tensor->data;
}

દશાંશ * ggml_get_data_f32(અચલ સંરચના ggml_tensor * tensor) {
    પ્રતિજ્ઞા(tensor->type == GGML_TYPE_F32);
    પાછા (દશાંશ *)(tensor->data);
}

ગણના ggml_unary_op ggml_get_unary_op(અચલ સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    પાછા (ગણના ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

ગણના ggml_glu_op ggml_get_glu_op(અચલ સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    પાછા (ગણના ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

અચલ અક્ષર * ggml_get_name(અચલ સંરચના ggml_tensor * tensor) {
    પાછા tensor->name;
}

સંરચના ggml_tensor * ggml_set_name(સંરચના ggml_tensor * tensor, અચલ અક્ષર * name) {
    size_t i;
    માટે (i = 0; i < કદ(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    પાછા tensor;
}

સંરચના ggml_tensor * ggml_format_name(સંરચના ggml_tensor * tensor, અચલ અક્ષર * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, કદ(tensor->name), fmt, args);
    va_end(args);
    પાછા tensor;
}

સંરચના ggml_tensor * ggml_view_tensor(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * src) {
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    પાછા result;
}

સંરચના ggml_tensor * ggml_get_first_tensor(અચલ સંરચના ggml_context * ctx) {
    સંરચના ggml_object * obj = ctx->objects_begin;

    અક્ષર * અચલ mem_buffer = ctx->mem_buffer;

    જ્યારે (obj != ખાલી) {
        જો (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            પાછા (સંરચના ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    પાછા ખાલી;
}

સંરચના ggml_tensor * ggml_get_next_tensor(અચલ સંરચના ggml_context * ctx, સંરચના ggml_tensor * tensor) {
    સંરચના ggml_object * obj = (સંરચના ggml_object *) ((અક્ષર *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    અક્ષર * અચલ mem_buffer = ctx->mem_buffer;

    જ્યારે (obj != ખાલી) {
        જો (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            પાછા (સંરચના ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    પાછા ખાલી;
}

સંરચના ggml_tensor * ggml_get_tensor(સંરચના ggml_context * ctx, અચલ અક્ષર * name) {
    સંરચના ggml_object * obj = ctx->objects_begin;

    અક્ષર * અચલ mem_buffer = ctx->mem_buffer;

    જ્યારે (obj != ખાલી) {
        જો (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            સંરચના ggml_tensor * cur = (સંરચના ggml_tensor *)(mem_buffer + obj->offs);
            જો (strcmp(cur->name, name) == 0) {
                પાછા cur;
            }
        }

        obj = obj->next;
    }

    પાછા ખાલી;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

સ્થિર સંરચના ggml_tensor * ggml_dup_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_dup(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_dup_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_dup_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_dup_impl(ctx, a, true);
}

// ggml_add

સ્થિર સંરચના ggml_tensor * ggml_add_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_add(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_add_impl(ctx, a, b, false);
}

સંરચના ggml_tensor * ggml_add_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

સ્થિર સંરચના ggml_tensor * ggml_add_cast_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        ગણના   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_add_cast(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        ગણના   ggml_type      type) {
    પાછા ggml_add_cast_impl(ctx, a, b, type);
}

સંરચના ggml_tensor * ggml_add_id(
            સંરચના ggml_context * ctx,
            સંરચના ggml_tensor  * a,
            સંરચના ggml_tensor  * b,
            સંરચના ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    પાછા result;
}

// ggml_add1

સ્થિર સંરચના ggml_tensor * ggml_add1_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_add1(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_add1_impl(ctx, a, b, false);
}

સંરચના ggml_tensor * ggml_add1_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

સ્થિર સંરચના ggml_tensor * ggml_acc_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_acc(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    પાછા ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

સંરચના ggml_tensor * ggml_acc_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    પાછા ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

સ્થિર સંરચના ggml_tensor * ggml_sub_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_sub(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_sub_impl(ctx, a, b, false);
}

સંરચના ggml_tensor * ggml_sub_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

સ્થિર સંરચના ggml_tensor * ggml_mul_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_mul(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_mul_impl(ctx, a, b, false);
}

સંરચના ggml_tensor * ggml_mul_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

સ્થિર સંરચના ggml_tensor * ggml_div_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_div(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_div_impl(ctx, a, b, false);
}

સંરચના ggml_tensor * ggml_div_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

સ્થિર સંરચના ggml_tensor * ggml_sqr_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_sqr(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sqr_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_sqr_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

સ્થિર સંરચના ggml_tensor * ggml_sqrt_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_sqrt(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sqrt_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_sqrt_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

સ્થિર સંરચના ggml_tensor * ggml_log_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_log(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_log_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_log_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_log_impl(ctx, a, true);
}

સંરચના ggml_tensor * ggml_expm1(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

સંરચના ggml_tensor * ggml_expm1_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

સંરચના ggml_tensor * ggml_softplus(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

સંરચના ggml_tensor * ggml_softplus_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

સ્થિર સંરચના ggml_tensor * ggml_sin_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_sin(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sin_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_sin_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_sin_impl(ctx, a, true);
}

// ggml_cos

સ્થિર સંરચના ggml_tensor * ggml_cos_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_cos(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_cos_impl(ctx, a, false);
}

સંરચના ggml_tensor * ggml_cos_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_cos_impl(ctx, a, true);
}

// ggml_sum

સંરચના ggml_tensor * ggml_sum(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    પાછા result;
}

// ggml_sum_rows

સંરચના ggml_tensor * ggml_sum_rows(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    પાછા result;
}

// ggml_cumsum

સંરચના ggml_tensor * ggml_cumsum(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    પાછા result;
}

// ggml_mean

સંરચના ggml_tensor * ggml_mean(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    પાછા result;
}

// ggml_argmax

સંરચના ggml_tensor * ggml_argmax(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    પાછા result;
}

// ggml_count_equal

સંરચના ggml_tensor * ggml_ગણતરી_equal(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_ગણતરી_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_repeat

સંરચના ggml_tensor * ggml_repeat(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_repeat_4d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    અચલ bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    પાછા result;
}

// ggml_repeat_back

સંરચના ggml_tensor * ggml_repeat_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    પાછા result;
}

// ggml_concat

સંરચના ggml_tensor * ggml_concat(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    સંરચના ggml_tensor  * b,
    પૂર્ણાંક                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    માટે (પૂર્ણાંક d = 0; d < GGML_MAX_DIMS; ++d) {
        જો (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ચાલુ;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_abs

સંરચના ggml_tensor * ggml_abs(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

સંરચના ggml_tensor * ggml_abs_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

સંરચના ggml_tensor * ggml_sgn(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

સંરચના ggml_tensor * ggml_sgn_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

સંરચના ggml_tensor * ggml_neg(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

સંરચના ggml_tensor * ggml_neg_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

સંરચના ggml_tensor * ggml_step(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

સંરચના ggml_tensor * ggml_step_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

સંરચના ggml_tensor * ggml_tanh(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

સંરચના ggml_tensor * ggml_tanh_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

સંરચના ggml_tensor * ggml_elu(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

સંરચના ggml_tensor * ggml_elu_inplace(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

સંરચના ggml_tensor * ggml_relu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

સંરચના ggml_tensor * ggml_relu_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

સંરચના ggml_tensor * ggml_leaky_relu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 negative_slope,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, કદ(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    પાછા result;
}

// ggml_sigmoid

સંરચના ggml_tensor * ggml_sigmoid(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

સંરચના ggml_tensor * ggml_sigmoid_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

સંરચના ggml_tensor * ggml_gelu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

સંરચના ggml_tensor * ggml_gelu_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

સંરચના ggml_tensor * ggml_gelu_erf(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

સંરચના ggml_tensor * ggml_gelu_erf_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

સંરચના ggml_tensor * ggml_gelu_quick(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

સંરચના ggml_tensor * ggml_gelu_quick_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

સંરચના ggml_tensor * ggml_silu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

સંરચના ggml_tensor * ggml_silu_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

સંરચના ggml_tensor * ggml_xielu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ alpha_n,
        દશાંશ alpha_p,
        દશાંશ beta,
        દશાંશ eps) {
    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    પાછા result;
}

// ggml_silu_back

સંરચના ggml_tensor * ggml_silu_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml hardswish

સંરચના ggml_tensor * ggml_hardswish(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

સંરચના ggml_tensor * ggml_hardsigmoid(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

સંરચના ggml_tensor * ggml_exp(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

સંરચના ggml_tensor * ggml_exp_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

સ્થિર સંરચના ggml_tensor * ggml_glu_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        ગણના ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    જો (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ખાલી, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_floor

સંરચના ggml_tensor * ggml_floor(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

સંરચના ggml_tensor * ggml_floor_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

સંરચના ggml_tensor * ggml_ceil(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

સંરચના ggml_tensor * ggml_ceil_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

સંરચના ggml_tensor * ggml_round(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

સંરચના ggml_tensor * ggml_round_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

સંરચના ggml_tensor * ggml_trunc(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

સંરચના ggml_tensor * ggml_trunc_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

સંરચના ggml_tensor * ggml_glu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_glu_op      op,
        bool                  swapped) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, op, swapped);
}

સંરચના ggml_tensor * ggml_glu_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        ગણના ggml_glu_op      op) {
    પાછા ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

સંરચના ggml_tensor * ggml_reglu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_REGLU, false);
}

સંરચના ggml_tensor * ggml_reglu_swapped(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_REGLU, true);
}

સંરચના ggml_tensor * ggml_reglu_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

સંરચના ggml_tensor * ggml_geglu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU, false);
}

સંરચના ggml_tensor * ggml_geglu_swapped(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU, true);
}

સંરચના ggml_tensor * ggml_geglu_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

સંરચના ggml_tensor * ggml_swiglu(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_SWIGLU, false);
}

સંરચના ggml_tensor * ggml_swiglu_swapped(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_SWIGLU, true);
}

સંરચના ggml_tensor * ggml_swiglu_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

સંરચના ggml_tensor * ggml_geglu_erf(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU_ERF, false);
}

સંરચના ggml_tensor * ggml_geglu_erf_swapped(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU_ERF, true);
}

સંરચના ggml_tensor * ggml_geglu_erf_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

સંરચના ggml_tensor * ggml_geglu_quick(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU_QUICK, false);
}

સંરચના ggml_tensor * ggml_geglu_quick_swapped(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_glu_impl(ctx, a, ખાલી, GGML_GLU_OP_GEGLU_QUICK, true);
}

સંરચના ggml_tensor * ggml_geglu_quick_split(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

સંરચના ggml_tensor * ggml_swiglu_oai(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 alpha,
        દશાંશ                 limit) {
    સંરચના ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    પાછા result;
}

સંરચના ggml_tensor * ggml_swiglu_clamp(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 limit) {
    સંરચના ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    પાછા result;
}

// ggml_norm

સ્થિર સંરચના ggml_tensor * ggml_norm_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, કદ(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_norm(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_norm_impl(ctx, a, eps, false);
}

સંરચના ggml_tensor * ggml_norm_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

સ્થિર સંરચના ggml_tensor * ggml_rms_norm_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, કદ(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_rms_norm(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_rms_norm_impl(ctx, a, eps, false);
}

સંરચના ggml_tensor * ggml_rms_norm_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

સંરચના ggml_tensor * ggml_rms_norm_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 eps) {
    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, કદ(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_group_norm

સ્થિર સંરચના ggml_tensor * ggml_group_norm_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_groups,
        દશાંશ                 eps,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_group_norm(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_groups,
        દશાંશ                 eps) {
    પાછા ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

સંરચના ggml_tensor * ggml_group_norm_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_groups,
        દશાંશ                 eps) {
    પાછા ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

સ્થિર સંરચના ggml_tensor * ggml_l2_norm_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_l2_norm(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_l2_norm_impl(ctx, a, eps, false);
}

સંરચના ggml_tensor * ggml_l2_norm_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 eps) {
    પાછા ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        સંરચના ggml_tensor * a,
        ગણના ggml_prec       prec) {
    પસંદગી (a->op) {
        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                અચલ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                અચલ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            તોડો;
        મૂળભૂત:
            પાછા false;
    };

    પાછા true;
}

bool ggml_prec_set_src(
        સંરચના ggml_tensor * a,
        ગણના ggml_prec       prec,
        પૂર્ણાંક                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    પસંદગી (a->op) {
        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                જો (idx != 1) {
                    પાછા false;
                }

                અચલ int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            તોડો;
        મૂળભૂત:
            પાછા false;
    };

    પાછા true;
}

// ggml_mul_mat

સ્થિર અંતઃરેખ bool ggml_can_mul_mat(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

સંરચના ggml_tensor * ggml_mul_mat(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    અચલ int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

શૂન્ય ggml_mul_mat_set_prec(
        સંરચના ggml_tensor * a,
        ગણના ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    અચલ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

શૂન્ય ggml_mul_mat_set_hint(
        સંરચના ggml_tensor * a,
        ગણના ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    અચલ int32_t hint_i32 = (int32_t) hint;

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
સંરચના ggml_tensor * ggml_mul_mat_id(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * as,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    અચલ int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    પાછા result;
}

// ggml_out_prod

સ્થિર અંતઃરેખ bool ggml_can_out_prod(અચલ સંરચના ggml_tensor * t0, અચલ સંરચના ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

સંરચના ggml_tensor * ggml_out_prod(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    અચલ int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_scale

સ્થિર સંરચના ggml_tensor * ggml_scale_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 s,
        દશાંશ                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    દશાંશ params[2] = { s, b };
    ggml_set_op_params(result, &params, કદ(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_scale(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 s) {
    પાછા ggml_scale_impl(ctx, a, s, 0.0, false);
}

સંરચના ggml_tensor * ggml_scale_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 s) {
    પાછા ggml_scale_impl(ctx, a, s, 0.0, true);
}

સંરચના ggml_tensor * ggml_scale_bias(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 s,
        દશાંશ                 b) {
    પાછા ggml_scale_impl(ctx, a, s, b, false);
}

સંરચના ggml_tensor * ggml_scale_bias_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 s,
        દશાંશ                 b) {
    પાછા ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

સ્થિર સંરચના ggml_tensor * ggml_set_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_set(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

સંરચના ggml_tensor * ggml_set_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

સંરચના ggml_tensor * ggml_set_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

સંરચના ggml_tensor * ggml_set_1d_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

સંરચના ggml_tensor * ggml_set_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

સંરચના ggml_tensor * ggml_set_2d_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    પાછા ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

સ્થિર સંરચના ggml_tensor * ggml_cpy_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, b);
    જો (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } નહીંતર {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_cpy(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor * a,
        સંરચના ggml_tensor * b) {
    પાછા ggml_cpy_impl(ctx, a, b);
}

સંરચના ggml_tensor * ggml_cast(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના   ggml_type      type) {
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    પાછા result;
}

// ggml_cont

સ્થિર સંરચના ggml_tensor * ggml_cont_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_cont(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor * a) {
    પાછા ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API સંરચના ggml_tensor * ggml_cont_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0) {
    પાછા ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API સંરચના ggml_tensor * ggml_cont_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    પાછા ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API સંરચના ggml_tensor * ggml_cont_3d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    પાછા ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

સંરચના ggml_tensor * ggml_cont_4d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    પાછા result;
}

// ggml_reshape

સંરચના ggml_tensor * ggml_reshape(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor * a,
        સંરચના ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_reshape_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    અચલ int64_t ne[1] = { ne0 };
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_reshape_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    અચલ int64_t ne[2] = { ne0, ne1 };
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_reshape_3d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    અચલ int64_t ne[3] = { ne0, ne1, ne2 };
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_reshape_4d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    અચલ int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    પાછા result;
}

સ્થિર સંરચના ggml_tensor * ggml_view_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_dims,
        અચલ int64_t       * ne,
        size_t                offset) {
    સંરચના ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, કદ(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    પાછા result;
}

// ggml_view_1d

સંરચના ggml_tensor * ggml_view_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    સંરચના ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    પાછા result;
}

// ggml_view_2d

સંરચના ggml_tensor * ggml_view_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    અચલ int64_t ne[2] = { ne0, ne1 };

    સંરચના ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    પાછા result;
}

// ggml_view_3d

સંરચના ggml_tensor * ggml_view_3d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    અચલ int64_t ne[3] = { ne0, ne1, ne2 };

    સંરચના ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    પાછા result;
}

// ggml_view_4d

સંરચના ggml_tensor * ggml_view_4d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    અચલ int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    સંરચના ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    પાછા result;
}

// ggml_permute

સંરચના ggml_tensor * ggml_permute(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   axis0,
        પૂર્ણાંક                   axis1,
        પૂર્ણાંક                   axis2,
        પૂર્ણાંક                   axis3) {
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

    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, કદ(params));

    પાછા result;
}

// ggml_transpose

સંરચના ggml_tensor * ggml_transpose(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    પાછા result;
}

// ggml_get_rows

સંરચના ggml_tensor * ggml_get_rows(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    ગણના ggml_type type = GGML_TYPE_F32;
    જો (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_get_rows_back

સંરચના ggml_tensor * ggml_get_rows_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    સંરચના ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_set_rows

સંરચના ggml_tensor * ggml_set_rows(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c) {
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

    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    પાછા result;
}

// ggml_diag

સંરચના ggml_tensor * ggml_diag(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    અચલ int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    પાછા result;
}

// ggml_diag_mask_inf

સ્થિર સંરચના ggml_tensor * ggml_diag_mask_inf_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_diag_mask_inf(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past) {
    પાછા ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

સંરચના ggml_tensor * ggml_diag_mask_inf_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past) {
    પાછા ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

સ્થિર સંરચના ggml_tensor * ggml_diag_mask_zero_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_diag_mask_zero(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past) {
    પાછા ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

સંરચના ggml_tensor * ggml_diag_mask_zero_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_past) {
    પાછા ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

સ્થિર સંરચના ggml_tensor * ggml_clamp_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        દશાંશ                 min,
        દશાંશ                 max,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    દશાંશ params[] = { min, max };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_clamp(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    દશાંશ                 min,
    દશાંશ                 max) {
    પાછા ggml_clamp_impl(ctx, a, min, max, false);
}

સંરચના ggml_tensor * ggml_clamp_inplace(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    દશાંશ                 min,
    દશાંશ                 max) {
    પાછા ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

સ્થિર સંરચના ggml_tensor * ggml_soft_max_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * mask,
        દશાંશ                 scale,
        દશાંશ                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    જો (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    જો (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    દશાંશ params[] = { scale, max_bias };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    પાછા result;
}

સંરચના ggml_tensor * ggml_soft_max(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_soft_max_impl(ctx, a, ખાલી, 1.0f, 0.0f, false);
}

સંરચના ggml_tensor * ggml_soft_max_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a) {
    પાછા ggml_soft_max_impl(ctx, a, ખાલી, 1.0f, 0.0f, true);
}

સંરચના ggml_tensor * ggml_soft_max_ext(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * mask,
        દશાંશ                 scale,
        દશાંશ                 max_bias) {
    પાછા ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

સંરચના ggml_tensor * ggml_soft_max_ext_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * mask,
        દશાંશ                 scale,
        દશાંશ                 max_bias) {
    પાછા ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

શૂન્ય ggml_soft_max_add_sinks(
        સંરચના ggml_tensor * a,
        સંરચના ggml_tensor * sinks) {
    જો (!sinks) {
        a->src[2] = ખાલી;
        પાછા;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ખાલી);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

સ્થિર સંરચના ggml_tensor * ggml_soft_max_ext_back_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 scale,
        દશાંશ                 max_bias,
        bool                  inplace) {
    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((દશાંશ *) result->op_params + 0, &scale,    કદ(દશાંશ));
    memcpy((દશાંશ *) result->op_params + 1, &max_bias, કદ(દશાંશ));

    પાછા result;
}

સંરચના ggml_tensor * ggml_soft_max_ext_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 scale,
        દશાંશ                 max_bias) {
    પાછા ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

સંરચના ggml_tensor * ggml_soft_max_ext_back_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        દશાંશ                 scale,
        દશાંશ                 max_bias) {
    પાછા ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

સ્થિર સંરચના ggml_tensor * ggml_rope_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   sections[GGML_MROPE_SECTIONS],
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    જો (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } નહીંતર {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    જો (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    કદ(દશાંશ));
    memcpy(params +  6, &freq_scale,   કદ(દશાંશ));
    memcpy(params +  7, &ext_factor,   કદ(દશાંશ));
    memcpy(params +  8, &attn_factor,  કદ(દશાંશ));
    memcpy(params +  9, &beta_fast,    કદ(દશાંશ));
    memcpy(params + 10, &beta_slow,    કદ(દશાંશ));
    જો (mrope_used && sections) {
        memcpy(params + 11, sections,  કદ(int32_t) * GGML_MROPE_SECTIONS);
    } નહીંતર {
        memset(params + 11, 0,         કદ(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    પાછા result;
}

સંરચના ggml_tensor * ggml_rope(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode) {
    પાછા ggml_rope_impl(
        ctx, a, b, ખાલી, n_dims, ખાલી, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

સંરચના ggml_tensor * ggml_rope_multi(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   sections[GGML_MROPE_SECTIONS],
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

સંરચના ggml_tensor * ggml_rope_multi_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   sections[GGML_MROPE_SECTIONS],
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

સંરચના ggml_tensor * ggml_rope_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode) {
    પાછા ggml_rope_impl(
        ctx, a, b, ખાલી, n_dims, ખાલી, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

સંરચના ggml_tensor * ggml_rope_ext(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, c, n_dims, ખાલી, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

સંરચના ggml_tensor * ggml_rope_ext_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, c, n_dims, ખાલી, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

સંરચના ggml_tensor * ggml_rope_custom(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, ખાલી, n_dims, ખાલી, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

સંરચના ggml_tensor * ggml_rope_custom_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    પાછા ggml_rope_impl(
        ctx, a, b, ખાલી, n_dims, ખાલી, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
સ્થિર દશાંશ ggml_rope_yarn_corr_dim(પૂર્ણાંક n_dims, પૂર્ણાંક n_ctx_orig, દશાંશ n_rot, દશાંશ base) {
    પાછા n_dims * logf(n_ctx_orig / (n_rot * 2 * (દશાંશ)M_PI)) / (2 * logf(base));
}

શૂન્ય ggml_rope_yarn_corr_dims(
    પૂર્ણાંક n_dims, પૂર્ણાંક n_ctx_orig, દશાંશ freq_base, દશાંશ beta_fast, દશાંશ beta_slow, દશાંશ dims[2]
) {
    // start and end correction dims
    દશાંશ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    દશાંશ end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

સંરચના ggml_tensor * ggml_rope_ext_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    સંરચના ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    પાછા result;
}

સંરચના ggml_tensor * ggml_rope_multi_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c,
        પૂર્ણાંક                   n_dims,
        પૂર્ણાંક                   sections[4],
        પૂર્ણાંક                   mode,
        પૂર્ણાંક                   n_ctx_orig,
        દશાંશ                 freq_base,
        દશાંશ                 freq_scale,
        દશાંશ                 ext_factor,
        દશાંશ                 attn_factor,
        દશાંશ                 beta_fast,
        દશાંશ                 beta_slow) {
    સંરચના ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    પાછા result;
}

સંરચના ggml_tensor * ggml_rope_set_offset(
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    અચલ int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    પાછા a;
}

સ્થિર int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, પૂર્ણાંક s, પૂર્ણાંક p, પૂર્ણાંક d) {
    પાછા (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
સંરચના ggml_tensor * ggml_im2col(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   d0,
        પૂર્ણાંક                   d1,
        bool                  is_2D,
        ગણના ggml_type        dst_type) {
    જો (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } નહીંતર {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    અચલ int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    અચલ int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    અચલ int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_im2col_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        int64_t             * ne,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   d0,
        પૂર્ણાંક                   d1,
        bool                  is_2D) {
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_conv_1d

સંરચના ggml_tensor * ggml_conv_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   d0) {
    સંરચના ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    સંરચના ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    પાછા result;
}

// ggml_conv_1d_ph

સંરચના ggml_tensor* ggml_conv_1d_ph(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s,
        પૂર્ણાંક                   d) {
    પાછા ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

સંરચના ggml_tensor * ggml_conv_1d_dw(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   d0) {
    સંરચના ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    સંરચના ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    સંરચના ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    પાછા result;
}

// ggml_conv_1d_dw_ph

સંરચના ggml_tensor * ggml_conv_1d_dw_ph(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   d0) {
    પાછા ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

સંરચના ggml_tensor * ggml_col2im_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   oc,
        પૂર્ણાંક                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    અચલ int64_t K_OC = a->ne[0];
    અચલ int64_t T_in = a->ne[1];
    અચલ int64_t K = K_OC / oc;
    અચલ int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    અચલ int64_t ne[4] = { T_out, oc, 1, 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    પાછા result;
}

// ggml_conv_transpose_1d

સ્થિર int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, પૂર્ણાંક s, પૂર્ણાંક p, પૂર્ણાંક d) {
    પાછા (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API સંરચના ggml_tensor * ggml_conv_transpose_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    અચલ int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
સંરચના ggml_tensor * ggml_conv_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   d0,
        પૂર્ણાંક                   d1) {
    સંરચના ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    સંરચના ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    પાછા result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
સંરચના ggml_tensor * ggml_im2col_3d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        int64_t               IC,
        પૂર્ણાંક                   s0, // stride width
        પૂર્ણાંક                   s1, // stride height
        પૂર્ણાંક                   s2, // stride depth
        પૂર્ણાંક                   p0, // padding width
        પૂર્ણાંક                   p1, // padding height
        પૂર્ણાંક                   p2, // padding depth
        પૂર્ણાંક                   d0, // dilation width
        પૂર્ણાંક                   d1, // dilation height
        પૂર્ણાંક                   d2, // dilation depth
        ગણના ggml_type        dst_type) {
    અચલ int64_t N = b->ne[3] / IC;
    અચલ int64_t ID = b->ne[2];
    અચલ int64_t IH = b->ne[1];
    અચલ int64_t IW = b->ne[0];

    અચલ int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    અચલ int64_t KD = a->ne[2];
    અચલ int64_t KH = a->ne[1];
    અચલ int64_t KW = a->ne[0];
    અચલ int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    અચલ int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    અચલ int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    અચલ int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
સંરચના ggml_tensor * ggml_conv_3d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        int64_t               IC,
        પૂર્ણાંક                   s0, // stride width
        પૂર્ણાંક                   s1, // stride height
        પૂર્ણાંક                   s2, // stride depth
        પૂર્ણાંક                   p0, // padding width
        પૂર્ણાંક                   p1, // padding height
        પૂર્ણાંક                   p2, // padding depth
        પૂર્ણાંક                   d0, // dilation width
        પૂર્ણાંક                   d1, // dilation height
        પૂર્ણાંક                   d2  // dilation depth
        ) {
    સંરચના ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    સંરચના ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    પાછા result;
}

// ggml_conv_2d_sk_p0

સંરચના ggml_tensor * ggml_conv_2d_sk_p0(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

સંરચના ggml_tensor * ggml_conv_2d_s1_ph(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    પાછા ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

સંરચના ggml_tensor * ggml_conv_2d_dw(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   d0,
        પૂર્ણાંક                   d1) {
    સંરચના ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    સંરચના ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    સંરચના ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    સંરચના ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    પાછા result;
}

// ggml_conv_2d_dw_direct

સંરચના ggml_tensor * ggml_conv_2d_dw_direct(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   stride0,
        પૂર્ણાંક                   stride1,
        પૂર્ણાંક                   pad0,
        પૂર્ણાંક                   pad1,
        પૂર્ણાંક                   dilation0,
        પૂર્ણાંક                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    જો (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        અચલ int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    પાછા result;
}

// ggml_conv_2d_direct

સંરચના ggml_tensor * ggml_conv_2d_direct(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        સંરચના ggml_tensor  * b,   // input data [W, H, C, N]
        પૂર્ણાંક                   s0,  // stride dimension 0
        પૂર્ણાંક                   s1,  // stride dimension 1
        પૂર્ણાંક                   p0,  // padding dimension 0
        પૂર્ણાંક                   p1,  // padding dimension 1
        પૂર્ણાંક                   d0,  // dilation dimension 0
        પૂર્ણાંક                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_conv_3d_direct

સંરચના ggml_tensor * ggml_conv_3d_direct(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        પૂર્ણાંક                   s2,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   p2,
        પૂર્ણાંક                   d0,
        પૂર્ણાંક                   d1,
        પૂર્ણાંક                   d2,
        પૂર્ણાંક                   c,
        પૂર્ણાંક                   n,
        પૂર્ણાંક                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    પાછા result;
}

// ggml_conv_transpose_2d_p0

સ્થિર int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, પૂર્ણાંક s, પૂર્ણાંક p) {
    પાછા (ins - 1) * s - 2 * p + ks;
}

સંરચના ggml_tensor * ggml_conv_transpose_2d_p0(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        પૂર્ણાંક                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    અચલ int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    સંરચના ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_pool_*

સ્થિર int64_t ggml_calc_pool_output_size(int64_t ins, પૂર્ણાંક ks, પૂર્ણાંક s, દશાંશ p) {
    પાછા (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

સંરચના ggml_tensor * ggml_pool_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_op_pool     op,
        પૂર્ણાંક                   k0,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   p0) {
    અચલ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    પાછા result;
}

// ggml_pool_2d

સંરચના ggml_tensor * ggml_pool_2d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_op_pool     op,
        પૂર્ણાંક                   k0,
        પૂર્ણાંક                   k1,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        દશાંશ                 p0,
        દશાંશ                 p1) {
    સંરચના ggml_tensor * result;
    અચલ int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_pool_2d_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * af,
        ગણના ggml_op_pool     op,
        પૂર્ણાંક                   k0,
        પૂર્ણાંક                   k1,
        પૂર્ણાંક                   s0,
        પૂર્ણાંક                   s1,
        દશાંશ                 p0,
        દશાંશ                 p1) {
    સંરચના ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    પાછા result;
}

// ggml_upscale / ggml_interpolate

સ્થિર સંરચના ggml_tensor * ggml_interpolate_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_ગણતરી);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_upscale(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   scale_factor,
        ગણના ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    પાછા ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

સંરચના ggml_tensor * ggml_upscale_ext(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   ne0,
        પૂર્ણાંક                   ne1,
        પૂર્ણાંક                   ne2,
        પૂર્ણાંક                   ne3,
        ગણના ggml_scale_mode  mode) {
    પાછા ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

સંરચના ggml_tensor * ggml_interpolate(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    પાછા ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

સંરચના ggml_tensor * ggml_pad(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   p2,
        પૂર્ણાંક                   p3) {
    પાછા ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

સંરચના ggml_tensor * ggml_pad_circular(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1,
        પૂર્ણાંક                   p2,
        પૂર્ણાંક                   p3) {
    પાછા ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

સંરચના ggml_tensor * ggml_pad_ext(
            સંરચના ggml_context * ctx,
            સંરચના ggml_tensor  * a,
            પૂર્ણાંક                  lp0,
            પૂર્ણાંક                  rp0,
            પૂર્ણાંક                  lp1,
            પૂર્ણાંક                  rp1,
            પૂર્ણાંક                  lp2,
            પૂર્ણાંક                  rp2,
            પૂર્ણાંક                  lp3,
            પૂર્ણાંક                  rp3
            ) {
    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    પાછા result;
}

// ggml_pad_ext_circular

સંરચના ggml_tensor * ggml_pad_ext_circular(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                  lp0,
        પૂર્ણાંક                  rp0,
        પૂર્ણાંક                  lp1,
        પૂર્ણાંક                  rp1,
        પૂર્ણાંક                  lp2,
        પૂર્ણાંક                  rp2,
        પૂર્ણાંક                  lp3,
        પૂર્ણાંક                  rp3
        ) {
    સંરચના ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    પાછા result;
}

// ggml_pad_reflect_1d

સંરચના ggml_tensor * ggml_pad_reflect_1d(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   p0,
        પૂર્ણાંક                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    પાછા result;
}

// ggml_roll

સંરચના ggml_tensor * ggml_roll(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   shift0,
        પૂર્ણાંક                   shift1,
        પૂર્ણાંક                   shift2,
        પૂર્ણાંક                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    પાછા result;
}

// ggml_timestep_embedding

સંરચના ggml_tensor * ggml_timestep_embedding(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * timesteps,
        પૂર્ણાંક                   dim,
        પૂર્ણાંક                   max_period) {

    સંરચના ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    પાછા result;
}

// ggml_tri

સંરચના ggml_tensor * ggml_tri(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    ગણના ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    પાછા result;
}

// ggml_fill

સ્થિર સંરચના ggml_tensor * ggml_fill_impl(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    દશાંશ                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_fill(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    દશાંશ                 c) {
    પાછા ggml_fill_impl(ctx, a, c, false);
}

સંરચના ggml_tensor * ggml_fill_inplace(
    સંરચના ggml_context * ctx,
    સંરચના ggml_tensor  * a,
    દશાંશ                 c) {
    પાછા ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

સંરચના ggml_tensor * ggml_argsort(
        સંરચના ggml_context  * ctx,
        સંરચના ggml_tensor   * a,
        ગણના ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    પાછા result;
}

// ggml_argsort_top_k

સંરચના ggml_tensor * ggml_argsort_ટોચ_k(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    સંરચના ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    પાછા result;
}

// ggml_top_k

સંરચના ggml_tensor * ggml_ટોચ_k(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_ટોચ_K;
    result->src[0] = a;

    પાછા result;
}

// ggml_arange

સંરચના ggml_tensor * ggml_arange(
        સંરચના ggml_context * ctx,
        દશાંશ                 start,
        દશાંશ                 stop,
        દશાંશ                 step) {
    GGML_ASSERT(stop > start);

    અચલ int64_t steps = (int64_t) ceilf((stop - start) / step);

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    પાછા result;
}

// ggml_flash_attn_ext

સંરચના ggml_tensor * ggml_flash_attn_ext(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * q,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * mask,
        દશાંશ                 scale,
        દશાંશ                 max_bias,
        દશાંશ                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    જો (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    જો (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    દશાંશ params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    પાછા result;
}


શૂન્ય ggml_flash_attn_ext_set_prec(
        સંરચના ggml_tensor * a,
        ગણના ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    અચલ int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

ગણના ggml_prec ggml_flash_attn_ext_get_prec(
        અચલ સંરચના ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    અચલ int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    પાછા (ગણના ggml_prec) prec_i32;
}

શૂન્ય ggml_flash_attn_ext_set_n_kv_max(
        સંરચના ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

શૂન્ય ggml_flash_attn_ext_add_sinks(
        સંરચના ggml_tensor * a,
        સંરચના ggml_tensor * sinks) {
    જો (!sinks) {
        a->src[4] = ખાલી;
        પાછા;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ખાલી);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

સંરચના ggml_tensor * ggml_flash_attn_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * q,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    અચલ int64_t     D = q->ne[0];
    અચલ int64_t     N = q->ne[1];
    અચલ int64_t     M = k->ne[1];
    અચલ int64_t   ne2 = q->ne[2];
    અચલ int64_t   ne3 = q->ne[3];
    અચલ int64_t kvne2 = k->ne[2];

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
    અચલ int64_t elem_q = ggml_nelements(q);
    અચલ int64_t elem_k = ggml_nelements(k);
    અચલ int64_t elem_v = ggml_nelements(v);

    ગણના ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    અચલ size_t tsize = ggml_type_size(result_type);

    અચલ size_t offs_q = 0;
    અચલ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    અચલ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    અચલ size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    અચલ size_t nelements = (end + tsize - 1)/tsize;

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, કદ(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    પાછા result;
}

// ggml_ssm_conv

સંરચના ggml_tensor * ggml_ssm_conv(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * sx,
        સંરચના ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    અચલ int64_t d_conv  = c->ne[0];
    અચલ int64_t d_inner = c->ne[1];
    અચલ int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    અચલ int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    સંરચના ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    પાછા result;
}

// ggml_ssm_scan

સંરચના ggml_tensor * ggml_ssm_scan(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * s,
        સંરચના ggml_tensor  * x,
        સંરચના ggml_tensor  * dt,
        સંરચના ggml_tensor  * A,
        સંરચના ggml_tensor  * B,
        સંરચના ggml_tensor  * C,
        સંરચના ggml_tensor  * ids,
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
        અચલ int64_t d_state      = s->ne[0];
        અચલ int64_t head_dim     = x->ne[0];
        અચલ int64_t n_head       = x->ne[1];
        અચલ int64_t n_seq_tokens = x->ne[2];
        અચલ int64_t n_seqs       = x->ne[3];

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

        જો (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    પાછા result;
}

// ggml_win_part

સંરચના ggml_tensor * ggml_win_part(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    અચલ પૂર્ણાંક px = (w - a->ne[1]%w)%w;
    અચલ પૂર્ણાંક py = (w - a->ne[2]%w)%w;

    અચલ પૂર્ણાંક npx = (px + a->ne[1])/w;
    અચલ પૂર્ણાંક npy = (py + a->ne[2])/w;
    અચલ પૂર્ણાંક np  = npx*npy;

    અચલ int64_t ne[4] = { a->ne[0], w, w, np, };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    પાછા result;
}

// ggml_win_unpart

સંરચના ggml_tensor * ggml_win_unpart(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   w0,
        પૂર્ણાંક                   h0,
        પૂર્ણાંક                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    અચલ int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, કદ(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    પાછા result;
}

// ggml_get_rel_pos

સંરચના ggml_tensor * ggml_get_rel_pos(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        પૂર્ણાંક                   qh,
        પૂર્ણાંક                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    અચલ int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    પાછા result;
}

// ggml_add_rel_pos

સ્થિર સંરચના ggml_tensor * ggml_add_rel_pos_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * pw,
        સંરચના ggml_tensor  * ph,
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

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    પાછા result;
}

સંરચના ggml_tensor * ggml_add_rel_pos(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * pw,
        સંરચના ggml_tensor  * ph) {
    પાછા ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

સંરચના ggml_tensor * ggml_add_rel_pos_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * pw,
        સંરચના ggml_tensor  * ph) {
    પાછા ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

સંરચના ggml_tensor * ggml_rwkv_wkv6(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * r,
        સંરચના ggml_tensor  * tf,
        સંરચના ggml_tensor  * td,
        સંરચના ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    અચલ int64_t S = k->ne[0];
    અચલ int64_t H = k->ne[1];
    અચલ int64_t n_tokens = k->ne[2];
    અચલ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    અચલ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    પાછા result;
}

// ggml_gated_linear_attn

સંરચના ggml_tensor * ggml_gated_linear_attn(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * q,
        સંરચના ggml_tensor  * g,
        સંરચના ggml_tensor  * state,
        દશાંશ scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    અચલ int64_t S = k->ne[0];
    અચલ int64_t H = k->ne[1];
    અચલ int64_t n_tokens = k->ne[2];
    અચલ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    અચલ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    પાછા result;
}

// ggml_rwkv_wkv7

સંરચના ggml_tensor * ggml_rwkv_wkv7(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * r,
        સંરચના ggml_tensor  * w,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    અચલ int64_t S = k->ne[0];
    અચલ int64_t H = k->ne[1];
    અચલ int64_t n_tokens = k->ne[2];
    અચલ int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    અચલ int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    પાછા result;
}

// ggml_unary

સ્થિર સંરચના ggml_tensor * ggml_unary_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_unary(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_unary_op    op) {
    પાછા ggml_unary_impl(ctx, a, op, false);
}

સંરચના ggml_tensor * ggml_unary_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        ગણના ggml_unary_op    op) {
    પાછા ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

સ્થિર સંરચના ggml_tensor * ggml_map_custom1_impl(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        અચલ  ggml_custom1_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    સંરચના ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, કદ(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    પાછા result;
}

સંરચના ggml_tensor * ggml_map_custom1(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        અચલ  ggml_custom1_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

સંરચના ggml_tensor * ggml_map_custom1_inplace(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        અચલ  ggml_custom1_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

સ્થિર સંરચના ggml_tensor * ggml_map_custom2_impl(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        અચલ  ggml_custom2_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    સંરચના ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, કદ(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

સંરચના ggml_tensor * ggml_map_custom2(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        અચલ  ggml_custom2_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

સંરચના ggml_tensor * ggml_map_custom2_inplace(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        અચલ  ggml_custom2_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

સ્થિર સંરચના ggml_tensor * ggml_map_custom3_impl(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        સંરચના ggml_tensor       * c,
        અચલ  ggml_custom3_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    સંરચના ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    સંરચના ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, કદ(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    પાછા result;
}

સંરચના ggml_tensor * ggml_map_custom3(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        સંરચના ggml_tensor       * c,
        અચલ  ggml_custom3_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

સંરચના ggml_tensor * ggml_map_custom3_inplace(
        સંરચના ggml_context      * ctx,
        સંરચના ggml_tensor       * a,
        સંરચના ggml_tensor       * b,
        સંરચના ggml_tensor       * c,
        અચલ  ggml_custom3_op_t   fun,
        પૂર્ણાંક                        n_tasks,
        શૂન્ય                     * userdata) {
    પાછા ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

સંરચના ggml_tensor * ggml_custom_4d(
        સંરચના ggml_context * ctx,
        ગણના ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        સંરચના ggml_tensor ** args,
        પૂર્ણાંક                   n_args,
        ggml_custom_op_t      fun,
        પૂર્ણાંક                   n_tasks,
        શૂન્ય                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    સંરચના ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, કદ(params));

    result->op = GGML_OP_CUSTOM;
    માટે (પૂર્ણાંક i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    પાછા result;
}

સંરચના ggml_tensor * ggml_custom_inplace(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor ** args,
        પૂર્ણાંક                   n_args,
        ggml_custom_op_t      fun,
        પૂર્ણાંક                   n_tasks,
        શૂન્ય                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);

    સંરચના ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, કદ(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    માટે (પૂર્ણાંક i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    પાછા result;
}
// ggml_cross_entropy_loss

સંરચના ggml_tensor * ggml_cross_entropy_loss(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_cross_entropy_loss_back

સંરચના ggml_tensor * ggml_cross_entropy_loss_back(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        સંરચના ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    સંરચના ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    પાછા result;
}

// opt_step_adamw

સંરચના ggml_tensor * ggml_opt_step_adamw(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * grad,
        સંરચના ggml_tensor  * m,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    પાછા result;
}

// opt_step_sgd

સંરચના ggml_tensor * ggml_opt_step_sgd(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * grad,
        સંરચના ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    સંરચના ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    પાછા result;
}

// solve_tri

સંરચના ggml_tensor * ggml_solve_tri(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * a,
        સંરચના ggml_tensor  * b,
        bool                  ડાબું,
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

    GGML_ASSERT(lower && ડાબું && !uni); // TODO: support other variants

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    પાછા result;
}

// ggml_gated_delta_net

સંરચના ggml_tensor * ggml_gated_delta_net(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * q,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * v,
        સંરચના ggml_tensor  * g,
        સંરચના ggml_tensor  * beta,
        સંરચના ggml_tensor  * state,
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

    અચલ int64_t S_v      = v->ne[0];
    અચલ int64_t H        = v->ne[1];
    અચલ int64_t n_tokens = v->ne[2];
    અચલ int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    અચલ int64_t state_rows = K * S_v * n_seqs;
    અચલ int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    પાછા result;
}

// ggml_lightning_indexer

સંરચના ggml_tensor * ggml_lightning_indexer(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * q,
        સંરચના ggml_tensor  * k,
        સંરચના ggml_tensor  * weights,
        સંરચના ggml_tensor  * mask) {

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
    સંરચના ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    પાછા result;
}

// ggml_dsv4_hc_comb

સંરચના ggml_tensor * ggml_dsv4_hc_comb(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * mixes,
        સંરચના ggml_tensor  * scale,
        સંરચના ggml_tensor  * base,
        દશાંશ                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    અચલ int64_t hc_mix_dim = mixes->ne[0];
    અચલ int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    માટે (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        જો ((2 + i)*i == hc_mix_dim) {
            hc = i;
            તોડો;
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

    સંરચના ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    પાછા result;
}

// ggml_dsv4_hc_pre

સ્થિર સંરચના ggml_tensor * ggml_dsv4_hc_pre_impl(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * x,
        સંરચના ggml_tensor  * weights,
        દશાંશ                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t hc       = x->ne[1];
    અચલ int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    જો (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } નહીંતર {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    સંરચના ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    પાછા result;
}

સંરચના ggml_tensor * ggml_dsv4_hc_pre(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * x,
        સંરચના ggml_tensor  * weights) {
    પાછા ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

સંરચના ggml_tensor * ggml_dsv4_hc_pre_gated(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * x,
        સંરચના ggml_tensor  * gate,
        દશાંશ                 scale) {
    પાછા ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

સંરચના ggml_tensor * ggml_dsv4_hc_post(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * x,
        સંરચના ggml_tensor  * residual,
        સંરચના ggml_tensor  * post,
        સંરચના ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t n_tokens = x->ne[1];
    અચલ int64_t hc       = residual->ne[1];

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

    જો (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    સંરચના ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    પાછા result;
}

// ggml_top_k_mask

સંરચના ggml_tensor * ggml_ટોચ_k_mask(
        સંરચના ggml_context * ctx,
        સંરચના ggml_tensor  * mask,
        સંરચના ggml_tensor  * ટોચ_k,
        સંરચના ggml_tensor  * score,
        સંરચના ggml_tensor  * blocks,
        સંરચના ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(ટોચ_k->type == GGML_TYPE_I32);
    GGML_ASSERT(ટોચ_k->ne[1] % mask->ne[1] == 0 && ટોચ_k->ne[2] % mask->ne[2] == 0 && ટોચ_k->ne[3] % mask->ne[3] == 0);

    જો (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == ટોચ_k->ne[1] && score->ne[2] == ટોચ_k->ne[2] && score->ne[3] == ટોચ_k->ne[3]);
    }

    જો (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    જો (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == ટોચ_k->ne[1] && extra->ne[2] == ટોચ_k->ne[2] && extra->ne[3] == ટોચ_k->ne[3]);
    }

    સંરચના ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], ટોચ_k->ne[1], ટોચ_k->ne[2], ટોચ_k->ne[3]);

    result->op     = GGML_OP_ટોચ_K_MASK;
    result->src[0] = mask;
    result->src[1] = ટોચ_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    પાછા result;
}

////////////////////////////////////////////////////////////////////////////////

સંરચના ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    સંરચના ggml_hash_set result;
    result.size = size;
    result.ચાવીઓ = GGML_MALLOC(કદ(સંરચના ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), કદ(ggml_bitset_t));
    પાછા result;
}

શૂન્ય ggml_hash_set_reset(સંરચના ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, કદ(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

શૂન્ય ggml_hash_set_free(સંરચના ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->ચાવીઓ);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    સ્થિર અચલ size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    સ્થિર અચલ size_t n_primes = કદ(primes)/કદ(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    જ્યારે (l < r) {
        size_t m = (l + r)/2;
        જો (primes[m] < min_sz) {
            l = m + 1;
        } નહીંતર {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    પાછા sz;
}

સંરચના hash_map {
    સંરચના ggml_hash_set set;
    સંરચના ggml_tensor ** vals;
};

સ્થિર સંરચના hash_map * ggml_new_hash_map(size_t size) {
    સંરચના hash_map * result = GGML_MALLOC(કદ(સંરચના hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, કદ(સંરચના ggml_tensor *));
    પાછા result;
}

સ્થિર શૂન્ય ggml_hash_map_free(સંરચના hash_map * map) {
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

સ્થિર શૂન્ય ggml_add_or_set(
        સંરચના ggml_context * ctx,
        સંરચના ggml_cgraph  * cgraph,
        size_t                isrc,
        સંરચના ggml_tensor  * tensor) {
    સંરચના ggml_tensor * src = cgraph->visited_hash_set.ચાવીઓ[isrc];
    GGML_ASSERT(src);
    જો (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } નહીંતર {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

સ્થિર શૂન્ય ggml_acc_or_set(
        સંરચના ggml_context * ctx,
        સંરચના ggml_cgraph  * cgraph,
        size_t                isrc,
        સંરચના ggml_tensor  * tensor,
        અચલ  size_t         nb1,
        અચલ  size_t         nb2,
        અચલ  size_t         nb3,
        અચલ  size_t         offset) {
    સંરચના ggml_tensor * src = cgraph->visited_hash_set.ચાવીઓ[isrc];
    GGML_ASSERT(src);
    જો (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } નહીંતર {
        સંરચના ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.ચાવીઓ[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

સ્થિર શૂન્ય ggml_add1_or_set(
        સંરચના ggml_context * ctx,
        સંરચના ggml_cgraph  * cgraph,
        size_t                isrc,
        સંરચના ggml_tensor  * tensor) {
    સંરચના ggml_tensor * src = cgraph->visited_hash_set.ચાવીઓ[isrc];
    GGML_ASSERT(src);
    જો (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } નહીંતર {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

સ્થિર શૂન્ય ggml_sub_or_set(
        સંરચના ggml_context * ctx,
        સંરચના ggml_cgraph  * cgraph,
        size_t                isrc,
        સંરચના ggml_tensor  * tensor) {
    સંરચના ggml_tensor * src = cgraph->visited_hash_set.ચાવીઓ[isrc];
    GGML_ASSERT(src);
    જો (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } નહીંતર {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

સ્થિર શૂન્ય ggml_compute_backward(
        સંરચના ggml_context * ctx, સંરચના ggml_cgraph * cgraph, પૂર્ણાંક i, અચલ bool * grads_needed) {
    સંરચના ggml_tensor * tensor = cgraph->nodes[i];
    સંરચના ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    જો (!grad) {
        પાછા;
    }

    સંરચના ggml_tensor * src0 = tensor->src[0];
    સંરચના ggml_tensor * src1 = tensor->src[1];
    સંરચના ggml_tensor * src2 = tensor->src[2];
    સંરચના ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    અચલ size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    અચલ size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    અચલ size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    અચલ bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    અચલ bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    અચલ bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    પસંદગી (tensor->op) {
        સ્થિતિ GGML_OP_DUP: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } તોડો;
        સ્થિતિ GGML_OP_ADD: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            જો (src1_needs_grads) {
                સંરચના ggml_tensor * tmp = grad;
                જો (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } તોડો;
        સ્થિતિ GGML_OP_ADD1: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            જો (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } તોડો;
        સ્થિતિ GGML_OP_ACC: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            જો (src1_needs_grads) {
                અચલ size_t nb1    = ((int32_t *) tensor->op_params)[0];
                અચલ size_t nb2    = ((int32_t *) tensor->op_params)[1];
                અચલ size_t nb3    = ((int32_t *) tensor->op_params)[2];
                અચલ size_t offset = ((int32_t *) tensor->op_params)[3];

                સંરચના ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SUB: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            જો (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } તોડો;
        સ્થિતિ GGML_OP_MUL: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            જો (src1_needs_grads) {
                સંરચના ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                જો (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } તોડો;
        સ્થિતિ GGML_OP_DIV: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            જો (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SQR: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SQRT: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } તોડો;
        સ્થિતિ GGML_OP_LOG: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SIN: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } તોડો;
        સ્થિતિ GGML_OP_COS: {
            જો (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SUM: {
            જો (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_MEAN: {
            જો (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } તોડો;
        સ્થિતિ GGML_OP_REPEAT: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_REPEAT_BACK: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_RMS_NORM: {
            જો (src0_needs_grads) {
                દશાંશ eps;
                memcpy(&eps, tensor->op_params, કદ(દશાંશ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } તોડો;
        સ્થિતિ GGML_OP_MUL_MAT: {
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

            જો (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                સંરચના ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                જો (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    અચલ int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    અચલ size_t nb2 = tmp->nb[2] * nr2;
                    અચલ size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            જો (src1_needs_grads) {
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
        } તોડો;
        સ્થિતિ GGML_OP_SCALE: {
            જો (src0_needs_grads) {
                દશાંશ s;
                memcpy(&s, tensor->op_params, કદ(દશાંશ));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SET: {
            અચલ size_t nb1    = ((અચલ int32_t *) tensor->op_params)[0];
            અચલ size_t nb2    = ((અચલ int32_t *) tensor->op_params)[1];
            અચલ size_t nb3    = ((અચલ int32_t *) tensor->op_params)[2];
            અચલ size_t offset = ((અચલ int32_t *) tensor->op_params)[3];

            સંરચના ggml_tensor * tensor_grad_view = ખાલી;

            જો (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            જો (src0_needs_grads) {
                સંરચના ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            જો (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } તોડો;
        સ્થિતિ GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            જો (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            જો (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } તોડો;
        સ્થિતિ GGML_OP_CONT: {
            // same as cpy
            જો (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_RESHAPE: {
            જો (src0_needs_grads) {
                સંરચના ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } તોડો;
        સ્થિતિ GGML_OP_VIEW: {
            જો (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, કદ(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                જો (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } તોડો;
        સ્થિતિ GGML_OP_PERMUTE: {
            જો (src0_needs_grads) {
                અચલ int32_t * axes = (અચલ int32_t *) tensor->op_params;
                અચલ પૂર્ણાંક axis0 = axes[0] & 0x3;
                અચલ પૂર્ણાંક axis1 = axes[1] & 0x3;
                અચલ પૂર્ણાંક axis2 = axes[2] & 0x3;
                અચલ પૂર્ણાંક axis3 = axes[3] & 0x3;
                પૂર્ણાંક axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } તોડો;
        સ્થિતિ GGML_OP_TRANSPOSE: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } તોડો;
        સ્થિતિ GGML_OP_GET_ROWS: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            જો (src1_needs_grads) {
                // noop
            }
        } તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_INF: {
            જો (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                અચલ પૂર્ણાંક n_past = ((અચલ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_ZERO: {
            જો (src0_needs_grads) {
                અચલ પૂર્ણાંક n_past = ((અચલ int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX: {
            જો (src0_needs_grads) {
                દશાંશ scale    = 1.0f;
                દશાંશ max_bias = 0.0f;

                memcpy(&scale,    (અચલ દશાંશ *) tensor->op_params + 0, કદ(દશાંશ));
                memcpy(&max_bias, (અચલ દશાંશ *) tensor->op_params + 1, કદ(દશાંશ));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } તોડો;
        સ્થિતિ GGML_OP_ROPE: {
            જો (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                અચલ પૂર્ણાંક n_dims     = ((અચલ int32_t *) tensor->op_params)[1];
                અચલ પૂર્ણાંક mode       = ((અચલ int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                અચલ પૂર્ણાંક n_ctx_orig = ((અચલ int32_t *) tensor->op_params)[4];
                દશાંશ freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                પૂર્ણાંક sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (અચલ દશાંશ *) tensor->op_params +  5, કદ(દશાંશ));
                memcpy(&freq_scale,  (અચલ દશાંશ *) tensor->op_params +  6, કદ(દશાંશ));
                memcpy(&ext_factor,  (અચલ દશાંશ *) tensor->op_params +  7, કદ(દશાંશ));
                memcpy(&attn_factor, (અચલ દશાંશ *) tensor->op_params +  8, કદ(દશાંશ));
                memcpy(&beta_fast,   (અચલ દશાંશ *) tensor->op_params +  9, કદ(દશાંશ));
                memcpy(&beta_slow,   (અચલ દશાંશ *) tensor->op_params + 10, કદ(દશાંશ));
                memcpy(&sections,                    tensor->op_params + 11, કદ(sections));

                સંરચના ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } તોડો;
        સ્થિતિ GGML_OP_IM2COL: {
            જો (src1_needs_grads) {
                અચલ int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                અચલ int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                અચલ int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                અચલ int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                અચલ int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                અચલ int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                અચલ bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } તોડો;
        સ્થિતિ GGML_OP_POOL_2D: {
            જો (src0_needs_grads) {
                અચલ ગણના ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                અચલ      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                અચલ      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                અચલ      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                અચલ      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                અચલ      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                અચલ      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } તોડો;
        સ્થિતિ GGML_OP_WIN_PART:
        સ્થિતિ GGML_OP_WIN_UNPART:
        સ્થિતિ GGML_OP_UNARY: {
            પસંદગી (ggml_get_unary_op(tensor)) {
                સ્થિતિ GGML_UNARY_OP_ABS: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_SGN: {
                    // noop
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_NEG: {
                    જો (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_STEP: {
                    // noop
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_RELU: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_SILU: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_EXP: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_EXPM1: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } તોડો;
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS: {
                    જો (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } તોડો;
                મૂળભૂત: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS: {
            જો (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } તોડો;
        સ્થિતિ GGML_OP_GLU: {
            પસંદગી (ggml_get_glu_op(tensor)) {
                સ્થિતિ GGML_GLU_OP_SWIGLU: {
                    જો (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    જો (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } તોડો;
                મૂળભૂત: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } તોડો;
        સ્થિતિ GGML_OP_NONE: {
            // noop
        } તોડો;
        સ્થિતિ GGML_OP_ગણતરી:
        મૂળભૂત: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

સ્થિર size_t ggml_visit_parents_graph(સંરચના ggml_cgraph * cgraph, સંરચના ggml_tensor * ગાંઠ, bool compute) {
    જો (ગાંઠ->op != GGML_OP_NONE && compute) {
        ગાંઠ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    અચલ size_t ગાંઠ_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ);
    GGML_ASSERT(ગાંઠ_hash_pos != GGML_HASHSET_FULL);

    જો (ggml_bitset_get(cgraph->visited_hash_set.used, ગાંઠ_hash_pos)) {
        // already visited

        જો (compute) {
            // update the compute flag regardless
            માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; ++i) {
                સંરચના ggml_tensor * src = ગાંઠ->src[i];
                જો (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        પાછા ગાંઠ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.ચાવીઓ[ગાંઠ_hash_pos] = ગાંઠ;
    ggml_bitset_set(cgraph->visited_hash_set.used, ગાંઠ_hash_pos);
    cgraph->use_counts[ગાંઠ_hash_pos] = 0;

    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; ++i) {
        અચલ પૂર્ણાંક k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_ડાબું_TO_જમણું) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_જમણું_TO_ડાબું) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        સંરચના ggml_tensor * src = ગાંઠ->src[k];
        જો (src) {
            અચલ size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    જો (ગાંઠ->op == GGML_OP_NONE && !(ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        જો (strlen(ગાંઠ->name) == 0) {
            ggml_format_name(ગાંઠ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ગાંઠ;
        cgraph->n_leafs++;
    } નહીંતર {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        જો (strlen(ગાંઠ->name) == 0) {
            ggml_format_name(ગાંઠ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ગાંઠ;
        cgraph->n_nodes++;
    }

    પાછા ગાંઠ_hash_pos;
}

સ્થિર શૂન્ય ggml_build_forward_impl(સંરચના ggml_cgraph * cgraph, સંરચના ggml_tensor * tensor, bool expand, bool compute) {
    જો (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    અચલ પૂર્ણાંક n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    અચલ પૂર્ણાંક n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    જો (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

સંરચના ggml_tensor * ggml_build_forward_select(
        સંરચના ggml_cgraph  * cgraph,
        સંરચના ggml_tensor ** tensors,
        પૂર્ણાંક                   n_tensors,
        પૂર્ણાંક                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    માટે (પૂર્ણાંક i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    પાછા tensors[idx];
}

શૂન્ય ggml_build_forward_expand(સંરચના ggml_cgraph * cgraph, સંરચના ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

શૂન્ય ggml_build_forward_order(સંરચના ggml_cgraph * cgraph, સંરચના ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

શૂન્ય ggml_build_backward_expand(
        સંરચના ggml_context *  ctx,
        સંરચના ggml_cgraph  *  cgraph,
        સંરચના ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    અચલ પૂર્ણાંક n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*કદ(સંરચના ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*કદ(સંરચના ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, કદ(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        માટે (પૂર્ણાંક i = 0; i < n_nodes_f; ++i) {
            સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];
            any_params = any_params || (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    માટે (પૂર્ણાંક i = 0; i < n_nodes_f; ++i) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        જો (ગાંઠ->type == GGML_TYPE_I32) {
            ચાલુ;
        }

        bool ગાંઠ_needs_grad = (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM) || (ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        પસંદગી (ગાંઠ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            સ્થિતિ GGML_OP_IM2COL:      // only used for its shape
            સ્થિતિ GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                તોડો;
            સ્થિતિ GGML_OP_UNARY: {
                અચલ ગણના ggml_unary_op uop = ggml_get_unary_op(ગાંઠ);
                // SGN and STEP unary ops are piecewise constant
                જો (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } તોડો;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            સ્થિતિ GGML_OP_CPY:           // gradients in CPY target are irrelevant
            સ્થિતિ GGML_OP_GET_ROWS:      // row indices not differentiable
            સ્થિતિ GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            સ્થિતિ GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                તોડો;

            મૂળભૂત:
                તોડો;
        }
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; ++j) {
            જો (!ગાંઠ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ->src[j])]) {
                ચાલુ;
            }
            GGML_ASSERT(ગાંઠ->src[j]->type == GGML_TYPE_F32 || ગાંઠ->src[j]->type == GGML_TYPE_F16);
            ગાંઠ_needs_grad = true;
            તોડો;
        }
        જો (!ગાંઠ_needs_grad) {
            ચાલુ;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ગાંઠ->view_src || ગાંઠ->op == GGML_OP_CPY || ગાંઠ->op == GGML_OP_SET_ROWS || ગાંઠ->op == GGML_OP_VIEW ||
            ગાંઠ->op == GGML_OP_RESHAPE || ગાંઠ->op == GGML_OP_PERMUTE || ગાંઠ->op == GGML_OP_TRANSPOSE);

        અચલ size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        જો (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } નહીંતર જો (ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ગાંઠ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    માટે (પૂર્ણાંક i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    મુક્ત(grads_needed);
}

સ્થિર શૂન્ય * incr_ptr_aligned(શૂન્ય ** p, size_t size, size_t align) {
    શૂન્ય * ptr = *p;
    ptr = (શૂન્ય *) GGML_PAD((uintptr_t) ptr, align);
    *p = (શૂન્ય *) ((અક્ષર *) ptr + size);
    પાછા ptr;
}

સ્થિર size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    શૂન્ય * p = (અક્ષર *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, કદ(સંરચના ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * કદ(int32_t), કદ(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)); // hash keys
    જો (grads) {
        incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * કદ(ggml_bitset_t), કદ(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    પાછા nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    પાછા GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(શૂન્ય) {
    પાછા ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

સંરચના ggml_cgraph * ggml_new_graph_custom(સંરચના ggml_context * ctx, size_t size, bool grads) {
    અચલ size_t obj_size = ggml_graph_nbytes(size, grads);
    સંરચના ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    સંરચના ggml_cgraph * cgraph = (સંરચના ggml_cgraph *) ((અક્ષર *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    શૂન્ય * p = cgraph + 1;

    સંરચના ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *));
    સંરચના ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * કદ(int32_t), કદ(int32_t));
    સંરચના ggml_tensor ** hash_ચાવીઓ_ptr  =         incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *));
    સંરચના ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)) : ખાલી;
    સંરચના ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * કદ(સંરચના ggml_tensor *), કદ(સંરચના ggml_tensor *)) : ખાલી;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * કદ(ggml_bitset_t), કદ(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    પ્રતિજ્ઞા(obj_size == (size_t)((અક્ષર *)p - (અક્ષર *)cgraph));

    *cgraph = (સંરચના ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_ચાવીઓ_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_ડાબું_TO_જમણું,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    જો (grads) {
        memset(cgraph->grads,     0, hash_size*કદ(સંરચના ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*કદ(સંરચના ggml_tensor *));
    }

    પાછા cgraph;
}

સંરચના ggml_cgraph * ggml_new_graph(સંરચના ggml_context * ctx) {
    પાછા ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

સંરચના ggml_cgraph ggml_graph_view(સંરચના ggml_cgraph * cgraph0, પૂર્ણાંક i0, પૂર્ણાંક i1) {
    સંરચના ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ખાલી, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ખાલી,
        /*.leafs            =*/ ખાલી,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    પાછા cgraph;
}

શૂન્ય ggml_graph_cpy(સંરચના ggml_cgraph * src, સંરચના ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    માટે (પૂર્ણાંક i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    માટે (પૂર્ણાંક i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    માટે (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        જો (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_ઉમેરો(&dst->visited_hash_set, src->visited_hash_set.ચાવીઓ[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    જો (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*કદ(સંરચના ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*કદ(સંરચના ggml_tensor *));
    }
    જો (src->grads) {
        GGML_ASSERT(dst->grads     != ખાલી);
        GGML_ASSERT(dst->grad_accs != ખાલી);
        માટે (પૂર્ણાંક i = 0; i < src->n_nodes; ++i) {
            અચલ size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            અચલ size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

સંરચના ggml_cgraph * ggml_graph_dup(સંરચના ggml_context * ctx, સંરચના ggml_cgraph * cgraph, bool force_grads) {
    સંરચના ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    પાછા result;
}

સંરચના ggml_tensor * ggml_set_zero(સંરચના ggml_tensor * tensor) {
    જો (ggml_is_empty(tensor)) {
        પાછા tensor;
    }
    જો (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } નહીંતર {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    પાછા tensor;
}

શૂન્ય ggml_graph_reset(સંરચના ggml_cgraph * cgraph) {
    જો (!cgraph) {
        પાછા;
    }
    GGML_ASSERT(cgraph->grads != ખાલી);

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ     = cgraph->nodes[i];
        સંરચના ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, ગાંઠ);

        જો (ગાંઠ->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(ગાંઠ->src[2]);
            ggml_set_zero(ગાંઠ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        જો (grad_acc) {
            જો (ગાંઠ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                અચલ દશાંશ onef = 1.0f;
                જો (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, કદ(દશાંશ));
                } નહીંતર {
                    GGML_ASSERT(grad_acc->data);
                    *((દશાંશ *) grad_acc->data) = onef;
                }
            } નહીંતર {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

શૂન્ય ggml_graph_clear(સંરચના ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

પૂર્ણાંક ggml_graph_size(સંરચના ggml_cgraph * cgraph) {
    પાછા cgraph->size;
}

સંરચના ggml_tensor * ggml_graph_ગાંઠ(સંરચના ggml_cgraph * cgraph, પૂર્ણાંક i) {
    જો (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        પાછા cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    પાછા cgraph->nodes[i];
}

સંરચના ggml_tensor ** ggml_graph_nodes(સંરચના ggml_cgraph * cgraph) {
    પાછા cgraph->nodes;
}

પૂર્ણાંક ggml_graph_n_nodes(સંરચના ggml_cgraph * cgraph) {
    પાછા cgraph->n_nodes;
}

શૂન્ય ggml_graph_add_ગાંઠ(સંરચના ggml_cgraph * cgraph, સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

સંરચના ggml_tensor * ggml_graph_get_tensor(અચલ સંરચના ggml_cgraph * cgraph, અચલ અક્ષર * name) {
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_leafs; i++) {
        સંરચના ggml_tensor * leaf = cgraph->leafs[i];

        જો (strcmp(leaf->name, name) == 0) {
            પાછા leaf;
        }
    }

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        જો (strcmp(ગાંઠ->name, name) == 0) {
            પાછા ગાંઠ;
        }
    }

    પાછા ખાલી;
}

સંરચના ggml_tensor * ggml_graph_get_grad(અચલ સંરચના ggml_cgraph * cgraph, અચલ સંરચના ggml_tensor * ગાંઠ) {
    અચલ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ);
    પાછા igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : ખાલી;
}

સંરચના ggml_tensor * ggml_graph_get_grad_acc(અચલ સંરચના ggml_cgraph * cgraph, અચલ સંરચના ggml_tensor * ગાંઠ) {
    અચલ size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ);
    પાછા igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ખાલી;
}

શૂન્ય ggml_graph_print(અચલ સંરચના ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ગાંઠ->ne[0], ગાંઠ->ne[1], ગાંઠ->ne[2],
                ggml_op_name(ગાંઠ->op), (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, ગાંઠ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_leafs; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ગાંઠ->ne[0], ગાંઠ->ne[1],
                ggml_op_name(ગાંઠ->op),
                ggml_get_name(ગાંઠ));
    }

    GGML_LOG_INFO("========================================\n");
}

સ્થિર પૂર્ણાંક ggml_ગાંઠ_list_find_tensor(અચલ સંરચના ggml_cgraph * cgraph,
                                      અચલ પૂર્ણાંક *                idxs,
                                      પૂર્ણાંક                        ગણતરી,
                                      અચલ સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    માટે (પૂર્ણાંક i = 0; i < ગણતરી; ++i) {
        અચલ પૂર્ણાંક ગાંઠ_idx = idxs[i];

        જો (ગાંઠ_idx >= cgraph->n_nodes) {
            પાછા -1;
        }
        જો (cgraph->nodes[ગાંઠ_idx] == tensor) {
            પાછા i;
        }
    }
    પાછા -1;
}

સ્થિર bool ggml_is_constant(અચલ સંરચના ggml_tensor * tensor) {
    પાછા tensor->buffer != ખાલી && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(અચલ સંરચના ggml_cgraph * cgraph,
                                અચલ પૂર્ણાંક *                ગાંઠ_idxs,
                                પૂર્ણાંક                        ગણતરી,
                                અચલ ગણના ggml_op *       ops,
                                અચલ પૂર્ણાંક *                outputs,
                                પૂર્ણાંક                        અંક_outputs) {
    GGML_ASSERT(outputs && અંક_outputs > 0);

    માટે (પૂર્ણાંક i = 0; i < ગણતરી; ++i) {
        જો (ગાંઠ_idxs[i] >= cgraph->n_nodes) {
            પાછા false;
        }

        અચલ સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idxs[i]];

        જો (ગાંઠ->op != ops[i]) {
            પાછા false;
        }

        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            પાછા false;
        }

        જો (ggml_ગાંઠ_list_find_tensor(cgraph, outputs, અંક_outputs, ગાંઠ) != -1) {
            ચાલુ;
        }

        જો (ગાંઠ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            પાછા false;
        }

        પૂર્ણાંક subgraph_uses = 0;
        માટે (પૂર્ણાંક j = i + 1; j < ગણતરી; ++j) {
            અચલ સંરચના ggml_tensor * other_ગાંઠ = cgraph->nodes[ગાંઠ_idxs[j]];
            માટે (પૂર્ણાંક src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                જો (other_ગાંઠ->src[src_idx] == ગાંઠ) {
                    subgraph_uses++;
                }
            }
        }

        જો (subgraph_uses != ggml_ગાંઠ_get_use_ગણતરી(cgraph, ગાંઠ_idxs[i])) {
            પાછા false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        સંરચના ggml_tensor * view_src = ગાંઠ->view_src;
        જ્યારે (view_src) {
            જો (ggml_ગાંઠ_list_find_tensor(cgraph, ગાંઠ_idxs, ગણતરી, view_src) == -1 && !ggml_is_constant(view_src)) {
                પાછા false;
            }
            view_src = view_src->view_src;
        }
    }

    પાછા true;
}

// check if node is part of the graph
સ્થિર bool ggml_graph_find(અચલ સંરચના ggml_cgraph * cgraph, અચલ સંરચના ggml_tensor * ગાંઠ) {
    જો (cgraph == ખાલી) {
        પાછા true;
    }

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        જો (cgraph->nodes[i] == ગાંઠ) {
            પાછા true;
        }
    }

    પાછા false;
}

સ્થિર સંરચના ggml_tensor * ggml_graph_get_parent(અચલ સંરચના ggml_cgraph * cgraph, અચલ સંરચના ggml_tensor * ગાંઠ) {
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * parent = cgraph->nodes[i];
        સંરચના ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        જો (grad == ગાંઠ) {
            પાછા parent;
        }
    }

    પાછા ખાલી;
}

સ્થિર શૂન્ય ggml_graph_dump_dot_ગાંઠ_edge(FILE * fp, અચલ સંરચના ggml_cgraph * gb, સંરચના ggml_tensor * ગાંઠ, સંરચના ggml_tensor * parent, અચલ અક્ષર * label)  {
    સંરચના ggml_tensor * gparent = ggml_graph_get_parent(gb, ગાંઠ);
    સંરચના ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (શૂન્ય *) gparent0 : (શૂન્ય *) parent,
            gparent ? (શૂન્ય *) gparent : (શૂન્ય *) ગાંઠ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

સ્થિર શૂન્ય ggml_graph_dump_dot_leaf_edge(FILE * fp, સંરચના ggml_tensor * ગાંઠ, સંરચના ggml_tensor * parent, અચલ અક્ષર * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (શૂન્ય *) parent,
            (શૂન્ય *) ગાંઠ,
            label);
}

શૂન્ય ggml_graph_dump_dot(અચલ સંરચના ggml_cgraph * gb, અચલ સંરચના ggml_cgraph * cgraph, અચલ અક્ષર * filename) {
    અક્ષર color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    માટે (પૂર્ણાંક i = 0; i < gb->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = gb->nodes[i];
        સંરચના ggml_tensor * grad = ggml_graph_get_grad(gb, ગાંઠ);

        જો (ggml_graph_get_parent(gb, ગાંઠ) != ખાલી) {
            ચાલુ;
        }

        જો (ગાંઠ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, કદ(color), "yellow");
        } નહીંતર જો (grad) {
            જો (ggml_graph_find(cgraph, ગાંઠ)) {
                snprintf(color, કદ(color), "green");
            } નહીંતર {
                snprintf(color, કદ(color), "lightblue");
            }
        } નહીંતર {
            snprintf(color, કદ(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (શૂન્ય *) ગાંઠ, color);

        જો (strlen(ગાંઠ->name) > 0) {
            fprintf(fp, "%s (%s)|", ગાંઠ->name, ggml_type_name(ગાંઠ->type));
        } નહીંતર {
            fprintf(fp, "(%s)|", ggml_type_name(ગાંઠ->type));
        }

        જો (ggml_is_matrix(ગાંઠ)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ગાંઠ->ne[0], ગાંઠ->ne[1], ggml_op_પ્રતીક(ગાંઠ->op));
        } નહીંતર {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ગાંઠ->ne[0], ગાંઠ->ne[1], ગાંઠ->ne[2], ggml_op_પ્રતીક(ગાંઠ->op));
        }

        જો (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_પ્રતીક(grad->op));
        } નહીંતર {
            fprintf(fp, "\"; ]\n");
        }
    }

    માટે (પૂર્ણાંક i = 0; i < gb->n_leafs; i++) {
        સંરચના ggml_tensor * ગાંઠ = gb->leafs[i];

        snprintf(color, કદ(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (શૂન્ય *) ગાંઠ, color);

        જો (strlen(ગાંઠ->name) > 0) {
            fprintf(fp, "%s (%s)|", ગાંઠ->name, ggml_type_name(ગાંઠ->type));
        } નહીંતર {
            fprintf(fp, "(%s)|", ggml_type_name(ગાંઠ->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ગાંઠ->ne[0], ગાંઠ->ne[1]);
        જો (ggml_nelements(ગાંઠ) < 5 && ગાંઠ->data != ખાલી) {
            fprintf(fp, " | (");
            માટે (પૂર્ણાંક j = 0; j < ggml_nelements(ગાંઠ); j++) {
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
                જો (j < ggml_nelements(ગાંઠ) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    માટે (પૂર્ણાંક i = 0; i < gb->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = gb->nodes[i];

        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            જો (ગાંઠ->src[j]) {
                અક્ષર label[16];
                snprintf(label, કદ(label), "src %d", j);
                ggml_graph_dump_dot_ગાંઠ_edge(fp, gb, ગાંઠ, ગાંઠ->src[j], label);
            }
        }
    }

    માટે (પૂર્ણાંક i = 0; i < gb->n_leafs; i++) {
        સંરચના ggml_tensor * ગાંઠ = gb->leafs[i];

        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            જો (ગાંઠ->src[j]) {
                અક્ષર label[16];
                snprintf(label, કદ(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, ગાંઠ, ગાંઠ->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

શૂન્ય ggml_set_આગત(સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_આગત;
}

શૂન્ય ggml_set_output(સંરચના ggml_tensor * tensor) {
    માટે (સંરચના ggml_tensor * cur = tensor; cur != ખાલી; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

શૂન્ય ggml_set_param(સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

શૂન્ય ggml_set_loss(સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

શૂન્ય ggml_quantize_init(ગણના ggml_type type) {
    ggml_critical_section_start();

    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); તોડો;
        સ્થિતિ GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); તોડો;
        સ્થિતિ GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); તોડો;
        મૂળભૂત: // nothing
            તોડો;
    }

    ggml_critical_section_end();
}

શૂન્ય ggml_quantize_free(શૂન્ય) {
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

bool ggml_quantize_requires_imatrix(ગણના ggml_type type) {
    પાછા
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        ગણના ggml_type   type,
           અચલ દશાંશ * src,
                  શૂન્ય * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           અચલ દશાંશ * imatrix) {
    અચલ int64_t n = nrows * n_per_row;

    જો (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ખાલી);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    અચલ size_t start_row = start / n_per_row;
    અચલ size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (અક્ષર *) dst + start_row * row_size, nrows, n_per_row, imatrix); તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                size_t elemsize = કદ(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                size_t elemsize = કદ(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                size_t elemsize = કદ(દશાંશ);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } તોડો;
        મૂળભૂત:
            પ્રતિજ્ઞા(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    પાછા result;
}

////////////////////////////////////////////////////////////////////////////////

શૂન્ય ggml_log_get(ggml_log_callback * log_callback, શૂન્ય ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

શૂન્ય ggml_log_set(ggml_log_callback log_callback, શૂન્ય * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

શૂન્ય ggml_threadpool_params_init(સંરચના ggml_threadpool_params * p, પૂર્ણાંક n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

સંરચના ggml_threadpool_params ggml_threadpool_params_default(પૂર્ણાંક n_threads) {
    સંરચના ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    પાછા p;
}

bool ggml_threadpool_params_match(અચલ સંરચના ggml_threadpool_params * p0, અચલ સંરચના ggml_threadpool_params * p1) {
    જો (p0->n_threads  != p1->n_threads  ) પાછા false;
    જો (p0->prio       != p1->prio       ) પાછા false;
    જો (p0->poll       != p1->poll       ) પાછા false;
    જો (p0->strict_cpu != p1->strict_cpu ) પાછા false;
    પાછા memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
