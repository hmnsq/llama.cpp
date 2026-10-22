#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#அந்தர்பாவ "ggml-version.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-threading.h"
#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "ggml.h"

// FIXME: required here for quantization functions
#அந்தர்பாவ "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#அந்தர்பாவ <hbwmalloc.h>
#endif

#எனில் defined(_MSC_VER) || defined(__MINGW32__)
#அந்தர்பாவ <malloc.h> // using malloc.h with MSC/MINGW
#அல்லதெனில் !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#அந்தர்பாவ <alloca.h>
#endif

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <errno.h>
#அந்தர்பாவ <time.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <inttypes.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <float.h>
#அந்தர்பாவ <limits.h>
#அந்தர்பாவ <stdarg.h>
#அந்தர்பாவ <signal.h>
#எனில் defined(__gnu_linux__)
#அந்தர்பாவ <syscall.h>
#endif

#எனில் defined(__APPLE__)
#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ <mach/mach.h>
#அந்தர்பாவ <TargetConditionals.h>
#endif

#எனில் defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#அந்தர்பாவ <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(வெறுமை) {
#ifdef _MSC_VER
#எனில் defined(_WIN32)
    நிலையான நிலையற்ற LONG counter = 1;
    திரும்பு (uint64_t) InterlockedIncrement(&counter) - 1;
#இல்லையெனில்
    நிலையான நிலையற்ற long long counter = 1;
    திரும்பு (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#இல்லையெனில்
    நிலையான uint64_t counter = 1;
    திரும்பு __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#எனில் defined(__AVX512BF16__)
#எனில் defined(_MSC_VER)
#define m512i(p) p
#இல்லையெனில்
#அந்தர்பாவ <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#எனில் defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <sys/stat.h>
#அந்தர்பாவ <sys/wait.h>
#எனில் defined(__linux__)
#அந்தர்பாவ <sys/prctl.h>
#endif

#எனில் defined(__ANDROID__)
#அந்தர்பாவ <unwind.h>
#அந்தர்பாவ <dlfcn.h>
#அந்தர்பாவ <stdio.h>

கட்டமைப்பு backtrace_state {
    வெறுமை ** current;
    வெறுமை ** end;
};

நிலையான _Unwind_Reason_Code unwind_callback(கட்டமைப்பு _Unwind_Context* context, வெறுமை* arg) {
    கட்டமைப்பு backtrace_state * state = (கட்டமைப்பு backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    எனில் (pc) {
        எனில் (state->current == state->end) {
            திரும்பு URC_END_OF_அடுக்கு;
        } இல்லையெனில் {
            *state->current++ = (வெறுமை*)pc;
        }
    }
    திரும்பு _URC_NO_REASON;
}

நிலையான வெறுமை ggml_print_backtrace_symbols(வெறுமை) {
    மாறிலி முழு max = 100;
    வெறுமை* buffer[max];

    கட்டமைப்பு backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    முழு மொத்தம் = state.current - buffer;

    க்கு (முழு idx = 0; idx < மொத்தம்; ++idx) {
        மாறிலி வெறுமை * addr = buffer[idx];
        மாறிலி எழுத்து * குறியீடு = "";

        Dl_info info;
        எனில் (dladdr(addr, &info) && info.dli_sname) {
            குறியீடு = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, குறியீடு);
    }
}
#அல்லதெனில் defined(__linux__) && defined(__GLIBC__)
#அந்தர்பாவ <execinfo.h>
நிலையான வெறுமை ggml_print_backtrace_symbols(வெறுமை) {
    வெறுமை * trace[100];
    முழு nptrs = backtrace(trace, அளவு(trace)/அளவு(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#அல்லதெனில் defined(__APPLE__)
#அந்தர்பாவ <execinfo.h>
நிலையான வெறுமை ggml_print_backtrace_symbols(வெறுமை) {
    வெறுமை * trace[100];
    முழு nptrs = backtrace(trace, அளவு(trace)/அளவு(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#இல்லையெனில்
நிலையான வெறுமை ggml_print_backtrace_symbols(வெறுமை) {
    // platform not supported
}
#endif

வெறுமை ggml_print_backtrace(வெறுமை) {
    மாறிலி எழுத்து * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    எனில் (GGML_NO_BACKTRACE) {
        திரும்பு;
    }
#எனில் defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    மாறிலி எழுத்து * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    எனில் (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        திரும்பு;
    }
#endif
#எனில் defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    எழுத்து * வரி = வெற்று;
    ssize_t length = 0;
    போது ((length = getline(&வரி, &size, f)) > 0) {
        எனில் (!strncmp(வரி, "TracerPid:", அளவு("TracerPid:") - 1) &&
            (length != அளவு("TracerPid:\t0\n") - 1 || வரி[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            விடுவி(வரி);
            fclose(f);
            திரும்பு;
        }
    }
    விடுவி(வரி);
    fclose(f);
    முழு lock[2] = { -1, -1 };
    (வெறுமை) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    மாறிலி முழு parent_pid = getpid();
    மாறிலி முழு child_pid = fork();
    எனில் (child_pid < 0) { // error
#எனில் defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        திரும்பு;
    } இல்லையெனில் எனில் (child_pid == 0) { // child
        எழுத்து attach[32];
        snprintf(attach, அளவு(attach), "attach %d", parent_pid);
#எனில் defined(__linux__)
        close(lock[1]);
        (வெறுமை) !பட(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (எழுத்து *) வெற்று);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[அளவு("attach ") - 1],
            (எழுத்து *) வெற்று);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } இல்லையெனில் { // parent
#எனில் defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, வெற்று, 0);
    }
}
#இல்லையெனில்
வெறுமை ggml_print_backtrace(வெறுமை) {
    // platform not supported
}
#endif

நிலையான ggml_abort_callback_t g_abort_callback = வெற்று;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    திரும்பு ret_val;
}

வெறுமை ggml_abort(மாறிலி எழுத்து * file, முழு வரி, மாறிலி எழுத்து * fmt, ...) {
    fflush(stdout);

    எழுத்து message[2048];
    முழு offset = snprintf(message, அளவு(message), "%s:%d: ", file, வரி);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, அளவு(message) - offset, fmt, args);
    va_end(args);

    எனில் (g_abort_callback) {
        g_abort_callback(message);
    } இல்லையெனில் {
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

கட்டமைப்பு ggml_logger_state {
    ggml_log_callback log_callback;
    வெறுமை * log_callback_user_data;
};
நிலையான கட்டமைப்பு ggml_logger_state g_logger_state = {ggml_log_callback_default, வெற்று};

நிலையான வெறுமை ggml_log_internal_v(எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * format, va_list args) {
    எனில் (format == வெற்று) {
        திரும்பு;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    எழுத்து buffer[128];
    முழு len = vsnprintf(buffer, 128, format, args);
    எனில் (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } இல்லையெனில் {
        எழுத்து * buffer2 = (எழுத்து *) calloc(len + 1, அளவு(எழுத்து));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        விடுவி(buffer2);
    }
    va_end(args_copy);
}

வெறுமை ggml_log_internal(எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

வெறுமை ggml_log_callback_default(எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * text, வெறுமை * user_data) {
    (வெறுமை) level;
    (வெறுமை) user_data;
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
#எனில் defined(__has_feature)
#எனில் __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#இல்லையெனில்
#எனில் defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

வெறுமை * ggml_aligned_malloc(size_t size) {
#எனில் defined(__s390x__)
    மாறிலி முழு alignment = 256;
#இல்லையெனில்
    மாறிலி முழு alignment = 64;
#endif

#எனில் defined(_MSC_VER) || defined(__MINGW32__)
    திரும்பு _aligned_malloc(size, alignment);
#இல்லையெனில்
    எனில் (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        திரும்பு வெற்று;
    }
    வெறுமை * aligned_memory = வெற்று;
  #ifdef GGML_USE_CPU_HBM
    முழு result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #அல்லதெனில் TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    முழு result = EFAULT;
    தேர்வு (alloc_status) {
        நிலை KERN_SUCCESS:
            result = 0;
            நிறுத்து;
        நிலை KERN_INVALID_ADDRESS:
            result = EINVAL;
            நிறுத்து;
        நிலை KERN_NO_SPACE:
            result = ENOMEM;
            நிறுத்து;
        இயல்பு:
            result = EFAULT;
            நிறுத்து;
    }
  #இல்லையெனில்
    முழு result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    எனில் (result != 0) {
        // Handle allocation failure
        மாறிலி எழுத்து *error_desc = "unknown allocation error";
        தேர்வு (result) {
            நிலை EINVAL:
                error_desc = "invalid alignment value";
                நிறுத்து;
            நிலை ENOMEM:
                error_desc = "insufficient memory";
                நிறுத்து;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        திரும்பு வெற்று;
    }
    திரும்பு aligned_memory;
#endif
}

வெறுமை ggml_aligned_free(வெறுமை * ptr, size_t size) {
    GGML_UNUSED(size);
#எனில் defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#அல்லதெனில் GGML_USE_CPU_HBM
    எனில் (ptr != வெற்று) {
        hbw_free(ptr);
    }
#அல்லதெனில் TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    எனில் (ptr != வெற்று) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#இல்லையெனில்
    விடுவி(ptr);
#endif
}


உள்வரி நிலையான வெறுமை * ggml_malloc(size_t size) {
    எனில் (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        திரும்பு வெற்று;
    }
    வெறுமை * result = malloc(size);
    எனில் (result == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    திரும்பு result;
}

// calloc
உள்வரி நிலையான வெறுமை * ggml_calloc(size_t இலக்கம், size_t size) {
    எனில் (இலக்கம் == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        திரும்பு வெற்று;
    }
    வெறுமை * result = calloc(இலக்கம், size);
    எனில் (result == வெற்று) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    திரும்பு result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(இலக்கம், size) ggml_calloc(இலக்கம், size)

#define GGML_FREE(ptr) விடுவி(ptr)

மாறிலி எழுத்து * ggml_status_to_string(எண்ணிக்கை ggml_status status) {
    தேர்வு (status) {
        நிலை GGML_STATUS_ALLOC_FAILED: திரும்பு "GGML status: error (failed to allocate memory)";
        நிலை GGML_STATUS_FAILED:       திரும்பு "GGML status: error (operation failed)";
        நிலை GGML_STATUS_SUCCESS:      திரும்பு "GGML status: success";
        நிலை GGML_STATUS_ABORTED:      திரும்பு "GGML status: warning (operation aborted)";
    }

    திரும்பு "GGML status: unknown";
}

தசம ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    திரும்பு GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(தசம x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    திரும்பு GGML_FP32_TO_FP16(x);
}

தசம ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    திரும்பு GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(தசம x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    திரும்பு GGML_FP32_TO_BF16(x);
}

வெறுமை ggml_fp16_to_fp32_row(மாறிலி ggml_fp16_t * x, தசம * y, int64_t n) {
    க்கு (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

வெறுமை ggml_fp32_to_fp16_row(மாறிலி தசம * x, ggml_fp16_t * y, int64_t n) {
    முழு i = 0;
    க்கு (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

வெறுமை ggml_bf16_to_fp32_row(மாறிலி ggml_bf16_t * x, தசம * y, int64_t n) {
    முழு i = 0;
    க்கு (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

வெறுமை ggml_fp32_to_bf16_row_ref(மாறிலி தசம * x, ggml_bf16_t * y, int64_t n) {
    க்கு (முழு i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

வெறுமை ggml_fp32_to_bf16_row(மாறிலி தசம * x, ggml_bf16_t * y, int64_t n) {
  முழு i = 0;
#எனில் defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  க்கு (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    க்கு (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    திரும்பு memcmp(guid_a, guid_b, அளவு(ggml_guid)) == 0;
}

மாறிலி எழுத்து * ggml_version(வெறுமை) {
    திரும்பு GGML_VERSION;
}

மாறிலி எழுத்து * ggml_commit(வெறுமை) {
    திரும்பு GGML_COMMIT;
}

//
// timing
//

#எனில் defined(_MSC_VER) || defined(__MINGW32__)
நிலையான int64_t timer_freq, timer_start;
நிலையான BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    திரும்பு TRUE;
}
வெறுமை ggml_time_init(வெறுமை) {
    நிலையான INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, வெற்று, வெற்று);
}
int64_t ggml_time_ms(வெறுமை) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    திரும்பு ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(வெறுமை) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    திரும்பு ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#இல்லையெனில்
வெறுமை ggml_time_init(வெறுமை) {}
int64_t ggml_time_ms(வெறுமை) {
    கட்டமைப்பு timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    திரும்பு (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(வெறுமை) {
    கட்டமைப்பு timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    திரும்பு (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(வெறுமை) {
    திரும்பு clock();
}

int64_t ggml_cycles_per_ms(வெறுமை) {
    திரும்பு CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
நிலையான wchar_t * ggml_mbstowcs(மாறிலி எழுத்து * mbs) {
    முழு wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, வெற்று, 0);
    எனில் (!wlen) {
        errno = EINVAL;
        திரும்பு வெற்று;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * அளவு(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    எனில் (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        திரும்பு வெற்று;
    }

    திரும்பு wbuf;
}
#endif

FILE * ggml_fopen(மாறிலி எழுத்து * fname, மாறிலி எழுத்து * mode) {
#ifdef _WIN32
    FILE * file = வெற்று;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    எனில் (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        எனில் (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    திரும்பு file;
#இல்லையெனில்
    திரும்பு fopen(fname, mode);
#endif

}

நிலையான மாறிலி கட்டமைப்பு ggml_type_traits type_traits[GGML_TYPE_மொத்தம்] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = அளவு(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = அளவு(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = அளவு(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = அளவு(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = அளவு(இரட்டை),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = அளவு(தசம),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = அளவு(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = அளவு(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = அளவு(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = அளவு(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = அளவு(block_q4_1),
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
        .type_size                = அளவு(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = அளவு(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = அளவு(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = அளவு(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = அளவு(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = அளவு(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = வெற்று,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = வெற்று,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = வெற்று,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = வெற்று,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = அளவு(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = அளவு(ggml_bf16_t),
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
        .type_size                = அளவு(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = அளவு(block_tq2_0),
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

மாறிலி கட்டமைப்பு ggml_type_traits * ggml_get_type_traits(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு &type_traits[type];
}

//
// ggml object
//

கட்டமைப்பு ggml_object {
    size_t offs;
    size_t size;

    கட்டமைப்பு ggml_object * next;

    எண்ணிக்கை ggml_object_type type;

    எழுத்து padding[4];
};

நிலையான மாறிலி size_t GGML_OBJECT_SIZE = அளவு(கட்டமைப்பு ggml_object);

//
// ggml context
//

கட்டமைப்பு ggml_context {
    size_t mem_size;
    வெறுமை * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    முழு    n_objects;

    கட்டமைப்பு ggml_object * objects_begin;
    கட்டமைப்பு ggml_object * objects_end;
};

//
// data types
//

நிலையான மாறிலி எழுத்து * GGML_OP_NAME[GGML_OP_மொத்தம்] = {
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

static_assert(GGML_OP_மொத்தம் == 102, "GGML_OP_COUNT != 102");

நிலையான மாறிலி எழுத்து * GGML_OP_குறியீடு[GGML_OP_மொத்தம்] = {
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

static_assert(GGML_OP_மொத்தம் == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_மொத்தம் == 2, "GGML_OP_POOL_COUNT != 2");

நிலையான மாறிலி எழுத்து * GGML_UNARY_OP_NAME[GGML_UNARY_OP_மொத்தம்] = {
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

static_assert(GGML_UNARY_OP_மொத்தம் == 22, "GGML_UNARY_OP_COUNT != 22");

நிலையான மாறிலி எழுத்து * GGML_GLU_OP_NAME[GGML_GLU_OP_மொத்தம்] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_மொத்தம் == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(அளவு(கட்டமைப்பு ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(அளவு(கட்டமைப்பு ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

வெறுமை ggml_print_object(மாறிலி கட்டமைப்பு ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (மாறிலி வெறுமை *) obj->next);
}

வெறுமை ggml_print_objects(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    கட்டமைப்பு ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (மாறிலி வெறுமை *) ctx);

    போது (obj != வெற்று) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; ++i) {
        எனில் (tensor->ne[i] <= 0) {
            திரும்பு 0;
        }
    }

    size_t nbytes;
    மாறிலி size_t blck_size = ggml_blck_size(tensor->type);
    எனில் (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        க்கு (முழு i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    இல்லையெனில் {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        க்கு (முழு i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    திரும்பு nbytes;
}

size_t ggml_nbytes_pad(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு type_traits[type].blck_size;
}

size_t ggml_type_size(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு type_traits[type].type_size;
}

size_t ggml_row_size(எண்ணிக்கை ggml_type type, int64_t ne) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    ப்ரதிஜ்ஞா(ne % ggml_blck_size(type) == 0);
    திரும்பு ggml_type_size(type)*ne/ggml_blck_size(type);
}

இரட்டை ggml_type_sizef(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு ((இரட்டை)(type_traits[type].type_size))/type_traits[type].blck_size;
}

மாறிலி எழுத்து * ggml_type_name(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு type_traits[type].type_name;
}

bool ggml_is_quantized(எண்ணிக்கை ggml_type type) {
    ப்ரதிஜ்ஞா(type >= 0);
    ப்ரதிஜ்ஞா(type < GGML_TYPE_மொத்தம்);
    திரும்பு type_traits[type].is_quantized;
}

மாறிலி எழுத்து * ggml_op_name(எண்ணிக்கை ggml_op op) {
    திரும்பு GGML_OP_NAME[op];
}

மாறிலி எழுத்து * ggml_op_குறியீடு(எண்ணிக்கை ggml_op op) {
    திரும்பு GGML_OP_குறியீடு[op];
}

மாறிலி எழுத்து * ggml_unary_op_name(எண்ணிக்கை ggml_unary_op op) {
    திரும்பு GGML_UNARY_OP_NAME[op];
}

மாறிலி எழுத்து * ggml_glu_op_name(எண்ணிக்கை ggml_glu_op op) {
    திரும்பு GGML_GLU_OP_NAME[op];
}

மாறிலி எழுத்து * ggml_op_desc(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (t->op == GGML_OP_UNARY) {
        எண்ணிக்கை ggml_unary_op uop = ggml_get_unary_op(t);
        திரும்பு ggml_unary_op_name(uop);
    }
    எனில் (t->op == GGML_OP_GLU) {
        எண்ணிக்கை ggml_glu_op gop = ggml_get_glu_op(t);
        திரும்பு ggml_glu_op_name(gop);
    }
    திரும்பு ggml_op_name(t->op);
}

size_t ggml_element_size(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_type_size(tensor->type);
}

bool ggml_is_scalar(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு tensor->ne[3] == 1;
}

முழு ggml_n_dims(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    க்கு (முழு i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        எனில் (tensor->ne[i] > 1) {
            திரும்பு i + 1;
        }
    }
    திரும்பு 1;
}

எண்ணிக்கை ggml_type ggml_ftype_to_ggml_type(எண்ணிக்கை ggml_ftype ftype) {
    எண்ணிக்கை ggml_type wtype = GGML_TYPE_மொத்தம்;

    தேர்வு (ftype) {
        நிலை GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    நிறுத்து;
        நிலை GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_மொத்தம்; நிறுத்து;
        நிலை GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_மொத்தம்; நிறுத்து;
    }

    GGML_ASSERT(wtype != GGML_TYPE_மொத்தம்);

    திரும்பு wtype;
}

size_t ggml_tensor_overhead(வெறுமை) {
    திரும்பு GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு tensor->nb[0] > tensor->nb[1];
}

நிலையான bool ggml_is_contiguous_m_n(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு m, முழு n) {
    size_t next_nb = ggml_type_size(tensor->type);
    எனில் (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        திரும்பு false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    க்கு (முழு i = 1; i < n; i++) {
        எனில் (i > m) {
            எனில் (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                திரும்பு false;
            }
            next_nb *= tensor->ne[i];
        } இல்லையெனில் {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    திரும்பு true;
}

bool ggml_is_contiguous(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

நிலையான உள்வரி bool ggml_is_padded_1d(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; ++i) {
        எனில் (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            திரும்பு true;
        }
    }
    திரும்பு false;
}

bool ggml_are_same_shape(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    திரும்பு ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

நிலையான உள்வரி bool ggml_can_repeat_rows(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

கட்டமைப்பு ggml_context * ggml_init(கட்டமைப்பு ggml_init_params params) {
    நிலையான bool is_first_call = true;

    ggml_critical_section_start();

    எனில் (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    கட்டமைப்பு ggml_context * ctx = GGML_MALLOC(அளவு(கட்டமைப்பு ggml_context));

    // allow to call ggml_init with 0 size
    எனில் (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    மாறிலி size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (கட்டமைப்பு ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ வெற்று,
        /*.objects_end        =*/ வெற்று,
    };

    GGML_ASSERT(ctx->mem_buffer != வெற்று);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    திரும்பு ctx;
}

வெறுமை ggml_reset(கட்டமைப்பு ggml_context * ctx) {
    எனில் (ctx == வெற்று) {
        திரும்பு;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = வெற்று;
    ctx->objects_end   = வெற்று;
}

வெறுமை ggml_free(கட்டமைப்பு ggml_context * ctx) {
    எனில் (ctx == வெற்று) {
        திரும்பு;
    }

    எனில் (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    திரும்பு ctx->objects_end == வெற்று ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(கட்டமைப்பு ggml_context * ctx) {
    திரும்பு ctx->no_alloc;
}

வெறுமை ggml_set_no_alloc(கட்டமைப்பு ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

வெறுமை * ggml_get_mem_buffer(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    திரும்பு ctx->mem_buffer;
}

size_t ggml_get_mem_size(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    திரும்பு ctx->mem_size;
}

size_t ggml_get_max_tensor_size(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    size_t max_size = 0;

    க்கு (கட்டமைப்பு ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != வெற்று; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    திரும்பு max_size;
}

////////////////////////////////////////////////////////////////////////////////

நிலையான கட்டமைப்பு ggml_object * ggml_new_object(கட்டமைப்பு ggml_context * ctx, எண்ணிக்கை ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    கட்டமைப்பு ggml_object * obj_cur = ctx->objects_end;

    மாறிலி size_t cur_offs = obj_cur == வெற்று ? 0 : obj_cur->offs;
    மாறிலி size_t cur_size = obj_cur == வெற்று ? 0 : obj_cur->size;
    மாறிலி size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    எழுத்து * மாறிலி mem_buffer = ctx->mem_buffer;
    கட்டமைப்பு ggml_object * மாறிலி obj_new = (கட்டமைப்பு ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    எனில் (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        திரும்பு வெற்று;
    }
    எனில் (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        திரும்பு வெற்று;
    }

    எனில் (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        திரும்பு வெற்று;
    }

    *obj_new = (கட்டமைப்பு ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = வெற்று,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    எனில் (obj_cur != வெற்று) {
        obj_cur->next = obj_new;
    } இல்லையெனில் {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    திரும்பு obj_new;
}

நிலையான கட்டமைப்பு ggml_tensor * ggml_new_tensor_impl(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type      type,
        முழு                   n_dims,
        மாறிலி int64_t       * ne,
        கட்டமைப்பு ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_மொத்தம்);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    எனில் (view_src != வெற்று && view_src->view_src != வெற்று) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    க்கு (முழு i = 1; i < n_dims; i++) {
        எனில் (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    க்கு (முழு i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == வெற்று || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    வெறுமை * data = view_src != வெற்று ? view_src->data : வெற்று;
    எனில் (data != வெற்று) {
        data = (எழுத்து *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    எனில் (view_src == வெற்று && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    கட்டமைப்பு ggml_object * மாறிலி obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    கட்டமைப்பு ggml_tensor * மாறிலி result = (கட்டமைப்பு ggml_tensor *)((எழுத்து *)ctx->mem_buffer + obj_new->offs);

    *result = (கட்டமைப்பு ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ வெற்று,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { வெற்று },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (வெறுமை *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ வெற்று,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    க்கு (முழு i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    க்கு (முழு i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_new_tensor(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type      type,
        முழு                   n_dims,
        மாறிலி int64_t       * ne) {
    திரும்பு ggml_new_tensor_impl(ctx, type, n_dims, ne, வெற்று, 0);
}

கட்டமைப்பு ggml_tensor * ggml_new_tensor_1d(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type      type,
        int64_t ne0) {
    திரும்பு ggml_new_tensor(ctx, type, 1, &ne0);
}

கட்டமைப்பு ggml_tensor * ggml_new_tensor_2d(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    மாறிலி int64_t ne[2] = { ne0, ne1 };
    திரும்பு ggml_new_tensor(ctx, type, 2, ne);
}

கட்டமைப்பு ggml_tensor * ggml_new_tensor_3d(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    மாறிலி int64_t ne[3] = { ne0, ne1, ne2 };
    திரும்பு ggml_new_tensor(ctx, type, 3, ne);
}

கட்டமைப்பு ggml_tensor * ggml_new_tensor_4d(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    மாறிலி int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    திரும்பு ggml_new_tensor(ctx, type, 4, ne);
}

வெறுமை * ggml_new_buffer(கட்டமைப்பு ggml_context * ctx, size_t nbytes) {
    கட்டமைப்பு ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    திரும்பு (uint8_t *)ctx->mem_buffer + obj->offs;
}

கட்டமைப்பு ggml_tensor * ggml_dup_tensor(கட்டமைப்பு ggml_context * ctx, மாறிலி கட்டமைப்பு ggml_tensor * src) {
    திரும்பு ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

வெறுமை ggml_unravel_index(மாறிலி கட்டமைப்பு ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    மாறிலி int64_t ne2 = tensor->ne[2];
    மாறிலி int64_t ne1 = tensor->ne[1];
    மாறிலி int64_t ne0 = tensor->ne[0];

    மாறிலி int64_t i3_ = (i/(ne2*ne1*ne0));
    மாறிலி int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    மாறிலி int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    மாறிலி int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    எனில் (i0) {
        * i0 = i0_;
    }
    எனில் (i1) {
        * i1 = i1_;
    }
    எனில் (i2) {
        * i2 = i2_;
    }
    எனில் (i3) {
        * i3 = i3_;
    }
}

வெறுமை * ggml_get_data(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு tensor->data;
}

தசம * ggml_get_data_f32(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    ப்ரதிஜ்ஞா(tensor->type == GGML_TYPE_F32);
    திரும்பு (தசம *)(tensor->data);
}

எண்ணிக்கை ggml_unary_op ggml_get_unary_op(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    திரும்பு (எண்ணிக்கை ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

எண்ணிக்கை ggml_glu_op ggml_get_glu_op(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    திரும்பு (எண்ணிக்கை ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

மாறிலி எழுத்து * ggml_get_name(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு tensor->name;
}

கட்டமைப்பு ggml_tensor * ggml_set_name(கட்டமைப்பு ggml_tensor * tensor, மாறிலி எழுத்து * name) {
    size_t i;
    க்கு (i = 0; i < அளவு(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    திரும்பு tensor;
}

கட்டமைப்பு ggml_tensor * ggml_format_name(கட்டமைப்பு ggml_tensor * tensor, மாறிலி எழுத்து * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, அளவு(tensor->name), fmt, args);
    va_end(args);
    திரும்பு tensor;
}

கட்டமைப்பு ggml_tensor * ggml_view_tensor(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * src) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_get_first_tensor(மாறிலி கட்டமைப்பு ggml_context * ctx) {
    கட்டமைப்பு ggml_object * obj = ctx->objects_begin;

    எழுத்து * மாறிலி mem_buffer = ctx->mem_buffer;

    போது (obj != வெற்று) {
        எனில் (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            திரும்பு (கட்டமைப்பு ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    திரும்பு வெற்று;
}

கட்டமைப்பு ggml_tensor * ggml_get_next_tensor(மாறிலி கட்டமைப்பு ggml_context * ctx, கட்டமைப்பு ggml_tensor * tensor) {
    கட்டமைப்பு ggml_object * obj = (கட்டமைப்பு ggml_object *) ((எழுத்து *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    எழுத்து * மாறிலி mem_buffer = ctx->mem_buffer;

    போது (obj != வெற்று) {
        எனில் (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            திரும்பு (கட்டமைப்பு ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    திரும்பு வெற்று;
}

கட்டமைப்பு ggml_tensor * ggml_get_tensor(கட்டமைப்பு ggml_context * ctx, மாறிலி எழுத்து * name) {
    கட்டமைப்பு ggml_object * obj = ctx->objects_begin;

    எழுத்து * மாறிலி mem_buffer = ctx->mem_buffer;

    போது (obj != வெற்று) {
        எனில் (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            கட்டமைப்பு ggml_tensor * cur = (கட்டமைப்பு ggml_tensor *)(mem_buffer + obj->offs);
            எனில் (strcmp(cur->name, name) == 0) {
                திரும்பு cur;
            }
        }

        obj = obj->next;
    }

    திரும்பு வெற்று;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

நிலையான கட்டமைப்பு ggml_tensor * ggml_dup_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_dup(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_dup_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_dup_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_dup_impl(ctx, a, true);
}

// ggml_add

நிலையான கட்டமைப்பு ggml_tensor * ggml_add_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_add(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_add_impl(ctx, a, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_add_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

நிலையான கட்டமைப்பு ggml_tensor * ggml_add_cast_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        எண்ணிக்கை   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_add_cast(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        எண்ணிக்கை   ggml_type      type) {
    திரும்பு ggml_add_cast_impl(ctx, a, b, type);
}

கட்டமைப்பு ggml_tensor * ggml_add_id(
            கட்டமைப்பு ggml_context * ctx,
            கட்டமைப்பு ggml_tensor  * a,
            கட்டமைப்பு ggml_tensor  * b,
            கட்டமைப்பு ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    திரும்பு result;
}

// ggml_add1

நிலையான கட்டமைப்பு ggml_tensor * ggml_add1_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_add1(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_add1_impl(ctx, a, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_add1_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

நிலையான கட்டமைப்பு ggml_tensor * ggml_acc_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_acc(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    திரும்பு ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

கட்டமைப்பு ggml_tensor * ggml_acc_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    திரும்பு ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

நிலையான கட்டமைப்பு ggml_tensor * ggml_sub_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_sub(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_sub_impl(ctx, a, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_sub_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

நிலையான கட்டமைப்பு ggml_tensor * ggml_mul_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_mul(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_mul_impl(ctx, a, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_mul_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

நிலையான கட்டமைப்பு ggml_tensor * ggml_div_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_div(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_div_impl(ctx, a, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_div_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

நிலையான கட்டமைப்பு ggml_tensor * ggml_sqr_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_sqr(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sqr_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_sqr_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

நிலையான கட்டமைப்பு ggml_tensor * ggml_sqrt_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_sqrt(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sqrt_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_sqrt_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

நிலையான கட்டமைப்பு ggml_tensor * ggml_log_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_log(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_log_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_log_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_log_impl(ctx, a, true);
}

கட்டமைப்பு ggml_tensor * ggml_expm1(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

கட்டமைப்பு ggml_tensor * ggml_expm1_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

கட்டமைப்பு ggml_tensor * ggml_softplus(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

கட்டமைப்பு ggml_tensor * ggml_softplus_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

நிலையான கட்டமைப்பு ggml_tensor * ggml_sin_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_sin(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sin_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_sin_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_sin_impl(ctx, a, true);
}

// ggml_cos

நிலையான கட்டமைப்பு ggml_tensor * ggml_cos_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_cos(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_cos_impl(ctx, a, false);
}

கட்டமைப்பு ggml_tensor * ggml_cos_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_cos_impl(ctx, a, true);
}

// ggml_sum

கட்டமைப்பு ggml_tensor * ggml_sum(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_sum_rows

கட்டமைப்பு ggml_tensor * ggml_sum_rows(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    க்கு (முழு i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_cumsum

கட்டமைப்பு ggml_tensor * ggml_cumsum(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_mean

கட்டமைப்பு ggml_tensor * ggml_mean(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_argmax

கட்டமைப்பு ggml_tensor * ggml_argmax(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_count_equal

கட்டமைப்பு ggml_tensor * ggml_மொத்தம்_equal(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_மொத்தம்_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_repeat

கட்டமைப்பு ggml_tensor * ggml_repeat(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_repeat_4d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    மாறிலி bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_repeat_back

கட்டமைப்பு ggml_tensor * ggml_repeat_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_concat

கட்டமைப்பு ggml_tensor * ggml_concat(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    கட்டமைப்பு ggml_tensor  * b,
    முழு                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    க்கு (முழு d = 0; d < GGML_MAX_DIMS; ++d) {
        எனில் (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            தொடர்;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_abs

கட்டமைப்பு ggml_tensor * ggml_abs(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

கட்டமைப்பு ggml_tensor * ggml_abs_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

கட்டமைப்பு ggml_tensor * ggml_sgn(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

கட்டமைப்பு ggml_tensor * ggml_sgn_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

கட்டமைப்பு ggml_tensor * ggml_neg(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

கட்டமைப்பு ggml_tensor * ggml_neg_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

கட்டமைப்பு ggml_tensor * ggml_step(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

கட்டமைப்பு ggml_tensor * ggml_step_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

கட்டமைப்பு ggml_tensor * ggml_tanh(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

கட்டமைப்பு ggml_tensor * ggml_tanh_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

கட்டமைப்பு ggml_tensor * ggml_elu(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

கட்டமைப்பு ggml_tensor * ggml_elu_inplace(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

கட்டமைப்பு ggml_tensor * ggml_relu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

கட்டமைப்பு ggml_tensor * ggml_relu_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

கட்டமைப்பு ggml_tensor * ggml_leaky_relu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 negative_slope,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, அளவு(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_sigmoid

கட்டமைப்பு ggml_tensor * ggml_sigmoid(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

கட்டமைப்பு ggml_tensor * ggml_sigmoid_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

கட்டமைப்பு ggml_tensor * ggml_gelu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

கட்டமைப்பு ggml_tensor * ggml_gelu_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

கட்டமைப்பு ggml_tensor * ggml_gelu_erf(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

கட்டமைப்பு ggml_tensor * ggml_gelu_erf_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

கட்டமைப்பு ggml_tensor * ggml_gelu_quick(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

கட்டமைப்பு ggml_tensor * ggml_gelu_quick_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

கட்டமைப்பு ggml_tensor * ggml_silu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

கட்டமைப்பு ggml_tensor * ggml_silu_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

கட்டமைப்பு ggml_tensor * ggml_xielu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம alpha_n,
        தசம alpha_p,
        தசம beta,
        தசம eps) {
    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_silu_back

கட்டமைப்பு ggml_tensor * ggml_silu_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml hardswish

கட்டமைப்பு ggml_tensor * ggml_hardswish(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

கட்டமைப்பு ggml_tensor * ggml_hardsigmoid(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

கட்டமைப்பு ggml_tensor * ggml_exp(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

கட்டமைப்பு ggml_tensor * ggml_exp_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

நிலையான கட்டமைப்பு ggml_tensor * ggml_glu_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        எண்ணிக்கை ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    எனில் (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; க்கு (முழு i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, வெற்று, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_floor

கட்டமைப்பு ggml_tensor * ggml_floor(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

கட்டமைப்பு ggml_tensor * ggml_floor_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

கட்டமைப்பு ggml_tensor * ggml_ceil(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

கட்டமைப்பு ggml_tensor * ggml_ceil_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

கட்டமைப்பு ggml_tensor * ggml_round(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

கட்டமைப்பு ggml_tensor * ggml_round_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

கட்டமைப்பு ggml_tensor * ggml_trunc(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

கட்டமைப்பு ggml_tensor * ggml_trunc_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

கட்டமைப்பு ggml_tensor * ggml_glu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_glu_op      op,
        bool                  swapped) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, op, swapped);
}

கட்டமைப்பு ggml_tensor * ggml_glu_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        எண்ணிக்கை ggml_glu_op      op) {
    திரும்பு ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

கட்டமைப்பு ggml_tensor * ggml_reglu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_REGLU, false);
}

கட்டமைப்பு ggml_tensor * ggml_reglu_swapped(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_REGLU, true);
}

கட்டமைப்பு ggml_tensor * ggml_reglu_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

கட்டமைப்பு ggml_tensor * ggml_geglu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU, false);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_swapped(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU, true);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

கட்டமைப்பு ggml_tensor * ggml_swiglu(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_SWIGLU, false);
}

கட்டமைப்பு ggml_tensor * ggml_swiglu_swapped(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_SWIGLU, true);
}

கட்டமைப்பு ggml_tensor * ggml_swiglu_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

கட்டமைப்பு ggml_tensor * ggml_geglu_erf(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU_ERF, false);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_erf_swapped(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU_ERF, true);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_erf_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

கட்டமைப்பு ggml_tensor * ggml_geglu_quick(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU_QUICK, false);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_quick_swapped(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_glu_impl(ctx, a, வெற்று, GGML_GLU_OP_GEGLU_QUICK, true);
}

கட்டமைப்பு ggml_tensor * ggml_geglu_quick_split(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

கட்டமைப்பு ggml_tensor * ggml_swiglu_oai(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 alpha,
        தசம                 limit) {
    கட்டமைப்பு ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_swiglu_clamp(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 limit) {
    கட்டமைப்பு ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    திரும்பு result;
}

// ggml_norm

நிலையான கட்டமைப்பு ggml_tensor * ggml_norm_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, அளவு(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_norm(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_norm_impl(ctx, a, eps, false);
}

கட்டமைப்பு ggml_tensor * ggml_norm_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

நிலையான கட்டமைப்பு ggml_tensor * ggml_rms_norm_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, அளவு(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_rms_norm(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_rms_norm_impl(ctx, a, eps, false);
}

கட்டமைப்பு ggml_tensor * ggml_rms_norm_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

கட்டமைப்பு ggml_tensor * ggml_rms_norm_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 eps) {
    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, அளவு(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_group_norm

நிலையான கட்டமைப்பு ggml_tensor * ggml_group_norm_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_groups,
        தசம                 eps,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_group_norm(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_groups,
        தசம                 eps) {
    திரும்பு ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

கட்டமைப்பு ggml_tensor * ggml_group_norm_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_groups,
        தசம                 eps) {
    திரும்பு ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

நிலையான கட்டமைப்பு ggml_tensor * ggml_l2_norm_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_l2_norm(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_l2_norm_impl(ctx, a, eps, false);
}

கட்டமைப்பு ggml_tensor * ggml_l2_norm_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 eps) {
    திரும்பு ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        கட்டமைப்பு ggml_tensor * a,
        எண்ணிக்கை ggml_prec       prec) {
    தேர்வு (a->op) {
        நிலை GGML_OP_MUL_MAT:
        நிலை GGML_OP_MUL_MAT_ID:
            {
                மாறிலி int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            நிறுத்து;
        நிலை GGML_OP_FLASH_ATTN_EXT:
            {
                மாறிலி int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            நிறுத்து;
        இயல்பு:
            திரும்பு false;
    };

    திரும்பு true;
}

bool ggml_prec_set_src(
        கட்டமைப்பு ggml_tensor * a,
        எண்ணிக்கை ggml_prec       prec,
        முழு                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    தேர்வு (a->op) {
        நிலை GGML_OP_MUL_MAT:
        நிலை GGML_OP_MUL_MAT_ID:
            {
                எனில் (idx != 1) {
                    திரும்பு false;
                }

                மாறிலி int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            நிறுத்து;
        இயல்பு:
            திரும்பு false;
    };

    திரும்பு true;
}

// ggml_mul_mat

நிலையான உள்வரி bool ggml_can_mul_mat(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

கட்டமைப்பு ggml_tensor * ggml_mul_mat(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    மாறிலி int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

வெறுமை ggml_mul_mat_set_prec(
        கட்டமைப்பு ggml_tensor * a,
        எண்ணிக்கை ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    மாறிலி int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

வெறுமை ggml_mul_mat_set_hint(
        கட்டமைப்பு ggml_tensor * a,
        எண்ணிக்கை ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    மாறிலி int32_t hint_i32 = (int32_t) hint;

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
கட்டமைப்பு ggml_tensor * ggml_mul_mat_id(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * as,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    மாறிலி int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    திரும்பு result;
}

// ggml_out_prod

நிலையான உள்வரி bool ggml_can_out_prod(மாறிலி கட்டமைப்பு ggml_tensor * t0, மாறிலி கட்டமைப்பு ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    திரும்பு (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

கட்டமைப்பு ggml_tensor * ggml_out_prod(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    மாறிலி int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_scale

நிலையான கட்டமைப்பு ggml_tensor * ggml_scale_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 s,
        தசம                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    தசம params[2] = { s, b };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_scale(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 s) {
    திரும்பு ggml_scale_impl(ctx, a, s, 0.0, false);
}

கட்டமைப்பு ggml_tensor * ggml_scale_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 s) {
    திரும்பு ggml_scale_impl(ctx, a, s, 0.0, true);
}

கட்டமைப்பு ggml_tensor * ggml_scale_bias(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 s,
        தசம                 b) {
    திரும்பு ggml_scale_impl(ctx, a, s, b, false);
}

கட்டமைப்பு ggml_tensor * ggml_scale_bias_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 s,
        தசம                 b) {
    திரும்பு ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

நிலையான கட்டமைப்பு ggml_tensor * ggml_set_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_set(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

கட்டமைப்பு ggml_tensor * ggml_set_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

கட்டமைப்பு ggml_tensor * ggml_set_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

கட்டமைப்பு ggml_tensor * ggml_set_1d_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

கட்டமைப்பு ggml_tensor * ggml_set_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

கட்டமைப்பு ggml_tensor * ggml_set_2d_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    திரும்பு ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

நிலையான கட்டமைப்பு ggml_tensor * ggml_cpy_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, b);
    எனில் (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } இல்லையெனில் {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_cpy(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor * a,
        கட்டமைப்பு ggml_tensor * b) {
    திரும்பு ggml_cpy_impl(ctx, a, b);
}

கட்டமைப்பு ggml_tensor * ggml_cast(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை   ggml_type      type) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    திரும்பு result;
}

// ggml_cont

நிலையான கட்டமைப்பு ggml_tensor * ggml_cont_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_cont(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor * a) {
    திரும்பு ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API கட்டமைப்பு ggml_tensor * ggml_cont_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0) {
    திரும்பு ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API கட்டமைப்பு ggml_tensor * ggml_cont_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    திரும்பு ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API கட்டமைப்பு ggml_tensor * ggml_cont_3d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    திரும்பு ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

கட்டமைப்பு ggml_tensor * ggml_cont_4d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_reshape

கட்டமைப்பு ggml_tensor * ggml_reshape(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor * a,
        கட்டமைப்பு ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_reshape_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    மாறிலி int64_t ne[1] = { ne0 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_reshape_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    மாறிலி int64_t ne[2] = { ne0, ne1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_reshape_3d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    மாறிலி int64_t ne[3] = { ne0, ne1, ne2 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_reshape_4d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    மாறிலி int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    திரும்பு result;
}

நிலையான கட்டமைப்பு ggml_tensor * ggml_view_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_dims,
        மாறிலி int64_t       * ne,
        size_t                offset) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, அளவு(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_view_1d

கட்டமைப்பு ggml_tensor * ggml_view_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    கட்டமைப்பு ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    திரும்பு result;
}

// ggml_view_2d

கட்டமைப்பு ggml_tensor * ggml_view_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    மாறிலி int64_t ne[2] = { ne0, ne1 };

    கட்டமைப்பு ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    திரும்பு result;
}

// ggml_view_3d

கட்டமைப்பு ggml_tensor * ggml_view_3d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    மாறிலி int64_t ne[3] = { ne0, ne1, ne2 };

    கட்டமைப்பு ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    திரும்பு result;
}

// ggml_view_4d

கட்டமைப்பு ggml_tensor * ggml_view_4d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    மாறிலி int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    கட்டமைப்பு ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    திரும்பு result;
}

// ggml_permute

கட்டமைப்பு ggml_tensor * ggml_permute(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   axis0,
        முழு                   axis1,
        முழு                   axis2,
        முழு                   axis3) {
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

    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, அளவு(params));

    திரும்பு result;
}

// ggml_transpose

கட்டமைப்பு ggml_tensor * ggml_transpose(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_get_rows

கட்டமைப்பு ggml_tensor * ggml_get_rows(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    எண்ணிக்கை ggml_type type = GGML_TYPE_F32;
    எனில் (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_get_rows_back

கட்டமைப்பு ggml_tensor * ggml_get_rows_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_set_rows

கட்டமைப்பு ggml_tensor * ggml_set_rows(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c) {
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

    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    திரும்பு result;
}

// ggml_diag

கட்டமைப்பு ggml_tensor * ggml_diag(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    மாறிலி int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_diag_mask_inf

நிலையான கட்டமைப்பு ggml_tensor * ggml_diag_mask_inf_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_diag_mask_inf(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past) {
    திரும்பு ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

கட்டமைப்பு ggml_tensor * ggml_diag_mask_inf_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past) {
    திரும்பு ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

நிலையான கட்டமைப்பு ggml_tensor * ggml_diag_mask_zero_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_diag_mask_zero(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past) {
    திரும்பு ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

கட்டமைப்பு ggml_tensor * ggml_diag_mask_zero_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_past) {
    திரும்பு ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

நிலையான கட்டமைப்பு ggml_tensor * ggml_clamp_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        தசம                 min,
        தசம                 max,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    தசம params[] = { min, max };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_clamp(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    தசம                 min,
    தசம                 max) {
    திரும்பு ggml_clamp_impl(ctx, a, min, max, false);
}

கட்டமைப்பு ggml_tensor * ggml_clamp_inplace(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    தசம                 min,
    தசம                 max) {
    திரும்பு ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

நிலையான கட்டமைப்பு ggml_tensor * ggml_soft_max_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * mask,
        தசம                 scale,
        தசம                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    எனில் (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    எனில் (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    தசம params[] = { scale, max_bias };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_soft_max(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_soft_max_impl(ctx, a, வெற்று, 1.0f, 0.0f, false);
}

கட்டமைப்பு ggml_tensor * ggml_soft_max_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a) {
    திரும்பு ggml_soft_max_impl(ctx, a, வெற்று, 1.0f, 0.0f, true);
}

கட்டமைப்பு ggml_tensor * ggml_soft_max_ext(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * mask,
        தசம                 scale,
        தசம                 max_bias) {
    திரும்பு ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

கட்டமைப்பு ggml_tensor * ggml_soft_max_ext_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * mask,
        தசம                 scale,
        தசம                 max_bias) {
    திரும்பு ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

வெறுமை ggml_soft_max_add_sinks(
        கட்டமைப்பு ggml_tensor * a,
        கட்டமைப்பு ggml_tensor * sinks) {
    எனில் (!sinks) {
        a->src[2] = வெற்று;
        திரும்பு;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == வெற்று);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

நிலையான கட்டமைப்பு ggml_tensor * ggml_soft_max_ext_back_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 scale,
        தசம                 max_bias,
        bool                  inplace) {
    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((தசம *) result->op_params + 0, &scale,    அளவு(தசம));
    memcpy((தசம *) result->op_params + 1, &max_bias, அளவு(தசம));

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_soft_max_ext_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 scale,
        தசம                 max_bias) {
    திரும்பு ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

கட்டமைப்பு ggml_tensor * ggml_soft_max_ext_back_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        தசம                 scale,
        தசம                 max_bias) {
    திரும்பு ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

நிலையான கட்டமைப்பு ggml_tensor * ggml_rope_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   sections[GGML_MROPE_SECTIONS],
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    எனில் (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } இல்லையெனில் {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    எனில் (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    அளவு(தசம));
    memcpy(params +  6, &freq_scale,   அளவு(தசம));
    memcpy(params +  7, &ext_factor,   அளவு(தசம));
    memcpy(params +  8, &attn_factor,  அளவு(தசம));
    memcpy(params +  9, &beta_fast,    அளவு(தசம));
    memcpy(params + 10, &beta_slow,    அளவு(தசம));
    எனில் (mrope_used && sections) {
        memcpy(params + 11, sections,  அளவு(int32_t) * GGML_MROPE_SECTIONS);
    } இல்லையெனில் {
        memset(params + 11, 0,         அளவு(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_rope(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   n_dims,
        முழு                   mode) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, வெற்று, n_dims, வெற்று, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_multi(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   sections[GGML_MROPE_SECTIONS],
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_multi_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   sections[GGML_MROPE_SECTIONS],
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   n_dims,
        முழு                   mode) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, வெற்று, n_dims, வெற்று, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_ext(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, c, n_dims, வெற்று, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_ext_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, c, n_dims, வெற்று, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_custom(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   n_dims,
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, வெற்று, n_dims, வெற்று, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

கட்டமைப்பு ggml_tensor * ggml_rope_custom_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   n_dims,
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    திரும்பு ggml_rope_impl(
        ctx, a, b, வெற்று, n_dims, வெற்று, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
நிலையான தசம ggml_rope_yarn_corr_dim(முழு n_dims, முழு n_ctx_orig, தசம n_rot, தசம base) {
    திரும்பு n_dims * logf(n_ctx_orig / (n_rot * 2 * (தசம)M_PI)) / (2 * logf(base));
}

வெறுமை ggml_rope_yarn_corr_dims(
    முழு n_dims, முழு n_ctx_orig, தசம freq_base, தசம beta_fast, தசம beta_slow, தசம dims[2]
) {
    // start and end correction dims
    தசம start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    தசம end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

கட்டமைப்பு ggml_tensor * ggml_rope_ext_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    கட்டமைப்பு ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_rope_multi_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c,
        முழு                   n_dims,
        முழு                   sections[4],
        முழு                   mode,
        முழு                   n_ctx_orig,
        தசம                 freq_base,
        தசம                 freq_scale,
        தசம                 ext_factor,
        தசம                 attn_factor,
        தசம                 beta_fast,
        தசம                 beta_slow) {
    கட்டமைப்பு ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_rope_set_offset(
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    மாறிலி int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    திரும்பு a;
}

நிலையான int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, முழு s, முழு p, முழு d) {
    திரும்பு (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
கட்டமைப்பு ggml_tensor * ggml_im2col(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   s1,
        முழு                   p0,
        முழு                   p1,
        முழு                   d0,
        முழு                   d1,
        bool                  is_2D,
        எண்ணிக்கை ggml_type        dst_type) {
    எனில் (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } இல்லையெனில் {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    மாறிலி int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    மாறிலி int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    மாறிலி int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_im2col_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        int64_t             * ne,
        முழு                   s0,
        முழு                   s1,
        முழு                   p0,
        முழு                   p1,
        முழு                   d0,
        முழு                   d1,
        bool                  is_2D) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_conv_1d

கட்டமைப்பு ggml_tensor * ggml_conv_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   p0,
        முழு                   d0) {
    கட்டமைப்பு ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    கட்டமைப்பு ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    திரும்பு result;
}

// ggml_conv_1d_ph

கட்டமைப்பு ggml_tensor* ggml_conv_1d_ph(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s,
        முழு                   d) {
    திரும்பு ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

கட்டமைப்பு ggml_tensor * ggml_conv_1d_dw(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   p0,
        முழு                   d0) {
    கட்டமைப்பு ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    கட்டமைப்பு ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    கட்டமைப்பு ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    திரும்பு result;
}

// ggml_conv_1d_dw_ph

கட்டமைப்பு ggml_tensor * ggml_conv_1d_dw_ph(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   d0) {
    திரும்பு ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

கட்டமைப்பு ggml_tensor * ggml_col2im_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   s0,
        முழு                   oc,
        முழு                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    மாறிலி int64_t K_OC = a->ne[0];
    மாறிலி int64_t T_in = a->ne[1];
    மாறிலி int64_t K = K_OC / oc;
    மாறிலி int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    மாறிலி int64_t ne[4] = { T_out, oc, 1, 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_conv_transpose_1d

நிலையான int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, முழு s, முழு p, முழு d) {
    திரும்பு (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API கட்டமைப்பு ggml_tensor * ggml_conv_transpose_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   p0,
        முழு                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    மாறிலி int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
கட்டமைப்பு ggml_tensor * ggml_conv_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   s1,
        முழு                   p0,
        முழு                   p1,
        முழு                   d0,
        முழு                   d1) {
    கட்டமைப்பு ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    கட்டமைப்பு ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    திரும்பு result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
கட்டமைப்பு ggml_tensor * ggml_im2col_3d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        int64_t               IC,
        முழு                   s0, // stride width
        முழு                   s1, // stride height
        முழு                   s2, // stride depth
        முழு                   p0, // padding width
        முழு                   p1, // padding height
        முழு                   p2, // padding depth
        முழு                   d0, // dilation width
        முழு                   d1, // dilation height
        முழு                   d2, // dilation depth
        எண்ணிக்கை ggml_type        dst_type) {
    மாறிலி int64_t N = b->ne[3] / IC;
    மாறிலி int64_t ID = b->ne[2];
    மாறிலி int64_t IH = b->ne[1];
    மாறிலி int64_t IW = b->ne[0];

    மாறிலி int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    மாறிலி int64_t KD = a->ne[2];
    மாறிலி int64_t KH = a->ne[1];
    மாறிலி int64_t KW = a->ne[0];
    மாறிலி int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    மாறிலி int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    மாறிலி int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    மாறிலி int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
கட்டமைப்பு ggml_tensor * ggml_conv_3d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        int64_t               IC,
        முழு                   s0, // stride width
        முழு                   s1, // stride height
        முழு                   s2, // stride depth
        முழு                   p0, // padding width
        முழு                   p1, // padding height
        முழு                   p2, // padding depth
        முழு                   d0, // dilation width
        முழு                   d1, // dilation height
        முழு                   d2  // dilation depth
        ) {
    கட்டமைப்பு ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    கட்டமைப்பு ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    திரும்பு result;
}

// ggml_conv_2d_sk_p0

கட்டமைப்பு ggml_tensor * ggml_conv_2d_sk_p0(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

கட்டமைப்பு ggml_tensor * ggml_conv_2d_s1_ph(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    திரும்பு ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

கட்டமைப்பு ggml_tensor * ggml_conv_2d_dw(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   s1,
        முழு                   p0,
        முழு                   p1,
        முழு                   d0,
        முழு                   d1) {
    கட்டமைப்பு ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    கட்டமைப்பு ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    கட்டமைப்பு ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    கட்டமைப்பு ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    திரும்பு result;
}

// ggml_conv_2d_dw_direct

கட்டமைப்பு ggml_tensor * ggml_conv_2d_dw_direct(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   stride0,
        முழு                   stride1,
        முழு                   pad0,
        முழு                   pad1,
        முழு                   dilation0,
        முழு                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    எனில் (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        மாறிலி int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    திரும்பு result;
}

// ggml_conv_2d_direct

கட்டமைப்பு ggml_tensor * ggml_conv_2d_direct(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        கட்டமைப்பு ggml_tensor  * b,   // input data [W, H, C, N]
        முழு                   s0,  // stride dimension 0
        முழு                   s1,  // stride dimension 1
        முழு                   p0,  // padding dimension 0
        முழு                   p1,  // padding dimension 1
        முழு                   d0,  // dilation dimension 0
        முழு                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_conv_3d_direct

கட்டமைப்பு ggml_tensor * ggml_conv_3d_direct(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   s0,
        முழு                   s1,
        முழு                   s2,
        முழு                   p0,
        முழு                   p1,
        முழு                   p2,
        முழு                   d0,
        முழு                   d1,
        முழு                   d2,
        முழு                   c,
        முழு                   n,
        முழு                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    திரும்பு result;
}

// ggml_conv_transpose_2d_p0

நிலையான int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, முழு s, முழு p) {
    திரும்பு (ins - 1) * s - 2 * p + ks;
}

கட்டமைப்பு ggml_tensor * ggml_conv_transpose_2d_p0(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        முழு                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    மாறிலி int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    கட்டமைப்பு ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_pool_*

நிலையான int64_t ggml_calc_pool_output_size(int64_t ins, முழு ks, முழு s, தசம p) {
    திரும்பு (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

கட்டமைப்பு ggml_tensor * ggml_pool_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_op_pool     op,
        முழு                   k0,
        முழு                   s0,
        முழு                   p0) {
    மாறிலி int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_pool_2d

கட்டமைப்பு ggml_tensor * ggml_pool_2d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_op_pool     op,
        முழு                   k0,
        முழு                   k1,
        முழு                   s0,
        முழு                   s1,
        தசம                 p0,
        தசம                 p1) {
    கட்டமைப்பு ggml_tensor * result;
    மாறிலி int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_pool_2d_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * af,
        எண்ணிக்கை ggml_op_pool     op,
        முழு                   k0,
        முழு                   k1,
        முழு                   s0,
        முழு                   s1,
        தசம                 p0,
        தசம                 p1) {
    கட்டமைப்பு ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    திரும்பு result;
}

// ggml_upscale / ggml_interpolate

நிலையான கட்டமைப்பு ggml_tensor * ggml_interpolate_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_மொத்தம்);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_upscale(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   scale_factor,
        எண்ணிக்கை ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    திரும்பு ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

கட்டமைப்பு ggml_tensor * ggml_upscale_ext(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   ne0,
        முழு                   ne1,
        முழு                   ne2,
        முழு                   ne3,
        எண்ணிக்கை ggml_scale_mode  mode) {
    திரும்பு ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

கட்டமைப்பு ggml_tensor * ggml_interpolate(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    திரும்பு ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

கட்டமைப்பு ggml_tensor * ggml_pad(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   p0,
        முழு                   p1,
        முழு                   p2,
        முழு                   p3) {
    திரும்பு ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

கட்டமைப்பு ggml_tensor * ggml_pad_circular(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   p0,
        முழு                   p1,
        முழு                   p2,
        முழு                   p3) {
    திரும்பு ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

கட்டமைப்பு ggml_tensor * ggml_pad_ext(
            கட்டமைப்பு ggml_context * ctx,
            கட்டமைப்பு ggml_tensor  * a,
            முழு                  lp0,
            முழு                  rp0,
            முழு                  lp1,
            முழு                  rp1,
            முழு                  lp2,
            முழு                  rp2,
            முழு                  lp3,
            முழு                  rp3
            ) {
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    திரும்பு result;
}

// ggml_pad_ext_circular

கட்டமைப்பு ggml_tensor * ggml_pad_ext_circular(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                  lp0,
        முழு                  rp0,
        முழு                  lp1,
        முழு                  rp1,
        முழு                  lp2,
        முழு                  rp2,
        முழு                  lp3,
        முழு                  rp3
        ) {
    கட்டமைப்பு ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    திரும்பு result;
}

// ggml_pad_reflect_1d

கட்டமைப்பு ggml_tensor * ggml_pad_reflect_1d(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   p0,
        முழு                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_roll

கட்டமைப்பு ggml_tensor * ggml_roll(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   shift0,
        முழு                   shift1,
        முழு                   shift2,
        முழு                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_timestep_embedding

கட்டமைப்பு ggml_tensor * ggml_timestep_embedding(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * timesteps,
        முழு                   dim,
        முழு                   max_period) {

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    திரும்பு result;
}

// ggml_tri

கட்டமைப்பு ggml_tensor * ggml_tri(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    எண்ணிக்கை ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_fill

நிலையான கட்டமைப்பு ggml_tensor * ggml_fill_impl(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    தசம                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_fill(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    தசம                 c) {
    திரும்பு ggml_fill_impl(ctx, a, c, false);
}

கட்டமைப்பு ggml_tensor * ggml_fill_inplace(
    கட்டமைப்பு ggml_context * ctx,
    கட்டமைப்பு ggml_tensor  * a,
    தசம                 c) {
    திரும்பு ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

கட்டமைப்பு ggml_tensor * ggml_argsort(
        கட்டமைப்பு ggml_context  * ctx,
        கட்டமைப்பு ggml_tensor   * a,
        எண்ணிக்கை ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_argsort_top_k

கட்டமைப்பு ggml_tensor * ggml_argsort_உச்சி_k(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    கட்டமைப்பு ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    திரும்பு result;
}

// ggml_top_k

கட்டமைப்பு ggml_tensor * ggml_உச்சி_k(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_உச்சி_K;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_arange

கட்டமைப்பு ggml_tensor * ggml_arange(
        கட்டமைப்பு ggml_context * ctx,
        தசம                 start,
        தசம                 stop,
        தசம                 step) {
    GGML_ASSERT(stop > start);

    மாறிலி int64_t steps = (int64_t) ceilf((stop - start) / step);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    திரும்பு result;
}

// ggml_flash_attn_ext

கட்டமைப்பு ggml_tensor * ggml_flash_attn_ext(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * q,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * mask,
        தசம                 scale,
        தசம                 max_bias,
        தசம                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    எனில் (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    எனில் (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    தசம params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    திரும்பு result;
}


வெறுமை ggml_flash_attn_ext_set_prec(
        கட்டமைப்பு ggml_tensor * a,
        எண்ணிக்கை ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    மாறிலி int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

எண்ணிக்கை ggml_prec ggml_flash_attn_ext_get_prec(
        மாறிலி கட்டமைப்பு ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    மாறிலி int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    திரும்பு (எண்ணிக்கை ggml_prec) prec_i32;
}

வெறுமை ggml_flash_attn_ext_set_n_kv_max(
        கட்டமைப்பு ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

வெறுமை ggml_flash_attn_ext_add_sinks(
        கட்டமைப்பு ggml_tensor * a,
        கட்டமைப்பு ggml_tensor * sinks) {
    எனில் (!sinks) {
        a->src[4] = வெற்று;
        திரும்பு;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == வெற்று);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

கட்டமைப்பு ggml_tensor * ggml_flash_attn_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * q,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    மாறிலி int64_t     D = q->ne[0];
    மாறிலி int64_t     N = q->ne[1];
    மாறிலி int64_t     M = k->ne[1];
    மாறிலி int64_t   ne2 = q->ne[2];
    மாறிலி int64_t   ne3 = q->ne[3];
    மாறிலி int64_t kvne2 = k->ne[2];

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
    மாறிலி int64_t elem_q = ggml_nelements(q);
    மாறிலி int64_t elem_k = ggml_nelements(k);
    மாறிலி int64_t elem_v = ggml_nelements(v);

    எண்ணிக்கை ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    மாறிலி size_t tsize = ggml_type_size(result_type);

    மாறிலி size_t offs_q = 0;
    மாறிலி size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    மாறிலி size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    மாறிலி size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    மாறிலி size_t nelements = (end + tsize - 1)/tsize;

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, அளவு(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    திரும்பு result;
}

// ggml_ssm_conv

கட்டமைப்பு ggml_tensor * ggml_ssm_conv(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * sx,
        கட்டமைப்பு ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    மாறிலி int64_t d_conv  = c->ne[0];
    மாறிலி int64_t d_inner = c->ne[1];
    மாறிலி int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    மாறிலி int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    திரும்பு result;
}

// ggml_ssm_scan

கட்டமைப்பு ggml_tensor * ggml_ssm_scan(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * s,
        கட்டமைப்பு ggml_tensor  * x,
        கட்டமைப்பு ggml_tensor  * dt,
        கட்டமைப்பு ggml_tensor  * A,
        கட்டமைப்பு ggml_tensor  * B,
        கட்டமைப்பு ggml_tensor  * C,
        கட்டமைப்பு ggml_tensor  * ids,
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
        மாறிலி int64_t d_state      = s->ne[0];
        மாறிலி int64_t head_dim     = x->ne[0];
        மாறிலி int64_t n_head       = x->ne[1];
        மாறிலி int64_t n_seq_tokens = x->ne[2];
        மாறிலி int64_t n_seqs       = x->ne[3];

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

        எனில் (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    திரும்பு result;
}

// ggml_win_part

கட்டமைப்பு ggml_tensor * ggml_win_part(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    மாறிலி முழு px = (w - a->ne[1]%w)%w;
    மாறிலி முழு py = (w - a->ne[2]%w)%w;

    மாறிலி முழு npx = (px + a->ne[1])/w;
    மாறிலி முழு npy = (py + a->ne[2])/w;
    மாறிலி முழு np  = npx*npy;

    மாறிலி int64_t ne[4] = { a->ne[0], w, w, np, };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_win_unpart

கட்டமைப்பு ggml_tensor * ggml_win_unpart(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   w0,
        முழு                   h0,
        முழு                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    மாறிலி int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, அளவு(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_get_rel_pos

கட்டமைப்பு ggml_tensor * ggml_get_rel_pos(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        முழு                   qh,
        முழு                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    மாறிலி int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    திரும்பு result;
}

// ggml_add_rel_pos

நிலையான கட்டமைப்பு ggml_tensor * ggml_add_rel_pos_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * pw,
        கட்டமைப்பு ggml_tensor  * ph,
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

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_add_rel_pos(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * pw,
        கட்டமைப்பு ggml_tensor  * ph) {
    திரும்பு ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

கட்டமைப்பு ggml_tensor * ggml_add_rel_pos_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * pw,
        கட்டமைப்பு ggml_tensor  * ph) {
    திரும்பு ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

கட்டமைப்பு ggml_tensor * ggml_rwkv_wkv6(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * r,
        கட்டமைப்பு ggml_tensor  * tf,
        கட்டமைப்பு ggml_tensor  * td,
        கட்டமைப்பு ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    மாறிலி int64_t S = k->ne[0];
    மாறிலி int64_t H = k->ne[1];
    மாறிலி int64_t n_tokens = k->ne[2];
    மாறிலி int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    மாறிலி int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    திரும்பு result;
}

// ggml_gated_linear_attn

கட்டமைப்பு ggml_tensor * ggml_gated_linear_attn(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * q,
        கட்டமைப்பு ggml_tensor  * g,
        கட்டமைப்பு ggml_tensor  * state,
        தசம scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    மாறிலி int64_t S = k->ne[0];
    மாறிலி int64_t H = k->ne[1];
    மாறிலி int64_t n_tokens = k->ne[2];
    மாறிலி int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    மாறிலி int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    திரும்பு result;
}

// ggml_rwkv_wkv7

கட்டமைப்பு ggml_tensor * ggml_rwkv_wkv7(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * r,
        கட்டமைப்பு ggml_tensor  * w,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    மாறிலி int64_t S = k->ne[0];
    மாறிலி int64_t H = k->ne[1];
    மாறிலி int64_t n_tokens = k->ne[2];
    மாறிலி int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    மாறிலி int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    திரும்பு result;
}

// ggml_unary

நிலையான கட்டமைப்பு ggml_tensor * ggml_unary_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_unary(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_unary_op    op) {
    திரும்பு ggml_unary_impl(ctx, a, op, false);
}

கட்டமைப்பு ggml_tensor * ggml_unary_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        எண்ணிக்கை ggml_unary_op    op) {
    திரும்பு ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

நிலையான கட்டமைப்பு ggml_tensor * ggml_map_custom1_impl(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        மாறிலி  ggml_custom1_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    கட்டமைப்பு ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_map_custom1(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        மாறிலி  ggml_custom1_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

கட்டமைப்பு ggml_tensor * ggml_map_custom1_inplace(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        மாறிலி  ggml_custom1_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

நிலையான கட்டமைப்பு ggml_tensor * ggml_map_custom2_impl(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        மாறிலி  ggml_custom2_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    கட்டமைப்பு ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_map_custom2(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        மாறிலி  ggml_custom2_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

கட்டமைப்பு ggml_tensor * ggml_map_custom2_inplace(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        மாறிலி  ggml_custom2_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

நிலையான கட்டமைப்பு ggml_tensor * ggml_map_custom3_impl(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        கட்டமைப்பு ggml_tensor       * c,
        மாறிலி  ggml_custom3_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    கட்டமைப்பு ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    கட்டமைப்பு ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_map_custom3(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        கட்டமைப்பு ggml_tensor       * c,
        மாறிலி  ggml_custom3_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

கட்டமைப்பு ggml_tensor * ggml_map_custom3_inplace(
        கட்டமைப்பு ggml_context      * ctx,
        கட்டமைப்பு ggml_tensor       * a,
        கட்டமைப்பு ggml_tensor       * b,
        கட்டமைப்பு ggml_tensor       * c,
        மாறிலி  ggml_custom3_op_t   fun,
        முழு                        n_tasks,
        வெறுமை                     * userdata) {
    திரும்பு ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

கட்டமைப்பு ggml_tensor * ggml_custom_4d(
        கட்டமைப்பு ggml_context * ctx,
        எண்ணிக்கை ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        கட்டமைப்பு ggml_tensor ** args,
        முழு                   n_args,
        ggml_custom_op_t      fun,
        முழு                   n_tasks,
        வெறுமை                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    கட்டமைப்பு ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op = GGML_OP_CUSTOM;
    க்கு (முழு i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_custom_inplace(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor ** args,
        முழு                   n_args,
        ggml_custom_op_t      fun,
        முழு                   n_tasks,
        வெறுமை                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);

    கட்டமைப்பு ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, அளவு(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    க்கு (முழு i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    திரும்பு result;
}
// ggml_cross_entropy_loss

கட்டமைப்பு ggml_tensor * ggml_cross_entropy_loss(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_cross_entropy_loss_back

கட்டமைப்பு ggml_tensor * ggml_cross_entropy_loss_back(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        கட்டமைப்பு ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    கட்டமைப்பு ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    திரும்பு result;
}

// opt_step_adamw

கட்டமைப்பு ggml_tensor * ggml_opt_step_adamw(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * grad,
        கட்டமைப்பு ggml_tensor  * m,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    திரும்பு result;
}

// opt_step_sgd

கட்டமைப்பு ggml_tensor * ggml_opt_step_sgd(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * grad,
        கட்டமைப்பு ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    கட்டமைப்பு ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    திரும்பு result;
}

// solve_tri

கட்டமைப்பு ggml_tensor * ggml_solve_tri(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * a,
        கட்டமைப்பு ggml_tensor  * b,
        bool                  இடது,
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

    GGML_ASSERT(lower && இடது && !uni); // TODO: support other variants

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    திரும்பு result;
}

// ggml_gated_delta_net

கட்டமைப்பு ggml_tensor * ggml_gated_delta_net(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * q,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * v,
        கட்டமைப்பு ggml_tensor  * g,
        கட்டமைப்பு ggml_tensor  * beta,
        கட்டமைப்பு ggml_tensor  * state,
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

    மாறிலி int64_t S_v      = v->ne[0];
    மாறிலி int64_t H        = v->ne[1];
    மாறிலி int64_t n_tokens = v->ne[2];
    மாறிலி int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    மாறிலி int64_t state_rows = K * S_v * n_seqs;
    மாறிலி int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    திரும்பு result;
}

// ggml_lightning_indexer

கட்டமைப்பு ggml_tensor * ggml_lightning_indexer(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * q,
        கட்டமைப்பு ggml_tensor  * k,
        கட்டமைப்பு ggml_tensor  * weights,
        கட்டமைப்பு ggml_tensor  * mask) {

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
    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    திரும்பு result;
}

// ggml_dsv4_hc_comb

கட்டமைப்பு ggml_tensor * ggml_dsv4_hc_comb(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * mixes,
        கட்டமைப்பு ggml_tensor  * scale,
        கட்டமைப்பு ggml_tensor  * base,
        தசம                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    மாறிலி int64_t hc_mix_dim = mixes->ne[0];
    மாறிலி int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    க்கு (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        எனில் ((2 + i)*i == hc_mix_dim) {
            hc = i;
            நிறுத்து;
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

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    திரும்பு result;
}

// ggml_dsv4_hc_pre

நிலையான கட்டமைப்பு ggml_tensor * ggml_dsv4_hc_pre_impl(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * x,
        கட்டமைப்பு ggml_tensor  * weights,
        தசம                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t hc       = x->ne[1];
    மாறிலி int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    எனில் (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } இல்லையெனில் {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_dsv4_hc_pre(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * x,
        கட்டமைப்பு ggml_tensor  * weights) {
    திரும்பு ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

கட்டமைப்பு ggml_tensor * ggml_dsv4_hc_pre_gated(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * x,
        கட்டமைப்பு ggml_tensor  * gate,
        தசம                 scale) {
    திரும்பு ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

கட்டமைப்பு ggml_tensor * ggml_dsv4_hc_post(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * x,
        கட்டமைப்பு ggml_tensor  * residual,
        கட்டமைப்பு ggml_tensor  * post,
        கட்டமைப்பு ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t n_tokens = x->ne[1];
    மாறிலி int64_t hc       = residual->ne[1];

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

    எனில் (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    திரும்பு result;
}

// ggml_top_k_mask

கட்டமைப்பு ggml_tensor * ggml_உச்சி_k_mask(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_tensor  * mask,
        கட்டமைப்பு ggml_tensor  * உச்சி_k,
        கட்டமைப்பு ggml_tensor  * score,
        கட்டமைப்பு ggml_tensor  * blocks,
        கட்டமைப்பு ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(உச்சி_k->type == GGML_TYPE_I32);
    GGML_ASSERT(உச்சி_k->ne[1] % mask->ne[1] == 0 && உச்சி_k->ne[2] % mask->ne[2] == 0 && உச்சி_k->ne[3] % mask->ne[3] == 0);

    எனில் (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == உச்சி_k->ne[1] && score->ne[2] == உச்சி_k->ne[2] && score->ne[3] == உச்சி_k->ne[3]);
    }

    எனில் (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    எனில் (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == உச்சி_k->ne[1] && extra->ne[2] == உச்சி_k->ne[2] && extra->ne[3] == உச்சி_k->ne[3]);
    }

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], உச்சி_k->ne[1], உச்சி_k->ne[2], உச்சி_k->ne[3]);

    result->op     = GGML_OP_உச்சி_K_MASK;
    result->src[0] = mask;
    result->src[1] = உச்சி_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    திரும்பு result;
}

////////////////////////////////////////////////////////////////////////////////

கட்டமைப்பு ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    கட்டமைப்பு ggml_hash_set result;
    result.size = size;
    result.சாவிகள் = GGML_MALLOC(அளவு(கட்டமைப்பு ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), அளவு(ggml_bitset_t));
    திரும்பு result;
}

வெறுமை ggml_hash_set_reset(கட்டமைப்பு ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, அளவு(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

வெறுமை ggml_hash_set_free(கட்டமைப்பு ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->சாவிகள்);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    நிலையான மாறிலி size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    நிலையான மாறிலி size_t n_primes = அளவு(primes)/அளவு(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    போது (l < r) {
        size_t m = (l + r)/2;
        எனில் (primes[m] < min_sz) {
            l = m + 1;
        } இல்லையெனில் {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    திரும்பு sz;
}

கட்டமைப்பு hash_map {
    கட்டமைப்பு ggml_hash_set set;
    கட்டமைப்பு ggml_tensor ** vals;
};

நிலையான கட்டமைப்பு hash_map * ggml_new_hash_map(size_t size) {
    கட்டமைப்பு hash_map * result = GGML_MALLOC(அளவு(கட்டமைப்பு hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, அளவு(கட்டமைப்பு ggml_tensor *));
    திரும்பு result;
}

நிலையான வெறுமை ggml_hash_map_free(கட்டமைப்பு hash_map * map) {
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

நிலையான வெறுமை ggml_add_or_set(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_cgraph  * cgraph,
        size_t                isrc,
        கட்டமைப்பு ggml_tensor  * tensor) {
    கட்டமைப்பு ggml_tensor * src = cgraph->visited_hash_set.சாவிகள்[isrc];
    GGML_ASSERT(src);
    எனில் (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } இல்லையெனில் {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

நிலையான வெறுமை ggml_acc_or_set(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_cgraph  * cgraph,
        size_t                isrc,
        கட்டமைப்பு ggml_tensor  * tensor,
        மாறிலி  size_t         nb1,
        மாறிலி  size_t         nb2,
        மாறிலி  size_t         nb3,
        மாறிலி  size_t         offset) {
    கட்டமைப்பு ggml_tensor * src = cgraph->visited_hash_set.சாவிகள்[isrc];
    GGML_ASSERT(src);
    எனில் (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } இல்லையெனில் {
        கட்டமைப்பு ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.சாவிகள்[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

நிலையான வெறுமை ggml_add1_or_set(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_cgraph  * cgraph,
        size_t                isrc,
        கட்டமைப்பு ggml_tensor  * tensor) {
    கட்டமைப்பு ggml_tensor * src = cgraph->visited_hash_set.சாவிகள்[isrc];
    GGML_ASSERT(src);
    எனில் (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } இல்லையெனில் {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

நிலையான வெறுமை ggml_sub_or_set(
        கட்டமைப்பு ggml_context * ctx,
        கட்டமைப்பு ggml_cgraph  * cgraph,
        size_t                isrc,
        கட்டமைப்பு ggml_tensor  * tensor) {
    கட்டமைப்பு ggml_tensor * src = cgraph->visited_hash_set.சாவிகள்[isrc];
    GGML_ASSERT(src);
    எனில் (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } இல்லையெனில் {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

நிலையான வெறுமை ggml_compute_backward(
        கட்டமைப்பு ggml_context * ctx, கட்டமைப்பு ggml_cgraph * cgraph, முழு i, மாறிலி bool * grads_needed) {
    கட்டமைப்பு ggml_tensor * tensor = cgraph->nodes[i];
    கட்டமைப்பு ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    எனில் (!grad) {
        திரும்பு;
    }

    கட்டமைப்பு ggml_tensor * src0 = tensor->src[0];
    கட்டமைப்பு ggml_tensor * src1 = tensor->src[1];
    கட்டமைப்பு ggml_tensor * src2 = tensor->src[2];
    கட்டமைப்பு ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    மாறிலி size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    மாறிலி size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    மாறிலி size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    மாறிலி bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    மாறிலி bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    மாறிலி bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    தேர்வு (tensor->op) {
        நிலை GGML_OP_DUP: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } நிறுத்து;
        நிலை GGML_OP_ADD: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            எனில் (src1_needs_grads) {
                கட்டமைப்பு ggml_tensor * tmp = grad;
                எனில் (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } நிறுத்து;
        நிலை GGML_OP_ADD1: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            எனில் (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } நிறுத்து;
        நிலை GGML_OP_ACC: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            எனில் (src1_needs_grads) {
                மாறிலி size_t nb1    = ((int32_t *) tensor->op_params)[0];
                மாறிலி size_t nb2    = ((int32_t *) tensor->op_params)[1];
                மாறிலி size_t nb3    = ((int32_t *) tensor->op_params)[2];
                மாறிலி size_t offset = ((int32_t *) tensor->op_params)[3];

                கட்டமைப்பு ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } நிறுத்து;
        நிலை GGML_OP_SUB: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            எனில் (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } நிறுத்து;
        நிலை GGML_OP_MUL: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            எனில் (src1_needs_grads) {
                கட்டமைப்பு ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                எனில் (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } நிறுத்து;
        நிலை GGML_OP_DIV: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            எனில் (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } நிறுத்து;
        நிலை GGML_OP_SQR: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } நிறுத்து;
        நிலை GGML_OP_SQRT: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } நிறுத்து;
        நிலை GGML_OP_LOG: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_SIN: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } நிறுத்து;
        நிலை GGML_OP_COS: {
            எனில் (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } நிறுத்து;
        நிலை GGML_OP_SUM: {
            எனில் (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } நிறுத்து;
        நிலை GGML_OP_SUM_ROWS: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_MEAN: {
            எனில் (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } நிறுத்து;
        நிலை GGML_OP_REPEAT: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_REPEAT_BACK: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_RMS_NORM: {
            எனில் (src0_needs_grads) {
                தசம eps;
                memcpy(&eps, tensor->op_params, அளவு(தசம));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } நிறுத்து;
        நிலை GGML_OP_MUL_MAT: {
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

            எனில் (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                கட்டமைப்பு ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                எனில் (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    மாறிலி int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    மாறிலி size_t nb2 = tmp->nb[2] * nr2;
                    மாறிலி size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            எனில் (src1_needs_grads) {
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
        } நிறுத்து;
        நிலை GGML_OP_SCALE: {
            எனில் (src0_needs_grads) {
                தசம s;
                memcpy(&s, tensor->op_params, அளவு(தசம));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } நிறுத்து;
        நிலை GGML_OP_SET: {
            மாறிலி size_t nb1    = ((மாறிலி int32_t *) tensor->op_params)[0];
            மாறிலி size_t nb2    = ((மாறிலி int32_t *) tensor->op_params)[1];
            மாறிலி size_t nb3    = ((மாறிலி int32_t *) tensor->op_params)[2];
            மாறிலி size_t offset = ((மாறிலி int32_t *) tensor->op_params)[3];

            கட்டமைப்பு ggml_tensor * tensor_grad_view = வெற்று;

            எனில் (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            எனில் (src0_needs_grads) {
                கட்டமைப்பு ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            எனில் (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } நிறுத்து;
        நிலை GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            எனில் (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            எனில் (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } நிறுத்து;
        நிலை GGML_OP_CONT: {
            // same as cpy
            எனில் (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_RESHAPE: {
            எனில் (src0_needs_grads) {
                கட்டமைப்பு ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } நிறுத்து;
        நிலை GGML_OP_VIEW: {
            எனில் (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, அளவு(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                எனில் (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } நிறுத்து;
        நிலை GGML_OP_PERMUTE: {
            எனில் (src0_needs_grads) {
                மாறிலி int32_t * axes = (மாறிலி int32_t *) tensor->op_params;
                மாறிலி முழு axis0 = axes[0] & 0x3;
                மாறிலி முழு axis1 = axes[1] & 0x3;
                மாறிலி முழு axis2 = axes[2] & 0x3;
                மாறிலி முழு axis3 = axes[3] & 0x3;
                முழு axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } நிறுத்து;
        நிலை GGML_OP_TRANSPOSE: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } நிறுத்து;
        நிலை GGML_OP_GET_ROWS: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            எனில் (src1_needs_grads) {
                // noop
            }
        } நிறுத்து;
        நிலை GGML_OP_DIAG_MASK_INF: {
            எனில் (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                மாறிலி முழு n_past = ((மாறிலி int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } நிறுத்து;
        நிலை GGML_OP_DIAG_MASK_ZERO: {
            எனில் (src0_needs_grads) {
                மாறிலி முழு n_past = ((மாறிலி int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } நிறுத்து;
        நிலை GGML_OP_SOFT_MAX: {
            எனில் (src0_needs_grads) {
                தசம scale    = 1.0f;
                தசம max_bias = 0.0f;

                memcpy(&scale,    (மாறிலி தசம *) tensor->op_params + 0, அளவு(தசம));
                memcpy(&max_bias, (மாறிலி தசம *) tensor->op_params + 1, அளவு(தசம));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } நிறுத்து;
        நிலை GGML_OP_ROPE: {
            எனில் (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                மாறிலி முழு n_dims     = ((மாறிலி int32_t *) tensor->op_params)[1];
                மாறிலி முழு mode       = ((மாறிலி int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                மாறிலி முழு n_ctx_orig = ((மாறிலி int32_t *) tensor->op_params)[4];
                தசம freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                முழு sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (மாறிலி தசம *) tensor->op_params +  5, அளவு(தசம));
                memcpy(&freq_scale,  (மாறிலி தசம *) tensor->op_params +  6, அளவு(தசம));
                memcpy(&ext_factor,  (மாறிலி தசம *) tensor->op_params +  7, அளவு(தசம));
                memcpy(&attn_factor, (மாறிலி தசம *) tensor->op_params +  8, அளவு(தசம));
                memcpy(&beta_fast,   (மாறிலி தசம *) tensor->op_params +  9, அளவு(தசம));
                memcpy(&beta_slow,   (மாறிலி தசம *) tensor->op_params + 10, அளவு(தசம));
                memcpy(&sections,                    tensor->op_params + 11, அளவு(sections));

                கட்டமைப்பு ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } நிறுத்து;
        நிலை GGML_OP_IM2COL: {
            எனில் (src1_needs_grads) {
                மாறிலி int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                மாறிலி int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                மாறிலி int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                மாறிலி int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                மாறிலி int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                மாறிலி int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                மாறிலி bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } நிறுத்து;
        நிலை GGML_OP_POOL_2D: {
            எனில் (src0_needs_grads) {
                மாறிலி எண்ணிக்கை ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                மாறிலி      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                மாறிலி      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                மாறிலி      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                மாறிலி      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                மாறிலி      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                மாறிலி      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } நிறுத்து;
        நிலை GGML_OP_WIN_PART:
        நிலை GGML_OP_WIN_UNPART:
        நிலை GGML_OP_UNARY: {
            தேர்வு (ggml_get_unary_op(tensor)) {
                நிலை GGML_UNARY_OP_ABS: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_SGN: {
                    // noop
                } நிறுத்து;
                நிலை GGML_UNARY_OP_NEG: {
                    எனில் (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_STEP: {
                    // noop
                } நிறுத்து;
                நிலை GGML_UNARY_OP_RELU: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_SILU: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_EXP: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_EXPM1: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } நிறுத்து;
                நிலை GGML_UNARY_OP_SOFTPLUS: {
                    எனில் (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } நிறுத்து;
                இயல்பு: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } நிறுத்து;
        நிலை GGML_OP_CROSS_ENTROPY_LOSS: {
            எனில் (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } நிறுத்து;
        நிலை GGML_OP_GLU: {
            தேர்வு (ggml_get_glu_op(tensor)) {
                நிலை GGML_GLU_OP_SWIGLU: {
                    எனில் (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    எனில் (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } நிறுத்து;
                இயல்பு: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } நிறுத்து;
        நிலை GGML_OP_NONE: {
            // noop
        } நிறுத்து;
        நிலை GGML_OP_மொத்தம்:
        இயல்பு: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

நிலையான size_t ggml_visit_parents_graph(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_tensor * முனை, bool compute) {
    எனில் (முனை->op != GGML_OP_NONE && compute) {
        முனை->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    மாறிலி size_t முனை_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, முனை);
    GGML_ASSERT(முனை_hash_pos != GGML_HASHSET_FULL);

    எனில் (ggml_bitset_get(cgraph->visited_hash_set.used, முனை_hash_pos)) {
        // already visited

        எனில் (compute) {
            // update the compute flag regardless
            க்கு (முழு i = 0; i < GGML_MAX_SRC; ++i) {
                கட்டமைப்பு ggml_tensor * src = முனை->src[i];
                எனில் (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        திரும்பு முனை_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.சாவிகள்[முனை_hash_pos] = முனை;
    ggml_bitset_set(cgraph->visited_hash_set.used, முனை_hash_pos);
    cgraph->use_counts[முனை_hash_pos] = 0;

    க்கு (முழு i = 0; i < GGML_MAX_SRC; ++i) {
        மாறிலி முழு k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_இடது_TO_வலது) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_வலது_TO_இடது) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        கட்டமைப்பு ggml_tensor * src = முனை->src[k];
        எனில் (src) {
            மாறிலி size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    எனில் (முனை->op == GGML_OP_NONE && !(முனை->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        எனில் (strlen(முனை->name) == 0) {
            ggml_format_name(முனை, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = முனை;
        cgraph->n_leafs++;
    } இல்லையெனில் {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        எனில் (strlen(முனை->name) == 0) {
            ggml_format_name(முனை, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = முனை;
        cgraph->n_nodes++;
    }

    திரும்பு முனை_hash_pos;
}

நிலையான வெறுமை ggml_build_forward_impl(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_tensor * tensor, bool expand, bool compute) {
    எனில் (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    மாறிலி முழு n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    மாறிலி முழு n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    எனில் (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

கட்டமைப்பு ggml_tensor * ggml_build_forward_select(
        கட்டமைப்பு ggml_cgraph  * cgraph,
        கட்டமைப்பு ggml_tensor ** tensors,
        முழு                   n_tensors,
        முழு                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    க்கு (முழு i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    திரும்பு tensors[idx];
}

வெறுமை ggml_build_forward_expand(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

வெறுமை ggml_build_forward_order(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

வெறுமை ggml_build_backward_expand(
        கட்டமைப்பு ggml_context *  ctx,
        கட்டமைப்பு ggml_cgraph  *  cgraph,
        கட்டமைப்பு ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    மாறிலி முழு n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*அளவு(கட்டமைப்பு ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*அளவு(கட்டமைப்பு ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, அளவு(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        க்கு (முழு i = 0; i < n_nodes_f; ++i) {
            கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];
            any_params = any_params || (முனை->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (முனை->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    க்கு (முழு i = 0; i < n_nodes_f; ++i) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        எனில் (முனை->type == GGML_TYPE_I32) {
            தொடர்;
        }

        bool முனை_needs_grad = (முனை->flags & GGML_TENSOR_FLAG_PARAM) || (முனை->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        தேர்வு (முனை->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            நிலை GGML_OP_IM2COL:      // only used for its shape
            நிலை GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                நிறுத்து;
            நிலை GGML_OP_UNARY: {
                மாறிலி எண்ணிக்கை ggml_unary_op uop = ggml_get_unary_op(முனை);
                // SGN and STEP unary ops are piecewise constant
                எனில் (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } நிறுத்து;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            நிலை GGML_OP_CPY:           // gradients in CPY target are irrelevant
            நிலை GGML_OP_GET_ROWS:      // row indices not differentiable
            நிலை GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            நிலை GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                நிறுத்து;

            இயல்பு:
                நிறுத்து;
        }
        க்கு (முழு j = 0; j < GGML_MAX_SRC; ++j) {
            எனில் (!முனை->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, முனை->src[j])]) {
                தொடர்;
            }
            GGML_ASSERT(முனை->src[j]->type == GGML_TYPE_F32 || முனை->src[j]->type == GGML_TYPE_F16);
            முனை_needs_grad = true;
            நிறுத்து;
        }
        எனில் (!முனை_needs_grad) {
            தொடர்;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!முனை->view_src || முனை->op == GGML_OP_CPY || முனை->op == GGML_OP_SET_ROWS || முனை->op == GGML_OP_VIEW ||
            முனை->op == GGML_OP_RESHAPE || முனை->op == GGML_OP_PERMUTE || முனை->op == GGML_OP_TRANSPOSE);

        மாறிலி size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, முனை);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        எனில் (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } இல்லையெனில் எனில் (முனை->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, முனை->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    க்கு (முழு i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    விடுவி(grads_needed);
}

நிலையான வெறுமை * incr_ptr_aligned(வெறுமை ** p, size_t size, size_t align) {
    வெறுமை * ptr = *p;
    ptr = (வெறுமை *) GGML_PAD((uintptr_t) ptr, align);
    *p = (வெறுமை *) ((எழுத்து *) ptr + size);
    திரும்பு ptr;
}

நிலையான size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    வெறுமை * p = (எழுத்து *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, அளவு(கட்டமைப்பு ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * அளவு(int32_t), அளவு(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)); // hash keys
    எனில் (grads) {
        incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * அளவு(ggml_bitset_t), அளவு(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    திரும்பு nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    திரும்பு GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(வெறுமை) {
    திரும்பு ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

கட்டமைப்பு ggml_cgraph * ggml_new_graph_custom(கட்டமைப்பு ggml_context * ctx, size_t size, bool grads) {
    மாறிலி size_t obj_size = ggml_graph_nbytes(size, grads);
    கட்டமைப்பு ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    கட்டமைப்பு ggml_cgraph * cgraph = (கட்டமைப்பு ggml_cgraph *) ((எழுத்து *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    வெறுமை * p = cgraph + 1;

    கட்டமைப்பு ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *));
    கட்டமைப்பு ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * அளவு(int32_t), அளவு(int32_t));
    கட்டமைப்பு ggml_tensor ** hash_சாவிகள்_ptr  =         incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *));
    கட்டமைப்பு ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)) : வெற்று;
    கட்டமைப்பு ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * அளவு(கட்டமைப்பு ggml_tensor *), அளவு(கட்டமைப்பு ggml_tensor *)) : வெற்று;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * அளவு(ggml_bitset_t), அளவு(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ப்ரதிஜ்ஞா(obj_size == (size_t)((எழுத்து *)p - (எழுத்து *)cgraph));

    *cgraph = (கட்டமைப்பு ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_சாவிகள்_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_இடது_TO_வலது,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    எனில் (grads) {
        memset(cgraph->grads,     0, hash_size*அளவு(கட்டமைப்பு ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*அளவு(கட்டமைப்பு ggml_tensor *));
    }

    திரும்பு cgraph;
}

கட்டமைப்பு ggml_cgraph * ggml_new_graph(கட்டமைப்பு ggml_context * ctx) {
    திரும்பு ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

கட்டமைப்பு ggml_cgraph ggml_graph_view(கட்டமைப்பு ggml_cgraph * cgraph0, முழு i0, முழு i1) {
    கட்டமைப்பு ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ வெற்று, // gradients would need visited_hash_set
        /*.grad_accs        =*/ வெற்று,
        /*.leafs            =*/ வெற்று,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    திரும்பு cgraph;
}

வெறுமை ggml_graph_cpy(கட்டமைப்பு ggml_cgraph * src, கட்டமைப்பு ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    க்கு (முழு i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    க்கு (முழு i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    க்கு (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        எனில் (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_செருகு(&dst->visited_hash_set, src->visited_hash_set.சாவிகள்[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    எனில் (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*அளவு(கட்டமைப்பு ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*அளவு(கட்டமைப்பு ggml_tensor *));
    }
    எனில் (src->grads) {
        GGML_ASSERT(dst->grads     != வெற்று);
        GGML_ASSERT(dst->grad_accs != வெற்று);
        க்கு (முழு i = 0; i < src->n_nodes; ++i) {
            மாறிலி size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            மாறிலி size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

கட்டமைப்பு ggml_cgraph * ggml_graph_dup(கட்டமைப்பு ggml_context * ctx, கட்டமைப்பு ggml_cgraph * cgraph, bool force_grads) {
    கட்டமைப்பு ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_set_zero(கட்டமைப்பு ggml_tensor * tensor) {
    எனில் (ggml_is_empty(tensor)) {
        திரும்பு tensor;
    }
    எனில் (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } இல்லையெனில் {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    திரும்பு tensor;
}

வெறுமை ggml_graph_reset(கட்டமைப்பு ggml_cgraph * cgraph) {
    எனில் (!cgraph) {
        திரும்பு;
    }
    GGML_ASSERT(cgraph->grads != வெற்று);

    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை     = cgraph->nodes[i];
        கட்டமைப்பு ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, முனை);

        எனில் (முனை->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(முனை->src[2]);
            ggml_set_zero(முனை->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        எனில் (grad_acc) {
            எனில் (முனை->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                மாறிலி தசம onef = 1.0f;
                எனில் (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, அளவு(தசம));
                } இல்லையெனில் {
                    GGML_ASSERT(grad_acc->data);
                    *((தசம *) grad_acc->data) = onef;
                }
            } இல்லையெனில் {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

வெறுமை ggml_graph_clear(கட்டமைப்பு ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

முழு ggml_graph_size(கட்டமைப்பு ggml_cgraph * cgraph) {
    திரும்பு cgraph->size;
}

கட்டமைப்பு ggml_tensor * ggml_graph_முனை(கட்டமைப்பு ggml_cgraph * cgraph, முழு i) {
    எனில் (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        திரும்பு cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    திரும்பு cgraph->nodes[i];
}

கட்டமைப்பு ggml_tensor ** ggml_graph_nodes(கட்டமைப்பு ggml_cgraph * cgraph) {
    திரும்பு cgraph->nodes;
}

முழு ggml_graph_n_nodes(கட்டமைப்பு ggml_cgraph * cgraph) {
    திரும்பு cgraph->n_nodes;
}

வெறுமை ggml_graph_add_முனை(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

கட்டமைப்பு ggml_tensor * ggml_graph_get_tensor(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி எழுத்து * name) {
    க்கு (முழு i = 0; i < cgraph->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * leaf = cgraph->leafs[i];

        எனில் (strcmp(leaf->name, name) == 0) {
            திரும்பு leaf;
        }
    }

    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        எனில் (strcmp(முனை->name, name) == 0) {
            திரும்பு முனை;
        }
    }

    திரும்பு வெற்று;
}

கட்டமைப்பு ggml_tensor * ggml_graph_get_grad(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி கட்டமைப்பு ggml_tensor * முனை) {
    மாறிலி size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, முனை);
    திரும்பு igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : வெற்று;
}

கட்டமைப்பு ggml_tensor * ggml_graph_get_grad_acc(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி கட்டமைப்பு ggml_tensor * முனை) {
    மாறிலி size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, முனை);
    திரும்பு igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : வெற்று;
}

வெறுமை ggml_graph_print(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                முனை->ne[0], முனை->ne[1], முனை->ne[2],
                ggml_op_name(முனை->op), (முனை->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, முனை) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    க்கு (முழு i = 0; i < cgraph->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                முனை->ne[0], முனை->ne[1],
                ggml_op_name(முனை->op),
                ggml_get_name(முனை));
    }

    GGML_LOG_INFO("========================================\n");
}

நிலையான முழு ggml_முனை_list_find_tensor(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
                                      மாறிலி முழு *                idxs,
                                      முழு                        மொத்தம்,
                                      மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    க்கு (முழு i = 0; i < மொத்தம்; ++i) {
        மாறிலி முழு முனை_idx = idxs[i];

        எனில் (முனை_idx >= cgraph->n_nodes) {
            திரும்பு -1;
        }
        எனில் (cgraph->nodes[முனை_idx] == tensor) {
            திரும்பு i;
        }
    }
    திரும்பு -1;
}

நிலையான bool ggml_is_constant(மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    திரும்பு tensor->buffer != வெற்று && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
                                மாறிலி முழு *                முனை_idxs,
                                முழு                        மொத்தம்,
                                மாறிலி எண்ணிக்கை ggml_op *       ops,
                                மாறிலி முழு *                outputs,
                                முழு                        இலக்கம்_outputs) {
    GGML_ASSERT(outputs && இலக்கம்_outputs > 0);

    க்கு (முழு i = 0; i < மொத்தம்; ++i) {
        எனில் (முனை_idxs[i] >= cgraph->n_nodes) {
            திரும்பு false;
        }

        மாறிலி கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_idxs[i]];

        எனில் (முனை->op != ops[i]) {
            திரும்பு false;
        }

        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            திரும்பு false;
        }

        எனில் (ggml_முனை_list_find_tensor(cgraph, outputs, இலக்கம்_outputs, முனை) != -1) {
            தொடர்;
        }

        எனில் (முனை->flags & GGML_TENSOR_FLAG_OUTPUT) {
            திரும்பு false;
        }

        முழு subgraph_uses = 0;
        க்கு (முழு j = i + 1; j < மொத்தம்; ++j) {
            மாறிலி கட்டமைப்பு ggml_tensor * other_முனை = cgraph->nodes[முனை_idxs[j]];
            க்கு (முழு src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                எனில் (other_முனை->src[src_idx] == முனை) {
                    subgraph_uses++;
                }
            }
        }

        எனில் (subgraph_uses != ggml_முனை_get_use_மொத்தம்(cgraph, முனை_idxs[i])) {
            திரும்பு false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        கட்டமைப்பு ggml_tensor * view_src = முனை->view_src;
        போது (view_src) {
            எனில் (ggml_முனை_list_find_tensor(cgraph, முனை_idxs, மொத்தம், view_src) == -1 && !ggml_is_constant(view_src)) {
                திரும்பு false;
            }
            view_src = view_src->view_src;
        }
    }

    திரும்பு true;
}

// check if node is part of the graph
நிலையான bool ggml_graph_find(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி கட்டமைப்பு ggml_tensor * முனை) {
    எனில் (cgraph == வெற்று) {
        திரும்பு true;
    }

    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        எனில் (cgraph->nodes[i] == முனை) {
            திரும்பு true;
        }
    }

    திரும்பு false;
}

நிலையான கட்டமைப்பு ggml_tensor * ggml_graph_get_parent(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி கட்டமைப்பு ggml_tensor * முனை) {
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * parent = cgraph->nodes[i];
        கட்டமைப்பு ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        எனில் (grad == முனை) {
            திரும்பு parent;
        }
    }

    திரும்பு வெற்று;
}

நிலையான வெறுமை ggml_graph_dump_dot_முனை_edge(FILE * fp, மாறிலி கட்டமைப்பு ggml_cgraph * gb, கட்டமைப்பு ggml_tensor * முனை, கட்டமைப்பு ggml_tensor * parent, மாறிலி எழுத்து * label)  {
    கட்டமைப்பு ggml_tensor * gparent = ggml_graph_get_parent(gb, முனை);
    கட்டமைப்பு ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (வெறுமை *) gparent0 : (வெறுமை *) parent,
            gparent ? (வெறுமை *) gparent : (வெறுமை *) முனை,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

நிலையான வெறுமை ggml_graph_dump_dot_leaf_edge(FILE * fp, கட்டமைப்பு ggml_tensor * முனை, கட்டமைப்பு ggml_tensor * parent, மாறிலி எழுத்து * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (வெறுமை *) parent,
            (வெறுமை *) முனை,
            label);
}

வெறுமை ggml_graph_dump_dot(மாறிலி கட்டமைப்பு ggml_cgraph * gb, மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி எழுத்து * filename) {
    எழுத்து color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    க்கு (முழு i = 0; i < gb->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = gb->nodes[i];
        கட்டமைப்பு ggml_tensor * grad = ggml_graph_get_grad(gb, முனை);

        எனில் (ggml_graph_get_parent(gb, முனை) != வெற்று) {
            தொடர்;
        }

        எனில் (முனை->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, அளவு(color), "yellow");
        } இல்லையெனில் எனில் (grad) {
            எனில் (ggml_graph_find(cgraph, முனை)) {
                snprintf(color, அளவு(color), "green");
            } இல்லையெனில் {
                snprintf(color, அளவு(color), "lightblue");
            }
        } இல்லையெனில் {
            snprintf(color, அளவு(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (வெறுமை *) முனை, color);

        எனில் (strlen(முனை->name) > 0) {
            fprintf(fp, "%s (%s)|", முனை->name, ggml_type_name(முனை->type));
        } இல்லையெனில் {
            fprintf(fp, "(%s)|", ggml_type_name(முனை->type));
        }

        எனில் (ggml_is_matrix(முனை)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, முனை->ne[0], முனை->ne[1], ggml_op_குறியீடு(முனை->op));
        } இல்லையெனில் {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, முனை->ne[0], முனை->ne[1], முனை->ne[2], ggml_op_குறியீடு(முனை->op));
        }

        எனில் (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_குறியீடு(grad->op));
        } இல்லையெனில் {
            fprintf(fp, "\"; ]\n");
        }
    }

    க்கு (முழு i = 0; i < gb->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * முனை = gb->leafs[i];

        snprintf(color, அளவு(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (வெறுமை *) முனை, color);

        எனில் (strlen(முனை->name) > 0) {
            fprintf(fp, "%s (%s)|", முனை->name, ggml_type_name(முனை->type));
        } இல்லையெனில் {
            fprintf(fp, "(%s)|", ggml_type_name(முனை->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, முனை->ne[0], முனை->ne[1]);
        எனில் (ggml_nelements(முனை) < 5 && முனை->data != வெற்று) {
            fprintf(fp, " | (");
            க்கு (முழு j = 0; j < ggml_nelements(முனை); j++) {
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
                எனில் (j < ggml_nelements(முனை) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    க்கு (முழு i = 0; i < gb->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = gb->nodes[i];

        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            எனில் (முனை->src[j]) {
                எழுத்து label[16];
                snprintf(label, அளவு(label), "src %d", j);
                ggml_graph_dump_dot_முனை_edge(fp, gb, முனை, முனை->src[j], label);
            }
        }
    }

    க்கு (முழு i = 0; i < gb->n_leafs; i++) {
        கட்டமைப்பு ggml_tensor * முனை = gb->leafs[i];

        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            எனில் (முனை->src[j]) {
                எழுத்து label[16];
                snprintf(label, அளவு(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, முனை, முனை->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

வெறுமை ggml_set_உள்ளீடு(கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_உள்ளீடு;
}

வெறுமை ggml_set_output(கட்டமைப்பு ggml_tensor * tensor) {
    க்கு (கட்டமைப்பு ggml_tensor * cur = tensor; cur != வெற்று; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

வெறுமை ggml_set_param(கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

வெறுமை ggml_set_loss(கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

வெறுமை ggml_quantize_init(எண்ணிக்கை ggml_type type) {
    ggml_critical_section_start();

    தேர்வு (type) {
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ2_S:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); நிறுத்து;
        நிலை GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); நிறுத்து;
        நிலை GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); நிறுத்து;
        இயல்பு: // nothing
            நிறுத்து;
    }

    ggml_critical_section_end();
}

வெறுமை ggml_quantize_free(வெறுமை) {
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

bool ggml_quantize_requires_imatrix(எண்ணிக்கை ggml_type type) {
    திரும்பு
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        எண்ணிக்கை ggml_type   type,
           மாறிலி தசம * src,
                  வெறுமை * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           மாறிலி தசம * imatrix) {
    மாறிலி int64_t n = nrows * n_per_row;

    எனில் (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != வெற்று);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    மாறிலி size_t start_row = start / n_per_row;
    மாறிலி size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    தேர்வு (type) {
        நிலை GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (எழுத்து *) dst + start_row * row_size, nrows, n_per_row, imatrix); நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                size_t elemsize = அளவு(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                size_t elemsize = அளவு(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                size_t elemsize = அளவு(தசம);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } நிறுத்து;
        இயல்பு:
            ப்ரதிஜ்ஞா(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    திரும்பு result;
}

////////////////////////////////////////////////////////////////////////////////

வெறுமை ggml_log_get(ggml_log_callback * log_callback, வெறுமை ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

வெறுமை ggml_log_set(ggml_log_callback log_callback, வெறுமை * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

வெறுமை ggml_threadpool_params_init(கட்டமைப்பு ggml_threadpool_params * p, முழு n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

கட்டமைப்பு ggml_threadpool_params ggml_threadpool_params_default(முழு n_threads) {
    கட்டமைப்பு ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    திரும்பு p;
}

bool ggml_threadpool_params_match(மாறிலி கட்டமைப்பு ggml_threadpool_params * p0, மாறிலி கட்டமைப்பு ggml_threadpool_params * p1) {
    எனில் (p0->n_threads  != p1->n_threads  ) திரும்பு false;
    எனில் (p0->prio       != p1->prio       ) திரும்பு false;
    எனில் (p0->poll       != p1->poll       ) திரும்பு false;
    எனில் (p0->strict_cpu != p1->strict_cpu ) திரும்பு false;
    திரும்பு memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
