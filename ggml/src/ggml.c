#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#ضمن "ggml-version.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-threading.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml.h"

// FIXME: required here for quantization functions
#ضمن "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#ضمن <hbwmalloc.h>
#endif

#إذا defined(_MSC_VER) || defined(__MINGW32__)
#ضمن <malloc.h> // using malloc.h with MSC/MINGW
#وإلاإذا !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ضمن <alloca.h>
#endif

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
#endif

#إذا defined(__APPLE__)
#ضمن <unistd.h>
#ضمن <mach/mach.h>
#ضمن <TargetConditionals.h>
#endif

#إذا defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#ضمن <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(فراغ) {
#ifdef _MSC_VER
#إذا defined(_WIN32)
    ساكن متطاير LONG counter = 1;
    أرجع (uint64_t) InterlockedIncrement(&counter) - 1;
#وإلا
    ساكن متطاير long long counter = 1;
    أرجع (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#وإلا
    ساكن uint64_t counter = 1;
    أرجع __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#إذا defined(__AVX512BF16__)
#إذا defined(_MSC_VER)
#define m512i(p) p
#وإلا
#ضمن <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#إذا defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#ضمن <unistd.h>
#ضمن <sys/types.h>
#ضمن <sys/stat.h>
#ضمن <sys/wait.h>
#إذا defined(__linux__)
#ضمن <sys/prctl.h>
#endif

#إذا defined(__ANDROID__)
#ضمن <unwind.h>
#ضمن <dlfcn.h>
#ضمن <stdio.h>

بنية backtrace_state {
    فراغ ** current;
    فراغ ** end;
};

ساكن _Unwind_Reason_Code unwind_callback(بنية _Unwind_Context* context, فراغ* arg) {
    بنية backtrace_state * state = (بنية backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    إذا (pc) {
        إذا (state->current == state->end) {
            أرجع URC_END_OF_مكدس;
        } وإلا {
            *state->current++ = (فراغ*)pc;
        }
    }
    أرجع _URC_NO_REASON;
}

ساكن فراغ ggml_اطبع_backtrace_symbols(فراغ) {
    ثابت صحيح max = 100;
    فراغ* buffer[max];

    بنية backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    صحيح عدد = state.current - buffer;

    لكل (صحيح idx = 0; idx < عدد; ++idx) {
        ثابت فراغ * addr = buffer[idx];
        ثابت حرف * رمز = "";

        Dl_info info;
        إذا (dladdr(addr, &info) && info.dli_sname) {
            رمز = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, رمز);
    }
}
#وإلاإذا defined(__linux__) && defined(__GLIBC__)
#ضمن <execinfo.h>
ساكن فراغ ggml_اطبع_backtrace_symbols(فراغ) {
    فراغ * trace[100];
    صحيح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وإلاإذا defined(__APPLE__)
#ضمن <execinfo.h>
ساكن فراغ ggml_اطبع_backtrace_symbols(فراغ) {
    فراغ * trace[100];
    صحيح nptrs = backtrace(trace, حجم(trace)/حجم(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وإلا
ساكن فراغ ggml_اطبع_backtrace_symbols(فراغ) {
    // platform not supported
}
#endif

فراغ ggml_اطبع_backtrace(فراغ) {
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
        ggml_اطبع_backtrace_symbols();
        أرجع;
    }
#endif
#إذا defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    حرف * سطر = عدم;
    ssize_t length = 0;
    طالما ((length = getline(&سطر, &size, f)) > 0) {
        إذا (!strncmp(سطر, "TracerPid:", حجم("TracerPid:") - 1) &&
            (length != حجم("TracerPid:\t0\n") - 1 || سطر[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            حرر(سطر);
            fclose(f);
            أرجع;
        }
    }
    حرر(سطر);
    fclose(f);
    صحيح lock[2] = { -1, -1 };
    (فراغ) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ثابت صحيح parent_pid = getpid();
    ثابت صحيح child_pid = fork();
    إذا (child_pid < 0) { // error
#إذا defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        أرجع;
    } وإلا إذا (child_pid == 0) { // child
        حرف attach[32];
        snprintf(attach, حجم(attach), "attach %d", parent_pid);
#إذا defined(__linux__)
        close(lock[1]);
        (فراغ) !اقرأ(lock[0], lock, 1);
        close(lock[0]);
#endif
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
        ggml_اطبع_backtrace_symbols();
        _Exit(0);
    } وإلا { // parent
#إذا defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, عدم, 0);
    }
}
#وإلا
فراغ ggml_اطبع_backtrace(فراغ) {
    // platform not supported
}
#endif

ساكن ggml_abort_callback_t g_abort_callback = عدم;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    أرجع ret_val;
}

فراغ ggml_abort(ثابت حرف * file, صحيح سطر, ثابت حرف * fmt, ...) {
    fflush(stdout);

    حرف message[2048];
    صحيح offset = snprintf(message, حجم(message), "%s:%d: ", file, سطر);

    va_list وسائط_2;
    va_start(وسائط_2, fmt);
    vsnprintf(message + offset, حجم(message) - offset, fmt, وسائط_2);
    va_end(وسائط_2);

    إذا (g_abort_callback) {
        g_abort_callback(message);
    } وإلا {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", message);
        ggml_اطبع_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

بنية ggml_logger_state {
    ggml_log_callback log_callback;
    فراغ * log_callback_user_data;
};
ساكن بنية ggml_logger_state g_logger_state = {ggml_log_callback_default, عدم};

ساكن فراغ ggml_log_internal_v(تعداد ggml_log_level level, ثابت حرف * format, va_list وسائط_2) {
    إذا (format == عدم) {
        أرجع;
    }
    va_list وسائط_copy;
    va_copy(وسائط_copy, وسائط_2);
    حرف buffer[128];
    صحيح len = vsnprintf(buffer, 128, format, وسائط_2);
    إذا (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } وإلا {
        حرف * buffer2 = (حرف *) calloc(len + 1, حجم(حرف));
        vsnprintf(buffer2, len + 1, format, وسائط_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        حرر(buffer2);
    }
    va_end(وسائط_copy);
}

فراغ ggml_log_internal(تعداد ggml_log_level level, ثابت حرف * format, ...) {
    va_list وسائط_2;
    va_start(وسائط_2, format);
    ggml_log_internal_v(level, format, وسائط_2);
    va_end(وسائط_2);
}

فراغ ggml_log_callback_default(تعداد ggml_log_level level, ثابت حرف * text, فراغ * user_data) {
    (فراغ) level;
    (فراغ) user_data;
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
#إذا defined(__has_feature)
#إذا __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#وإلا
#إذا defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

فراغ * ggml_aligned_malloc(size_t size) {
#إذا defined(__s390x__)
    ثابت صحيح alignment = 256;
#وإلا
    ثابت صحيح alignment = 64;
#endif

#إذا defined(_MSC_VER) || defined(__MINGW32__)
    أرجع _aligned_malloc(size, alignment);
#وإلا
    إذا (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        أرجع عدم;
    }
    فراغ * aligned_memory = عدم;
  #ifdef GGML_USE_CPU_HBM
    صحيح result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #وإلاإذا TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    صحيح result = EFAULT;
    بدل (alloc_status) {
        حالة KERN_SUCCESS:
            result = 0;
            اكسر;
        حالة KERN_INVALID_ADDRESS:
            result = EINVAL;
            اكسر;
        حالة KERN_NO_SPACE:
            result = ENOMEM;
            اكسر;
        افتراضي:
            result = EFAULT;
            اكسر;
    }
  #وإلا
    صحيح result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    إذا (result != 0) {
        // Handle allocation failure
        ثابت حرف *error_desc = "unknown allocation error";
        بدل (result) {
            حالة EINVAL:
                error_desc = "invalid alignment value";
                اكسر;
            حالة ENOMEM:
                error_desc = "insufficient memory";
                اكسر;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        أرجع عدم;
    }
    أرجع aligned_memory;
#endif
}

فراغ ggml_aligned_free(فراغ * ptr, size_t size) {
    GGML_UNUSED(size);
#إذا defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#وإلاإذا GGML_USE_CPU_HBM
    إذا (ptr != عدم) {
        hbw_free(ptr);
    }
#وإلاإذا TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    إذا (ptr != عدم) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#وإلا
    حرر(ptr);
#endif
}


مضمن ساكن فراغ * ggml_malloc(size_t size) {
    إذا (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        أرجع عدم;
    }
    فراغ * result = malloc(size);
    إذا (result == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    أرجع result;
}

// calloc
مضمن ساكن فراغ * ggml_calloc(size_t رقم, size_t size) {
    إذا (رقم == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        أرجع عدم;
    }
    فراغ * result = calloc(رقم, size);
    إذا (result == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    أرجع result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(رقم, size) ggml_calloc(رقم, size)

#define GGML_FREE(ptr) حرر(ptr)

ثابت حرف * ggml_status_to_string(تعداد ggml_status status) {
    بدل (status) {
        حالة GGML_STATUS_ALLOC_FAILED: أرجع "GGML status: error (failed to allocate memory)";
        حالة GGML_STATUS_FAILED:       أرجع "GGML status: error (operation failed)";
        حالة GGML_STATUS_SUCCESS:      أرجع "GGML status: success";
        حالة GGML_STATUS_ABORTED:      أرجع "GGML status: warning (operation aborted)";
    }

    أرجع "GGML status: unknown";
}

عائم ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    أرجع GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(عائم x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    أرجع GGML_FP32_TO_FP16(x);
}

عائم ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    أرجع GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(عائم x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    أرجع GGML_FP32_TO_BF16(x);
}

فراغ ggml_fp16_to_fp32_row(ثابت ggml_fp16_t * x, عائم * y, int64_t n) {
    لكل (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

فراغ ggml_fp32_to_fp16_row(ثابت عائم * x, ggml_fp16_t * y, int64_t n) {
    صحيح i = 0;
    لكل (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

فراغ ggml_bf16_to_fp32_row(ثابت ggml_bf16_t * x, عائم * y, int64_t n) {
    صحيح i = 0;
    لكل (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

فراغ ggml_fp32_to_bf16_row_ref(ثابت عائم * x, ggml_bf16_t * y, int64_t n) {
    لكل (صحيح i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

فراغ ggml_fp32_to_bf16_row(ثابت عائم * x, ggml_bf16_t * y, int64_t n) {
  صحيح i = 0;
#إذا defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  لكل (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    لكل (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    أرجع memcmp(guid_a, guid_b, حجم(ggml_guid)) == 0;
}

ثابت حرف * ggml_version(فراغ) {
    أرجع GGML_VERSION;
}

ثابت حرف * ggml_commit(فراغ) {
    أرجع GGML_COMMIT;
}

//
// timing
//

#إذا defined(_MSC_VER) || defined(__MINGW32__)
ساكن int64_t timer_freq, timer_start;
ساكن BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    أرجع TRUE;
}
فراغ ggml_time_init(فراغ) {
    ساكن INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, عدم, عدم);
}
int64_t ggml_time_ms(فراغ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    أرجع ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(فراغ) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    أرجع ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#وإلا
فراغ ggml_time_init(فراغ) {}
int64_t ggml_time_ms(فراغ) {
    بنية timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    أرجع (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(فراغ) {
    بنية timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    أرجع (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(فراغ) {
    أرجع clock();
}

int64_t ggml_cycles_per_ms(فراغ) {
    أرجع CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
ساكن wchar_t * ggml_mbstowcs(ثابت حرف * mbs) {
    صحيح wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, عدم, 0);
    إذا (!wlen) {
        errno = EINVAL;
        أرجع عدم;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * حجم(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    إذا (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        أرجع عدم;
    }

    أرجع wbuf;
}
#endif

FILE * ggml_fopen(ثابت حرف * fname, ثابت حرف * mode) {
#ifdef _WIN32
    FILE * file = عدم;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    إذا (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        إذا (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    أرجع file;
#وإلا
    أرجع fopen(fname, mode);
#endif

}

ساكن ثابت بنية ggml_type_traits type_traits[GGML_TYPE_عدد] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = حجم(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = حجم(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = حجم(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = حجم(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = حجم(مزدوج),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = حجم(عائم),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = حجم(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = حجم(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = حجم(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = حجم(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = حجم(block_q4_1),
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
        .type_size                = حجم(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = حجم(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = حجم(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = حجم(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = حجم(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = حجم(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = عدم,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = حجم(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = حجم(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = حجم(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = حجم(ggml_bf16_t),
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
        .type_size                = حجم(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = حجم(block_tq2_0),
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

ثابت بنية ggml_type_traits * ggml_get_type_traits(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع &type_traits[type];
}

//
// ggml object
//

بنية ggml_object {
    size_t offs;
    size_t size;

    بنية ggml_object * next;

    تعداد ggml_object_type type;

    حرف padding[4];
};

ساكن ثابت size_t GGML_OBJECT_SIZE = حجم(بنية ggml_object);

//
// ggml context
//

بنية ggml_context {
    size_t mem_size;
    فراغ * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    صحيح    n_objects;

    بنية ggml_object * objects_begin;
    بنية ggml_object * objects_end;
};

//
// data types
//

ساكن ثابت حرف * GGML_OP_NAME[GGML_OP_عدد] = {
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

static_assert(GGML_OP_عدد == 102, "GGML_OP_COUNT != 102");

ساكن ثابت حرف * GGML_OP_رمز[GGML_OP_عدد] = {
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

static_assert(GGML_OP_عدد == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_عدد == 2, "GGML_OP_POOL_COUNT != 2");

ساكن ثابت حرف * GGML_UNARY_OP_NAME[GGML_UNARY_OP_عدد] = {
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

static_assert(GGML_UNARY_OP_عدد == 22, "GGML_UNARY_OP_COUNT != 22");

ساكن ثابت حرف * GGML_GLU_OP_NAME[GGML_GLU_OP_عدد] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_عدد == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(حجم(بنية ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(حجم(بنية ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

فراغ ggml_اطبع_object(ثابت بنية ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ثابت فراغ *) obj->next);
}

فراغ ggml_اطبع_objects(ثابت بنية ggml_context * ctx) {
    بنية ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ثابت فراغ *) ctx);

    طالما (obj != عدم) {
        ggml_اطبع_object(obj);
        obj = obj->next;
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
    ثابت size_t blck_size = ggml_blck_size(tensor->type);
    إذا (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    وإلا {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    أرجع nbytes;
}

size_t ggml_nbytes_pad(ثابت بنية ggml_tensor * tensor) {
    أرجع GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع type_traits[type].blck_size;
}

size_t ggml_type_size(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع type_traits[type].type_size;
}

size_t ggml_row_size(تعداد ggml_type type, int64_t ne) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أكد(ne % ggml_blck_size(type) == 0);
    أرجع ggml_type_size(type)*ne/ggml_blck_size(type);
}

مزدوج ggml_type_sizef(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع ((مزدوج)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ثابت حرف * ggml_type_name(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع type_traits[type].type_name;
}

bool ggml_is_quantized(تعداد ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_عدد);
    أرجع type_traits[type].is_quantized;
}

ثابت حرف * ggml_op_name(تعداد ggml_op op) {
    أرجع GGML_OP_NAME[op];
}

ثابت حرف * ggml_op_رمز(تعداد ggml_op op) {
    أرجع GGML_OP_رمز[op];
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

size_t ggml_element_size(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ثابت بنية ggml_tensor * tensor) {
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

تعداد ggml_type ggml_ftype_to_ggml_type(تعداد ggml_ftype ftype) {
    تعداد ggml_type wtype = GGML_TYPE_عدد;

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
        حالة GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_عدد; اكسر;
        حالة GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_عدد; اكسر;
    }

    GGML_ASSERT(wtype != GGML_TYPE_عدد);

    أرجع wtype;
}

size_t ggml_tensor_overhead(فراغ) {
    أرجع GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->nb[0] > tensor->nb[1];
}

ساكن bool ggml_is_contiguous_m_n(ثابت بنية ggml_tensor * tensor, صحيح m, صحيح n) {
    size_t next_nb = ggml_type_size(tensor->type);
    إذا (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        أرجع false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    لكل (صحيح i = 1; i < n; i++) {
        إذا (i > m) {
            إذا (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                أرجع false;
            }
            next_nb *= tensor->ne[i];
        } وإلا {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
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

bool ggml_is_contiguous_to_1(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ثابت بنية ggml_tensor * tensor) {
    أرجع
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ثابت بنية ggml_tensor * tensor) {
    أرجع
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

ساكن مضمن bool ggml_is_padded_1d(ثابت بنية ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ثابت بنية ggml_tensor * tensor) {
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

    أرجع ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ساكن مضمن bool ggml_can_repeat_rows(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

بنية ggml_context * ggml_init(بنية ggml_init_params params) {
    ساكن bool is_first_call = true;

    ggml_critical_section_start();

    إذا (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    بنية ggml_context * ctx = GGML_MALLOC(حجم(بنية ggml_context));

    // allow to call ggml_init with 0 size
    إذا (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ثابت size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (بنية ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ عدم,
        /*.objects_end        =*/ عدم,
    };

    GGML_ASSERT(ctx->mem_buffer != عدم);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_اطبع_DEBUG("%s: context initialized\n", __func__);

    أرجع ctx;
}

فراغ ggml_reset(بنية ggml_context * ctx) {
    إذا (ctx == عدم) {
        أرجع;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = عدم;
    ctx->objects_end   = عدم;
}

فراغ ggml_free(بنية ggml_context * ctx) {
    إذا (ctx == عدم) {
        أرجع;
    }

    إذا (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ثابت بنية ggml_context * ctx) {
    أرجع ctx->objects_end == عدم ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(بنية ggml_context * ctx) {
    أرجع ctx->no_alloc;
}

فراغ ggml_set_no_alloc(بنية ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

فراغ * ggml_get_mem_buffer(ثابت بنية ggml_context * ctx) {
    أرجع ctx->mem_buffer;
}

size_t ggml_get_mem_size(ثابت بنية ggml_context * ctx) {
    أرجع ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ثابت بنية ggml_context * ctx) {
    size_t max_size = 0;

    لكل (بنية ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != عدم; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    أرجع max_size;
}

////////////////////////////////////////////////////////////////////////////////

ساكن بنية ggml_object * ggml_new_object(بنية ggml_context * ctx, تعداد ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    بنية ggml_object * obj_cur = ctx->objects_end;

    ثابت size_t cur_offs = obj_cur == عدم ? 0 : obj_cur->offs;
    ثابت size_t cur_size = obj_cur == عدم ? 0 : obj_cur->size;
    ثابت size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    حرف * ثابت mem_buffer = ctx->mem_buffer;
    بنية ggml_object * ثابت obj_new = (بنية ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    إذا (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        أرجع عدم;
    }
    إذا (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        أرجع عدم;
    }

    إذا (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        أرجع عدم;
    }

    *obj_new = (بنية ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = عدم,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    إذا (obj_cur != عدم) {
        obj_cur->next = obj_new;
    } وإلا {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

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

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_عدد);
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

    size_t data_size = ggml_row_size(type, ne[0]);
    لكل (صحيح i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == عدم || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    فراغ * data = view_src != عدم ? view_src->data : عدم;
    إذا (data != عدم) {
        data = (حرف *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    إذا (view_src == عدم && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    بنية ggml_object * ثابت obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    بنية ggml_tensor * ثابت result = (بنية ggml_tensor *)((حرف *)ctx->mem_buffer + obj_new->offs);

    *result = (بنية ggml_tensor) {
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
        /*.data         =*/ obj_alloc_size > 0 ? (فراغ *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ عدم,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    لكل (صحيح i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    أرجع result;
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

فراغ * ggml_new_buffer(بنية ggml_context * ctx, size_t nbytes) {
    بنية ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    أرجع (uint8_t *)ctx->mem_buffer + obj->offs;
}

بنية ggml_tensor * ggml_dup_tensor(بنية ggml_context * ctx, ثابت بنية ggml_tensor * src) {
    أرجع ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

فراغ ggml_unravel_index(ثابت بنية ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
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

فراغ * ggml_get_data(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->data;
}

عائم * ggml_get_data_f32(ثابت بنية ggml_tensor * tensor) {
    أكد(tensor->type == GGML_TYPE_F32);
    أرجع (عائم *)(tensor->data);
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

بنية ggml_tensor * ggml_set_name(بنية ggml_tensor * tensor, ثابت حرف * name) {
    size_t i;
    لكل (i = 0; i < حجم(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    أرجع tensor;
}

بنية ggml_tensor * ggml_format_name(بنية ggml_tensor * tensor, ثابت حرف * fmt, ...) {
    va_list وسائط_2;
    va_start(وسائط_2, fmt);
    vsnprintf(tensor->name, حجم(tensor->name), fmt, وسائط_2);
    va_end(وسائط_2);
    أرجع tensor;
}

بنية ggml_tensor * ggml_view_tensor(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * src) {
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    أرجع result;
}

بنية ggml_tensor * ggml_get_first_tensor(ثابت بنية ggml_context * ctx) {
    بنية ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_buffer = ctx->mem_buffer;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            أرجع (بنية ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_get_next_tensor(ثابت بنية ggml_context * ctx, بنية ggml_tensor * tensor) {
    بنية ggml_object * obj = (بنية ggml_object *) ((حرف *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    حرف * ثابت mem_buffer = ctx->mem_buffer;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            أرجع (بنية ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_get_tensor(بنية ggml_context * ctx, ثابت حرف * name) {
    بنية ggml_object * obj = ctx->objects_begin;

    حرف * ثابت mem_buffer = ctx->mem_buffer;

    طالما (obj != عدم) {
        إذا (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            بنية ggml_tensor * cur = (بنية ggml_tensor *)(mem_buffer + obj->offs);
            إذا (strcmp(cur->name, name) == 0) {
                أرجع cur;
            }
        }

        obj = obj->next;
    }

    أرجع عدم;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ساكن بنية ggml_tensor * ggml_dup_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    أرجع result;
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

ساكن بنية ggml_tensor * ggml_add_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

بنية ggml_tensor * ggml_add(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_add_impl(ctx, a, b, false);
}

بنية ggml_tensor * ggml_add_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

ساكن بنية ggml_tensor * ggml_add_cast_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    بنية ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

بنية ggml_tensor * ggml_add_cast(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        تعداد   ggml_type      type) {
    أرجع ggml_add_cast_impl(ctx, a, b, type);
}

بنية ggml_tensor * ggml_add_id(
            بنية ggml_context * ctx,
            بنية ggml_tensor  * a,
            بنية ggml_tensor  * b,
            بنية ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    أرجع result;
}

// ggml_add1

ساكن بنية ggml_tensor * ggml_add1_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

بنية ggml_tensor * ggml_acc(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    أرجع ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

بنية ggml_tensor * ggml_acc_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    أرجع ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

ساكن بنية ggml_tensor * ggml_sub_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    أرجع result;
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

بنية ggml_tensor * ggml_sum(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    أرجع result;
}

// ggml_sum_rows

بنية ggml_tensor * ggml_sum_rows(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    أرجع result;
}

// ggml_cumsum

بنية ggml_tensor * ggml_cumsum(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    أرجع result;
}

// ggml_mean

بنية ggml_tensor * ggml_mean(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    أرجع result;
}

// ggml_argmax

بنية ggml_tensor * ggml_argmax(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    أرجع result;
}

// ggml_count_equal

بنية ggml_tensor * ggml_عدد_equal(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_عدد_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_repeat

بنية ggml_tensor * ggml_repeat(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_repeat_4d(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ثابت bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    أرجع result;
}

// ggml_repeat_back

بنية ggml_tensor * ggml_repeat_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    أرجع result;
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

    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

بنية ggml_tensor * ggml_step(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

بنية ggml_tensor * ggml_step_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
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
        عائم                 negative_slope,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, حجم(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    أرجع result;
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

بنية ggml_tensor * ggml_gelu_quick(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

بنية ggml_tensor * ggml_gelu_quick_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
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
    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    أرجع result;
}

// ggml_silu_back

بنية ggml_tensor * ggml_silu_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, عدم, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

بنية ggml_tensor * ggml_round(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

بنية ggml_tensor * ggml_round_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
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

بنية ggml_tensor * ggml_glu_split(
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

بنية ggml_tensor * ggml_reglu_split(
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

بنية ggml_tensor * ggml_geglu_split(
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

بنية ggml_tensor * ggml_swiglu_split(
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

بنية ggml_tensor * ggml_geglu_erf_split(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

بنية ggml_tensor * ggml_geglu_quick(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_QUICK, false);
}

بنية ggml_tensor * ggml_geglu_quick_swapped(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    أرجع ggml_glu_impl(ctx, a, عدم, GGML_GLU_OP_GEGLU_QUICK, true);
}

بنية ggml_tensor * ggml_geglu_quick_split(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    أرجع ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

بنية ggml_tensor * ggml_swiglu_oai(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 alpha,
        عائم                 limit) {
    بنية ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    أرجع result;
}

بنية ggml_tensor * ggml_swiglu_clamp(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 limit) {
    بنية ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    أرجع result;
}

// ggml_norm

ساكن بنية ggml_tensor * ggml_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, حجم(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, حجم(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    أرجع result;
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

بنية ggml_tensor * ggml_rms_norm_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 eps) {
    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, حجم(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_group_norm

ساكن بنية ggml_tensor * ggml_group_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_group_norm(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps) {
    أرجع ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

بنية ggml_tensor * ggml_group_norm_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_groups,
        عائم                 eps) {
    أرجع ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ساكن بنية ggml_tensor * ggml_l2_norm_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 eps,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    أرجع result;
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

bool ggml_prec_set_acc(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    بدل (a->op) {
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            اكسر;
        افتراضي:
            أرجع false;
    };

    أرجع true;
}

bool ggml_prec_set_src(
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
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

فراغ ggml_mul_mat_set_prec(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

فراغ ggml_mul_mat_set_hint(
        بنية ggml_tensor * a,
        تعداد ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t hint_i32 = (int32_t) hint;

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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    أرجع result;
}

// ggml_out_prod

ساكن مضمن bool ggml_can_out_prod(ثابت بنية ggml_tensor * t0, ثابت بنية ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

بنية ggml_tensor * ggml_out_prod(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ثابت int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_scale

ساكن بنية ggml_tensor * ggml_scale_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 s,
        عائم                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[2] = { s, b };
    ggml_set_op_params(result, &params, حجم(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    أرجع result;
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

ساكن بنية ggml_tensor * ggml_set_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

بنية ggml_tensor * ggml_set(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

بنية ggml_tensor * ggml_set_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

بنية ggml_tensor * ggml_set_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

بنية ggml_tensor * ggml_set_1d_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

بنية ggml_tensor * ggml_set_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

بنية ggml_tensor * ggml_set_2d_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    أرجع ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

ساكن بنية ggml_tensor * ggml_cpy_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    بنية ggml_tensor * result = ggml_view_tensor(ctx, b);
    إذا (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } وإلا {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    أرجع result;
}

// ggml_cont

ساكن بنية ggml_tensor * ggml_cont_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    أرجع result;
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

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    أرجع result;
}

// ggml_reshape

بنية ggml_tensor * ggml_reshape(
        بنية ggml_context * ctx,
        بنية ggml_tensor * a,
        بنية ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_reshape_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ثابت int64_t ne[1] = { ne0 };
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_reshape_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ثابت int64_t ne[2] = { ne0, ne1 };
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    أرجع result;
}

ساكن بنية ggml_tensor * ggml_view_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_dims,
        ثابت int64_t       * ne,
        size_t                offset) {
    بنية ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, حجم(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    أرجع result;
}

// ggml_view_1d

بنية ggml_tensor * ggml_view_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    بنية ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    أرجع result;
}

// ggml_view_2d

بنية ggml_tensor * ggml_view_2d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ثابت int64_t ne[2] = { ne0, ne1 };

    بنية ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    أرجع result;
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
        size_t                offset) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };

    بنية ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    أرجع result;
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
        size_t                offset) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    بنية ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    أرجع result;
}

// ggml_permute

بنية ggml_tensor * ggml_permute(
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

    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, حجم(params));

    أرجع result;
}

// ggml_transpose

بنية ggml_tensor * ggml_transpose(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    أرجع result;
}

// ggml_get_rows

بنية ggml_tensor * ggml_get_rows(
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
    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_get_rows_back

بنية ggml_tensor * ggml_get_rows_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    بنية ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_set_rows

بنية ggml_tensor * ggml_set_rows(
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

    GGML_ASSERT(ggml_is_contiguous_rows(a));
    GGML_ASSERT(ggml_is_contiguous_rows(b));

    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    أرجع result;
}

// ggml_diag

بنية ggml_tensor * ggml_diag(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ثابت int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    أرجع result;
}

// ggml_diag_mask_inf

ساكن بنية ggml_tensor * ggml_diag_mask_inf_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_diag_mask_inf(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

بنية ggml_tensor * ggml_diag_mask_inf_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ساكن بنية ggml_tensor * ggml_diag_mask_zero_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_diag_mask_zero(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

بنية ggml_tensor * ggml_diag_mask_zero_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   n_past) {
    أرجع ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ساكن بنية ggml_tensor * ggml_clamp_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        عائم                 min,
        عائم                 max,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[] = { min, max };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    أرجع result;
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
        بنية ggml_tensor  * mask,
        عائم                 scale,
        عائم                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    إذا (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    إذا (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    عائم params[] = { scale, max_bias };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    أرجع result;
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
        بنية ggml_tensor  * mask,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

بنية ggml_tensor * ggml_soft_max_ext_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * mask,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

فراغ ggml_soft_max_add_sinks(
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

ساكن بنية ggml_tensor * ggml_soft_max_ext_back_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias,
        bool                  inplace) {
    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((عائم *) result->op_params + 0, &scale,    حجم(عائم));
    memcpy((عائم *) result->op_params + 1, &max_bias, حجم(عائم));

    أرجع result;
}

بنية ggml_tensor * ggml_soft_max_ext_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

بنية ggml_tensor * ggml_soft_max_ext_back_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        عائم                 scale,
        عائم                 max_bias) {
    أرجع ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ساكن بنية ggml_tensor * ggml_rope_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    إذا (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } وإلا {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    إذا (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    حجم(عائم));
    memcpy(params +  6, &freq_scale,   حجم(عائم));
    memcpy(params +  7, &ext_factor,   حجم(عائم));
    memcpy(params +  8, &attn_factor,  حجم(عائم));
    memcpy(params +  9, &beta_fast,    حجم(عائم));
    memcpy(params + 10, &beta_slow,    حجم(عائم));
    إذا (mrope_used && sections) {
        memcpy(params + 11, sections,  حجم(int32_t) * GGML_MROPE_SECTIONS);
    } وإلا {
        memset(params + 11, 0,         حجم(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    أرجع result;
}

بنية ggml_tensor * ggml_rope(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   mode) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

بنية ggml_tensor * ggml_rope_multi(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_multi_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[GGML_MROPE_SECTIONS],
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

بنية ggml_tensor * ggml_rope_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   mode) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

بنية ggml_tensor * ggml_rope_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, عدم, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_ext_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, c, n_dims, عدم, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

بنية ggml_tensor * ggml_rope_custom(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

بنية ggml_tensor * ggml_rope_custom_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   n_dims,
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    أرجع ggml_rope_impl(
        ctx, a, b, عدم, n_dims, عدم, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ساكن عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم base) {
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم)M_PI)) / (2 * logf(base));
}

فراغ ggml_rope_yarn_corr_dims(
    صحيح n_dims, صحيح n_ctx_orig, عائم freq_base, عائم beta_fast, عائم beta_slow, عائم dims[2]
) {
    // start and end correction dims
    عائم start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    عائم end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

بنية ggml_tensor * ggml_rope_ext_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    بنية ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    أرجع result;
}

بنية ggml_tensor * ggml_rope_multi_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c,
        صحيح                   n_dims,
        صحيح                   sections[4],
        صحيح                   mode,
        صحيح                   n_ctx_orig,
        عائم                 freq_base,
        عائم                 freq_scale,
        عائم                 ext_factor,
        عائم                 attn_factor,
        عائم                 beta_fast,
        عائم                 beta_slow) {
    بنية ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    أرجع result;
}

بنية ggml_tensor * ggml_rope_set_offset(
        بنية ggml_tensor  * a,
        صحيح                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ثابت int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    أرجع a;
}

ساكن int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, صحيح s, صحيح p, صحيح d) {
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

    ثابت int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ثابت int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ثابت int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    بنية ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

بنية ggml_tensor * ggml_im2col_back(
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    بنية ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    أرجع result;
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

    بنية ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    أرجع result;
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
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ثابت int64_t K_OC = a->ne[0];
    ثابت int64_t T_in = a->ne[1];
    ثابت int64_t K = K_OC / oc;
    ثابت int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ثابت int64_t ne[4] = { T_out, oc, 1, 1 };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    أرجع result;
}

// ggml_conv_transpose_1d

ساكن int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, صحيح s, صحيح p, صحيح d) {
    أرجع (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API بنية ggml_tensor * ggml_conv_transpose_1d(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   s0,
        صحيح                   p0,
        صحيح                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    بنية ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    أرجع result;
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
    UNUSED(OC);
    ثابت int64_t KD = a->ne[2];
    ثابت int64_t KH = a->ne[1];
    ثابت int64_t KW = a->ne[0];
    ثابت int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    ثابت int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    ثابت int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ثابت int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    بنية ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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
    بنية ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    أرجع result;
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
    بنية ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    أرجع result;
}

// ggml_conv_2d_dw_direct

بنية ggml_tensor * ggml_conv_2d_dw_direct(
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
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    بنية ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    إذا (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ثابت int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    أرجع result;
}

// ggml_conv_2d_direct

بنية ggml_tensor * ggml_conv_2d_direct(
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
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    بنية ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_conv_3d_direct

بنية ggml_tensor * ggml_conv_3d_direct(
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
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    أرجع result;
}

// ggml_conv_transpose_2d_p0

ساكن int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, صحيح s, صحيح p) {
    أرجع (ins - 1) * s - 2 * p + ks;
}

بنية ggml_tensor * ggml_conv_transpose_2d_p0(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        صحيح                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    بنية ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_pool_*

ساكن int64_t ggml_calc_pool_output_size(int64_t ins, صحيح ks, صحيح s, عائم p) {
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
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result;
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_pool_2d_back(
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
    بنية ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    أرجع result;
}

// ggml_upscale / ggml_interpolate

ساكن بنية ggml_tensor * ggml_interpolate_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_عدد);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    أرجع result;
}

بنية ggml_tensor * ggml_upscale(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   scale_factor,
        تعداد ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    أرجع ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

بنية ggml_tensor * ggml_upscale_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   ne0,
        صحيح                   ne1,
        صحيح                   ne2,
        صحيح                   ne3,
        تعداد ggml_scale_mode  mode) {
    أرجع ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

بنية ggml_tensor * ggml_interpolate(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    أرجع ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
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
    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    أرجع result;
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
    بنية ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    أرجع result;
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

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    أرجع result;
}

// ggml_roll

بنية ggml_tensor * ggml_roll(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   shift0,
        صحيح                   shift1,
        صحيح                   shift2,
        صحيح                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    أرجع result;
}

// ggml_timestep_embedding

بنية ggml_tensor * ggml_timestep_embedding(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * timesteps,
        صحيح                   dim,
        صحيح                   max_period) {

    بنية ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    أرجع result;
}

// ggml_tri

بنية ggml_tensor * ggml_tri(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    تعداد ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    بنية ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    أرجع result;
}

// ggml_fill

ساكن بنية ggml_tensor * ggml_fill_impl(
    بنية ggml_context * ctx,
    بنية ggml_tensor  * a,
    عائم                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    أرجع result;
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
        تعداد ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    أرجع result;
}

// ggml_argsort_top_k

بنية ggml_tensor * ggml_argsort_قمة_k(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    بنية ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    أرجع result;
}

// ggml_top_k

بنية ggml_tensor * ggml_قمة_k(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        صحيح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_قمة_K;
    result->src[0] = a;

    أرجع result;
}

// ggml_arange

بنية ggml_tensor * ggml_arange(
        بنية ggml_context * ctx,
        عائم                 start,
        عائم                 stop,
        عائم                 step) {
    GGML_ASSERT(stop > start);

    ثابت int64_t steps = (int64_t) ceilf((stop - start) / step);

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    أرجع result;
}

// ggml_flash_attn_ext

بنية ggml_tensor * ggml_flash_attn_ext(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * v,
        بنية ggml_tensor  * mask,
        عائم                 scale,
        عائم                 max_bias,
        عائم                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    إذا (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    إذا (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    عائم params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    أرجع result;
}


فراغ ggml_flash_attn_ext_set_prec(
        بنية ggml_tensor * a,
        تعداد ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

تعداد ggml_prec ggml_flash_attn_ext_get_prec(
        ثابت بنية ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    أرجع (تعداد ggml_prec) prec_i32;
}

فراغ ggml_flash_attn_ext_set_n_kv_max(
        بنية ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

فراغ ggml_flash_attn_ext_add_sinks(
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

بنية ggml_tensor * ggml_flash_attn_back(
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

    تعداد ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ثابت size_t tsize = ggml_type_size(result_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ثابت size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ثابت size_t nelements = (end + tsize - 1)/tsize;

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, حجم(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    أرجع result;
}

// ggml_ssm_conv

بنية ggml_tensor * ggml_ssm_conv(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * sx,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ثابت int64_t d_conv  = c->ne[0];
    ثابت int64_t d_inner = c->ne[1];
    ثابت int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ثابت int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    بنية ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    أرجع result;
}

// ggml_ssm_scan

بنية ggml_tensor * ggml_ssm_scan(
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
    GGML_ASSERT(x->nb[0] == ggml_type_size(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_size(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_size(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ثابت int64_t d_state      = s->ne[0];
        ثابت int64_t head_dim     = x->ne[0];
        ثابت int64_t n_head       = x->ne[1];
        ثابت int64_t n_seq_tokens = x->ne[2];
        ثابت int64_t n_seqs       = x->ne[3];

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

        إذا (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, حجم(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    أرجع result;
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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    أرجع result;
}

// ggml_add_rel_pos

ساكن بنية ggml_tensor * ggml_add_rel_pos_impl(
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    أرجع result;
}

بنية ggml_tensor * ggml_add_rel_pos(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * pw,
        بنية ggml_tensor  * ph) {
    أرجع ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

بنية ggml_tensor * ggml_add_rel_pos_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * pw,
        بنية ggml_tensor  * ph) {
    أرجع ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
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
    ثابت int64_t n_tokens = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    أرجع result;
}

// ggml_gated_linear_attn

بنية ggml_tensor * ggml_gated_linear_attn(
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
    ثابت int64_t n_tokens = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    أرجع result;
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
    ثابت int64_t n_tokens = k->ne[2];
    ثابت int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ثابت int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    أرجع result;
}

// ggml_unary

ساكن بنية ggml_tensor * ggml_unary_impl(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        تعداد ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    أرجع result;
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, حجم(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    أرجع result;
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, حجم(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    بنية ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    بنية ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, حجم(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    أرجع result;
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
        بنية ggml_tensor ** وسائط_2,
        صحيح                   n_وسائط,
        ggml_custom_op_t      fun,
        صحيح                   n_tasks,
        فراغ                * userdata) {

    GGML_ASSERT(n_وسائط < GGML_MAX_SRC);

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    بنية ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, حجم(params));

    result->op = GGML_OP_CUSTOM;
    لكل (صحيح i = 0; i < n_وسائط; i++) {
        result->src[i] = وسائط_2[i];
    }

    أرجع result;
}

بنية ggml_tensor * ggml_custom_inplace(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor ** وسائط_2,
        صحيح                   n_وسائط,
        ggml_custom_op_t      fun,
        صحيح                   n_tasks,
        فراغ                * userdata) {

    GGML_ASSERT(n_وسائط < GGML_MAX_SRC - 1);

    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);

    بنية ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, حجم(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    لكل (صحيح i = 0; i < n_وسائط; i++) {
        result->src[i + 1] = وسائط_2[i];
    }

    أرجع result;
}
// ggml_cross_entropy_loss

بنية ggml_tensor * ggml_cross_entropy_loss(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
}

// ggml_cross_entropy_loss_back

بنية ggml_tensor * ggml_cross_entropy_loss_back(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        بنية ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    بنية ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    أرجع result;
}

// opt_step_adamw

بنية ggml_tensor * ggml_opt_step_adamw(
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

    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    أرجع result;
}

// opt_step_sgd

بنية ggml_tensor * ggml_opt_step_sgd(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * grad,
        بنية ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    بنية ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    أرجع result;
}

// solve_tri

بنية ggml_tensor * ggml_solve_tri(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * a,
        بنية ggml_tensor  * b,
        bool                  يسار,
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

    GGML_ASSERT(lower && يسار && !uni); // TODO: support other variants

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    أرجع result;
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

    ثابت int64_t S_v      = v->ne[0];
    ثابت int64_t H        = v->ne[1];
    ثابت int64_t n_tokens = v->ne[2];
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
    ثابت int64_t state_rows = K * S_v * n_seqs;
    ثابت int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    أرجع result;
}

// ggml_lightning_indexer

بنية ggml_tensor * ggml_lightning_indexer(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * q,
        بنية ggml_tensor  * k,
        بنية ggml_tensor  * weights,
        بنية ggml_tensor  * mask) {

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
    بنية ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    أرجع result;
}

// ggml_dsv4_hc_comb

بنية ggml_tensor * ggml_dsv4_hc_comb(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * mixes,
        بنية ggml_tensor  * scale,
        بنية ggml_tensor  * base,
        عائم                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ثابت int64_t hc_mix_dim = mixes->ne[0];
    ثابت int64_t n_tokens   = mixes->ne[1];

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
    GGML_ASSERT(base->ne[0] == hc_mix_dim);
    GGML_ASSERT(base->ne[1] == 1);
    GGML_ASSERT(base->ne[2] == 1);
    GGML_ASSERT(base->ne[3] == 1);

    بنية ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    أرجع result;
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
    ثابت int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    إذا (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } وإلا {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    بنية ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    أرجع result;
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
    ثابت int64_t n_tokens = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

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

    إذا (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    بنية ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    أرجع result;
}

// ggml_top_k_mask

بنية ggml_tensor * ggml_قمة_k_mask(
        بنية ggml_context * ctx,
        بنية ggml_tensor  * mask,
        بنية ggml_tensor  * قمة_k,
        بنية ggml_tensor  * score,
        بنية ggml_tensor  * blocks,
        بنية ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(قمة_k->type == GGML_TYPE_I32);
    GGML_ASSERT(قمة_k->ne[1] % mask->ne[1] == 0 && قمة_k->ne[2] % mask->ne[2] == 0 && قمة_k->ne[3] % mask->ne[3] == 0);

    إذا (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == قمة_k->ne[1] && score->ne[2] == قمة_k->ne[2] && score->ne[3] == قمة_k->ne[3]);
    }

    إذا (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    إذا (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == قمة_k->ne[1] && extra->ne[2] == قمة_k->ne[2] && extra->ne[3] == قمة_k->ne[3]);
    }

    بنية ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], قمة_k->ne[1], قمة_k->ne[2], قمة_k->ne[3]);

    result->op     = GGML_OP_قمة_K_MASK;
    result->src[0] = mask;
    result->src[1] = قمة_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    أرجع result;
}

////////////////////////////////////////////////////////////////////////////////

بنية ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    بنية ggml_hash_set result;
    result.size = size;
    result.مفاتيح = GGML_MALLOC(حجم(بنية ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), حجم(ggml_bitset_t));
    أرجع result;
}

فراغ ggml_hash_set_reset(بنية ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, حجم(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

فراغ ggml_hash_set_free(بنية ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->مفاتيح);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    ساكن ثابت size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ساكن ثابت size_t n_primes = حجم(primes)/حجم(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    طالما (l < r) {
        size_t m = (l + r)/2;
        إذا (primes[m] < min_sz) {
            l = m + 1;
        } وإلا {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    أرجع sz;
}

بنية hash_map {
    بنية ggml_hash_set set;
    بنية ggml_tensor ** vals;
};

ساكن بنية hash_map * ggml_new_hash_map(size_t size) {
    بنية hash_map * result = GGML_MALLOC(حجم(بنية hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, حجم(بنية ggml_tensor *));
    أرجع result;
}

ساكن فراغ ggml_hash_map_free(بنية hash_map * map) {
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

ساكن فراغ ggml_add_or_set(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->visited_hash_set.مفاتيح[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } وإلا {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_acc_or_set(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor,
        ثابت  size_t         nb1,
        ثابت  size_t         nb2,
        ثابت  size_t         nb3,
        ثابت  size_t         offset) {
    بنية ggml_tensor * src = cgraph->visited_hash_set.مفاتيح[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } وإلا {
        بنية ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.مفاتيح[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_add1_or_set(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->visited_hash_set.مفاتيح[isrc];
    GGML_ASSERT(src);
    إذا (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } وإلا {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ساكن فراغ ggml_sub_or_set(
        بنية ggml_context * ctx,
        بنية ggml_cgraph  * cgraph,
        size_t                isrc,
        بنية ggml_tensor  * tensor) {
    بنية ggml_tensor * src = cgraph->visited_hash_set.مفاتيح[isrc];
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
    بنية ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    إذا (!grad) {
        أرجع;
    }

    بنية ggml_tensor * src0 = tensor->src[0];
    بنية ggml_tensor * src1 = tensor->src[1];
    بنية ggml_tensor * src2 = tensor->src[2];
    بنية ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ثابت size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ثابت size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ثابت size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ثابت bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ثابت bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ثابت bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    بدل (tensor->op) {
        حالة GGML_OP_DUP: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } اكسر;
        حالة GGML_OP_ADD: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                بنية ggml_tensor * tmp = grad;
                إذا (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } اكسر;
        حالة GGML_OP_ADD1: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } اكسر;
        حالة GGML_OP_ACC: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ثابت size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ثابت size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ثابت size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ثابت size_t offset = ((int32_t *) tensor->op_params)[3];

                بنية ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } اكسر;
        حالة GGML_OP_SUB: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            إذا (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } اكسر;
        حالة GGML_OP_MUL: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            إذا (src1_needs_grads) {
                بنية ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                إذا (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } اكسر;
        حالة GGML_OP_DIV: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            إذا (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } اكسر;
        حالة GGML_OP_SQR: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } اكسر;
        حالة GGML_OP_SQRT: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } اكسر;
        حالة GGML_OP_LOG: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_SIN: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } اكسر;
        حالة GGML_OP_COS: {
            إذا (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } اكسر;
        حالة GGML_OP_SUM: {
            إذا (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } اكسر;
        حالة GGML_OP_SUM_ROWS: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_MEAN: {
            إذا (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } اكسر;
        حالة GGML_OP_REPEAT: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_REPEAT_BACK: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_RMS_NORM: {
            إذا (src0_needs_grads) {
                عائم eps;
                memcpy(&eps, tensor->op_params, حجم(عائم));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
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
                    ggml_out_prod(ctx, // [n,m,qq,rr]
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
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            إذا (src1_needs_grads) {
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
        } اكسر;
        حالة GGML_OP_SCALE: {
            إذا (src0_needs_grads) {
                عائم s;
                memcpy(&s, tensor->op_params, حجم(عائم));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } اكسر;
        حالة GGML_OP_SET: {
            ثابت size_t nb1    = ((ثابت int32_t *) tensor->op_params)[0];
            ثابت size_t nb2    = ((ثابت int32_t *) tensor->op_params)[1];
            ثابت size_t nb3    = ((ثابت int32_t *) tensor->op_params)[2];
            ثابت size_t offset = ((ثابت int32_t *) tensor->op_params)[3];

            بنية ggml_tensor * tensor_grad_view = عدم;

            إذا (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            إذا (src0_needs_grads) {
                بنية ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            إذا (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } اكسر;
        حالة GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            إذا (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
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
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } اكسر;
        حالة GGML_OP_RESHAPE: {
            إذا (src0_needs_grads) {
                بنية ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } اكسر;
        حالة GGML_OP_VIEW: {
            إذا (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, حجم(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                إذا (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } اكسر;
        حالة GGML_OP_PERMUTE: {
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
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } اكسر;
        حالة GGML_OP_TRANSPOSE: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } اكسر;
        حالة GGML_OP_GET_ROWS: {
            إذا (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            إذا (src1_needs_grads) {
                // noop
            }
        } اكسر;
        حالة GGML_OP_DIAG_MASK_INF: {
            إذا (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ثابت صحيح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } اكسر;
        حالة GGML_OP_DIAG_MASK_ZERO: {
            إذا (src0_needs_grads) {
                ثابت صحيح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } اكسر;
        حالة GGML_OP_SOFT_MAX: {
            إذا (src0_needs_grads) {
                عائم scale    = 1.0f;
                عائم max_bias = 0.0f;

                memcpy(&scale,    (ثابت عائم *) tensor->op_params + 0, حجم(عائم));
                memcpy(&max_bias, (ثابت عائم *) tensor->op_params + 1, حجم(عائم));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } اكسر;
        حالة GGML_OP_ROPE: {
            إذا (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ثابت صحيح n_dims     = ((ثابت int32_t *) tensor->op_params)[1];
                ثابت صحيح mode       = ((ثابت int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ثابت صحيح n_ctx_orig = ((ثابت int32_t *) tensor->op_params)[4];
                عائم freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                صحيح sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ثابت عائم *) tensor->op_params +  5, حجم(عائم));
                memcpy(&freq_scale,  (ثابت عائم *) tensor->op_params +  6, حجم(عائم));
                memcpy(&ext_factor,  (ثابت عائم *) tensor->op_params +  7, حجم(عائم));
                memcpy(&attn_factor, (ثابت عائم *) tensor->op_params +  8, حجم(عائم));
                memcpy(&beta_fast,   (ثابت عائم *) tensor->op_params +  9, حجم(عائم));
                memcpy(&beta_slow,   (ثابت عائم *) tensor->op_params + 10, حجم(عائم));
                memcpy(&sections,                    tensor->op_params + 11, حجم(sections));

                بنية ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } اكسر;
        حالة GGML_OP_IM2COL: {
            إذا (src1_needs_grads) {
                ثابت int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ثابت int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ثابت int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ثابت int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ثابت int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ثابت int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ثابت bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
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

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } اكسر;
        حالة GGML_OP_WIN_PART:
        حالة GGML_OP_WIN_UNPART:
        حالة GGML_OP_UNARY: {
            بدل (ggml_get_unary_op(tensor)) {
                حالة GGML_UNARY_OP_ABS: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SGN: {
                    // noop
                } اكسر;
                حالة GGML_UNARY_OP_NEG: {
                    إذا (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } اكسر;
                حالة GGML_UNARY_OP_STEP: {
                    // noop
                } اكسر;
                حالة GGML_UNARY_OP_RELU: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SILU: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_EXP: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_EXPM1: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } اكسر;
                حالة GGML_UNARY_OP_SOFTPLUS: {
                    إذا (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
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
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } اكسر;
        حالة GGML_OP_GLU: {
            بدل (ggml_get_glu_op(tensor)) {
                حالة GGML_GLU_OP_SWIGLU: {
                    إذا (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    إذا (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
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
        حالة GGML_OP_عدد:
        افتراضي: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ساكن size_t ggml_visit_parents_graph(بنية ggml_cgraph * cgraph, بنية ggml_tensor * عقدة, bool compute) {
    إذا (عقدة->op != GGML_OP_NONE && compute) {
        عقدة->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ثابت size_t عقدة_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, عقدة);
    GGML_ASSERT(عقدة_hash_pos != GGML_HASHSET_FULL);

    إذا (ggml_bitset_get(cgraph->visited_hash_set.used, عقدة_hash_pos)) {
        // already visited

        إذا (compute) {
            // update the compute flag regardless
            لكل (صحيح i = 0; i < GGML_MAX_SRC; ++i) {
                بنية ggml_tensor * src = عقدة->src[i];
                إذا (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        أرجع عقدة_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.مفاتيح[عقدة_hash_pos] = عقدة;
    ggml_bitset_set(cgraph->visited_hash_set.used, عقدة_hash_pos);
    cgraph->use_counts[عقدة_hash_pos] = 0;

    لكل (صحيح i = 0; i < GGML_MAX_SRC; ++i) {
        ثابت صحيح k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_يسار_TO_يمين) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_يمين_TO_يسار) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        بنية ggml_tensor * src = عقدة->src[k];
        إذا (src) {
            ثابت size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    إذا (عقدة->op == GGML_OP_NONE && !(عقدة->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        إذا (strlen(عقدة->name) == 0) {
            ggml_format_name(عقدة, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = عقدة;
        cgraph->n_leafs++;
    } وإلا {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        إذا (strlen(عقدة->name) == 0) {
            ggml_format_name(عقدة, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = عقدة;
        cgraph->n_nodes++;
    }

    أرجع عقدة_hash_pos;
}

ساكن فراغ ggml_build_forward_impl(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor, bool expand, bool compute) {
    إذا (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ثابت صحيح n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ثابت صحيح n_new = cgraph->n_nodes - n_old;
    GGML_اطبع_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

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

فراغ ggml_build_forward_order(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor) {
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

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*حجم(بنية ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*حجم(بنية ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, حجم(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        لكل (صحيح i = 0; i < n_nodes_f; ++i) {
            بنية ggml_tensor * عقدة = cgraph->nodes[i];
            any_params = any_params || (عقدة->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (عقدة->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    لكل (صحيح i = 0; i < n_nodes_f; ++i) {
        بنية ggml_tensor * عقدة = cgraph->nodes[i];

        إذا (عقدة->type == GGML_TYPE_I32) {
            استمر;
        }

        bool عقدة_needs_grad = (عقدة->flags & GGML_TENSOR_FLAG_PARAM) || (عقدة->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        بدل (عقدة->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            حالة GGML_OP_IM2COL:      // only used for its shape
            حالة GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                اكسر;
            حالة GGML_OP_UNARY: {
                ثابت تعداد ggml_unary_op uop = ggml_get_unary_op(عقدة);
                // SGN and STEP unary ops are piecewise constant
                إذا (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } اكسر;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            حالة GGML_OP_CPY:           // gradients in CPY target are irrelevant
            حالة GGML_OP_GET_ROWS:      // row indices not differentiable
            حالة GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            حالة GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                اكسر;

            افتراضي:
                اكسر;
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            إذا (!عقدة->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, عقدة->src[j])]) {
                استمر;
            }
            GGML_ASSERT(عقدة->src[j]->type == GGML_TYPE_F32 || عقدة->src[j]->type == GGML_TYPE_F16);
            عقدة_needs_grad = true;
            اكسر;
        }
        إذا (!عقدة_needs_grad) {
            استمر;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!عقدة->view_src || عقدة->op == GGML_OP_CPY || عقدة->op == GGML_OP_SET_ROWS || عقدة->op == GGML_OP_VIEW ||
            عقدة->op == GGML_OP_RESHAPE || عقدة->op == GGML_OP_PERMUTE || عقدة->op == GGML_OP_TRANSPOSE);

        ثابت size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, عقدة);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        إذا (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } وإلا إذا (عقدة->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, عقدة->ne);
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

ساكن فراغ * incr_ptr_aligned(فراغ ** p, size_t size, size_t align) {
    فراغ * ptr = *p;
    ptr = (فراغ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (فراغ *) ((حرف *) ptr + size);
    أرجع ptr;
}

ساكن size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    فراغ * p = (حرف *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, حجم(بنية ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * حجم(int32_t), حجم(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // hash keys
    إذا (grads) {
        incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    أرجع nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    أرجع GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(فراغ) {
    أرجع ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

بنية ggml_cgraph * ggml_new_graph_custom(بنية ggml_context * ctx, size_t size, bool grads) {
    ثابت size_t obj_size = ggml_graph_nbytes(size, grads);
    بنية ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    بنية ggml_cgraph * cgraph = (بنية ggml_cgraph *) ((حرف *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    فراغ * p = cgraph + 1;

    بنية ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    بنية ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * حجم(int32_t), حجم(int32_t));
    بنية ggml_tensor ** hash_مفاتيح_ptr  =         incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *));
    بنية ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)) : عدم;
    بنية ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * حجم(بنية ggml_tensor *), حجم(بنية ggml_tensor *)) : عدم;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * حجم(ggml_bitset_t), حجم(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    أكد(obj_size == (size_t)((حرف *)p - (حرف *)cgraph));

    *cgraph = (بنية ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_مفاتيح_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_يسار_TO_يمين,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    إذا (grads) {
        memset(cgraph->grads,     0, hash_size*حجم(بنية ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*حجم(بنية ggml_tensor *));
    }

    أرجع cgraph;
}

بنية ggml_cgraph * ggml_new_graph(بنية ggml_context * ctx) {
    أرجع ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

بنية ggml_cgraph ggml_graph_view(بنية ggml_cgraph * cgraph0, صحيح i0, صحيح i1) {
    بنية ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ عدم, // gradients would need visited_hash_set
        /*.grad_accs        =*/ عدم,
        /*.leafs            =*/ عدم,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    أرجع cgraph;
}

فراغ ggml_graph_cpy(بنية ggml_cgraph * src, بنية ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    لكل (صحيح i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    لكل (صحيح i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    لكل (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        إذا (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_أدرج(&dst->visited_hash_set, src->visited_hash_set.مفاتيح[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    إذا (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*حجم(بنية ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*حجم(بنية ggml_tensor *));
    }
    إذا (src->grads) {
        GGML_ASSERT(dst->grads     != عدم);
        GGML_ASSERT(dst->grad_accs != عدم);
        لكل (صحيح i = 0; i < src->n_nodes; ++i) {
            ثابت size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ثابت size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

بنية ggml_cgraph * ggml_graph_dup(بنية ggml_context * ctx, بنية ggml_cgraph * cgraph, bool force_grads) {
    بنية ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    أرجع result;
}

بنية ggml_tensor * ggml_set_zero(بنية ggml_tensor * tensor) {
    إذا (ggml_is_empty(tensor)) {
        أرجع tensor;
    }
    إذا (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } وإلا {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    أرجع tensor;
}

فراغ ggml_graph_reset(بنية ggml_cgraph * cgraph) {
    إذا (!cgraph) {
        أرجع;
    }
    GGML_ASSERT(cgraph->grads != عدم);

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * عقدة     = cgraph->nodes[i];
        بنية ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, عقدة);

        إذا (عقدة->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(عقدة->src[2]);
            ggml_set_zero(عقدة->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        إذا (grad_acc) {
            إذا (عقدة->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ثابت عائم onef = 1.0f;
                إذا (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, حجم(عائم));
                } وإلا {
                    GGML_ASSERT(grad_acc->data);
                    *((عائم *) grad_acc->data) = onef;
                }
            } وإلا {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

فراغ ggml_graph_clear(بنية ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

صحيح ggml_graph_size(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->size;
}

بنية ggml_tensor * ggml_graph_عقدة(بنية ggml_cgraph * cgraph, صحيح i) {
    إذا (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        أرجع cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    أرجع cgraph->nodes[i];
}

بنية ggml_tensor ** ggml_graph_nodes(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->nodes;
}

صحيح ggml_graph_n_nodes(بنية ggml_cgraph * cgraph) {
    أرجع cgraph->n_nodes;
}

فراغ ggml_graph_add_عقدة(بنية ggml_cgraph * cgraph, بنية ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

بنية ggml_tensor * ggml_graph_get_tensor(ثابت بنية ggml_cgraph * cgraph, ثابت حرف * name) {
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        بنية ggml_tensor * leaf = cgraph->leafs[i];

        إذا (strcmp(leaf->name, name) == 0) {
            أرجع leaf;
        }
    }

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = cgraph->nodes[i];

        إذا (strcmp(عقدة->name, name) == 0) {
            أرجع عقدة;
        }
    }

    أرجع عدم;
}

بنية ggml_tensor * ggml_graph_get_grad(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * عقدة) {
    ثابت size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, عقدة);
    أرجع igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : عدم;
}

بنية ggml_tensor * ggml_graph_get_grad_acc(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * عقدة) {
    ثابت size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, عقدة);
    أرجع igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : عدم;
}

فراغ ggml_graph_اطبع(ثابت بنية ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * عقدة = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                عقدة->ne[0], عقدة->ne[1], عقدة->ne[2],
                ggml_op_name(عقدة->op), (عقدة->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, عقدة) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        بنية ggml_tensor * عقدة = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                عقدة->ne[0], عقدة->ne[1],
                ggml_op_name(عقدة->op),
                ggml_get_name(عقدة));
    }

    GGML_LOG_INFO("========================================\n");
}

ساكن صحيح ggml_عقدة_list_find_tensor(ثابت بنية ggml_cgraph * cgraph,
                                      ثابت صحيح *                idxs,
                                      صحيح                        عدد,
                                      ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    لكل (صحيح i = 0; i < عدد; ++i) {
        ثابت صحيح عقدة_idx = idxs[i];

        إذا (عقدة_idx >= cgraph->n_nodes) {
            أرجع -1;
        }
        إذا (cgraph->nodes[عقدة_idx] == tensor) {
            أرجع i;
        }
    }
    أرجع -1;
}

ساكن bool ggml_is_constant(ثابت بنية ggml_tensor * tensor) {
    أرجع tensor->buffer != عدم && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ثابت بنية ggml_cgraph * cgraph,
                                ثابت صحيح *                عقدة_idxs,
                                صحيح                        عدد,
                                ثابت تعداد ggml_op *       ops,
                                ثابت صحيح *                outputs,
                                صحيح                        رقم_outputs) {
    GGML_ASSERT(outputs && رقم_outputs > 0);

    لكل (صحيح i = 0; i < عدد; ++i) {
        إذا (عقدة_idxs[i] >= cgraph->n_nodes) {
            أرجع false;
        }

        ثابت بنية ggml_tensor * عقدة = cgraph->nodes[عقدة_idxs[i]];

        إذا (عقدة->op != ops[i]) {
            أرجع false;
        }

        إذا ((عقدة->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            أرجع false;
        }

        إذا (ggml_عقدة_list_find_tensor(cgraph, outputs, رقم_outputs, عقدة) != -1) {
            استمر;
        }

        إذا (عقدة->flags & GGML_TENSOR_FLAG_OUTPUT) {
            أرجع false;
        }

        صحيح subgraph_uses = 0;
        لكل (صحيح j = i + 1; j < عدد; ++j) {
            ثابت بنية ggml_tensor * other_عقدة = cgraph->nodes[عقدة_idxs[j]];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                إذا (other_عقدة->src[src_idx] == عقدة) {
                    subgraph_uses++;
                }
            }
        }

        إذا (subgraph_uses != ggml_عقدة_get_use_عدد(cgraph, عقدة_idxs[i])) {
            أرجع false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        بنية ggml_tensor * view_src = عقدة->view_src;
        طالما (view_src) {
            إذا (ggml_عقدة_list_find_tensor(cgraph, عقدة_idxs, عدد, view_src) == -1 && !ggml_is_constant(view_src)) {
                أرجع false;
            }
            view_src = view_src->view_src;
        }
    }

    أرجع true;
}

// check if node is part of the graph
ساكن bool ggml_graph_find(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * عقدة) {
    إذا (cgraph == عدم) {
        أرجع true;
    }

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i] == عقدة) {
            أرجع true;
        }
    }

    أرجع false;
}

ساكن بنية ggml_tensor * ggml_graph_get_parent(ثابت بنية ggml_cgraph * cgraph, ثابت بنية ggml_tensor * عقدة) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * parent = cgraph->nodes[i];
        بنية ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        إذا (grad == عقدة) {
            أرجع parent;
        }
    }

    أرجع عدم;
}

ساكن فراغ ggml_graph_dump_dot_عقدة_edge(FILE * fp, ثابت بنية ggml_cgraph * gb, بنية ggml_tensor * عقدة, بنية ggml_tensor * parent, ثابت حرف * label)  {
    بنية ggml_tensor * gparent = ggml_graph_get_parent(gb, عقدة);
    بنية ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (فراغ *) gparent0 : (فراغ *) parent,
            gparent ? (فراغ *) gparent : (فراغ *) عقدة,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ساكن فراغ ggml_graph_dump_dot_leaf_edge(FILE * fp, بنية ggml_tensor * عقدة, بنية ggml_tensor * parent, ثابت حرف * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (فراغ *) parent,
            (فراغ *) عقدة,
            label);
}

فراغ ggml_graph_dump_dot(ثابت بنية ggml_cgraph * gb, ثابت بنية ggml_cgraph * cgraph, ثابت حرف * filename) {
    حرف color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    لكل (صحيح i = 0; i < gb->n_nodes; i++) {
        بنية ggml_tensor * عقدة = gb->nodes[i];
        بنية ggml_tensor * grad = ggml_graph_get_grad(gb, عقدة);

        إذا (ggml_graph_get_parent(gb, عقدة) != عدم) {
            استمر;
        }

        إذا (عقدة->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, حجم(color), "yellow");
        } وإلا إذا (grad) {
            إذا (ggml_graph_find(cgraph, عقدة)) {
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
                (فراغ *) عقدة, color);

        إذا (strlen(عقدة->name) > 0) {
            fprintf(fp, "%s (%s)|", عقدة->name, ggml_type_name(عقدة->type));
        } وإلا {
            fprintf(fp, "(%s)|", ggml_type_name(عقدة->type));
        }

        إذا (ggml_is_matrix(عقدة)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, عقدة->ne[0], عقدة->ne[1], ggml_op_رمز(عقدة->op));
        } وإلا {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, عقدة->ne[0], عقدة->ne[1], عقدة->ne[2], ggml_op_رمز(عقدة->op));
        }

        إذا (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_رمز(grad->op));
        } وإلا {
            fprintf(fp, "\"; ]\n");
        }
    }

    لكل (صحيح i = 0; i < gb->n_leafs; i++) {
        بنية ggml_tensor * عقدة = gb->leafs[i];

        snprintf(color, حجم(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (فراغ *) عقدة, color);

        إذا (strlen(عقدة->name) > 0) {
            fprintf(fp, "%s (%s)|", عقدة->name, ggml_type_name(عقدة->type));
        } وإلا {
            fprintf(fp, "(%s)|", ggml_type_name(عقدة->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, عقدة->ne[0], عقدة->ne[1]);
        إذا (ggml_nelements(عقدة) < 5 && عقدة->data != عدم) {
            fprintf(fp, " | (");
            لكل (صحيح j = 0; j < ggml_nelements(عقدة); j++) {
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
                إذا (j < ggml_nelements(عقدة) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    لكل (صحيح i = 0; i < gb->n_nodes; i++) {
        بنية ggml_tensor * عقدة = gb->nodes[i];

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (عقدة->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_graph_dump_dot_عقدة_edge(fp, gb, عقدة, عقدة->src[j], label);
            }
        }
    }

    لكل (صحيح i = 0; i < gb->n_leafs; i++) {
        بنية ggml_tensor * عقدة = gb->leafs[i];

        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (عقدة->src[j]) {
                حرف label[16];
                snprintf(label, حجم(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, عقدة, عقدة->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_set_مدخل(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_مدخل;
}

فراغ ggml_set_output(بنية ggml_tensor * tensor) {
    لكل (بنية ggml_tensor * cur = tensor; cur != عدم; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

فراغ ggml_set_param(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

فراغ ggml_set_loss(بنية ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_quantize_init(تعداد ggml_type type) {
    ggml_critical_section_start();

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

    ggml_critical_section_end();
}

فراغ ggml_quantize_free(فراغ) {
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
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ثابت عائم * imatrix) {
    ثابت int64_t n = nrows * n_per_row;

    إذا (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != عدم);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ثابت size_t start_row = start / n_per_row;
    ثابت size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    بدل (type) {
        حالة GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (حرف *) dst + start_row * row_size, nrows, n_per_row, imatrix); اكسر;
        حالة GGML_TYPE_F16:
            {
                size_t elemsize = حجم(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                size_t elemsize = حجم(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                size_t elemsize = حجم(عائم);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } اكسر;
        افتراضي:
            أكد(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    أرجع result;
}

////////////////////////////////////////////////////////////////////////////////

فراغ ggml_log_get(ggml_log_callback * log_callback, فراغ ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

فراغ ggml_log_set(ggml_log_callback log_callback, فراغ * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

فراغ ggml_threadpool_params_init(بنية ggml_threadpool_params * p, صحيح n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

بنية ggml_threadpool_params ggml_threadpool_params_default(صحيح n_threads) {
    بنية ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    أرجع p;
}

bool ggml_threadpool_params_match(ثابت بنية ggml_threadpool_params * p0, ثابت بنية ggml_threadpool_params * p1) {
    إذا (p0->n_threads  != p1->n_threads  ) أرجع false;
    إذا (p0->prio       != p1->prio       ) أرجع false;
    إذا (p0->poll       != p1->poll       ) أرجع false;
    إذا (p0->strict_cpu != p1->strict_cpu ) أرجع false;
    أرجع memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
