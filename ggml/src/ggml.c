#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#شامل "ggml-version.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "ggml-threading.h"
#شامل "ggml-cpu.h"
#شامل "ggml.h"

// FIXME: required here for quantization functions
#شامل "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#شامل <hbwmalloc.h>
#endif

#اگر defined(_MSC_VER) || defined(__MINGW32__)
#شامل <malloc.h> // using malloc.h with MSC/MINGW
#وگرنهاگر !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#شامل <alloca.h>
#endif

#شامل <assert.h>
#شامل <errno.h>
#شامل <time.h>
#شامل <math.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <stdint.h>
#شامل <inttypes.h>
#شامل <stdio.h>
#شامل <float.h>
#شامل <limits.h>
#شامل <stdarg.h>
#شامل <signal.h>
#اگر defined(__gnu_linux__)
#شامل <syscall.h>
#endif

#اگر defined(__APPLE__)
#شامل <unistd.h>
#شامل <mach/mach.h>
#شامل <TargetConditionals.h>
#endif

#اگر defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#شامل <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(تهی) {
#ifdef _MSC_VER
#اگر defined(_WIN32)
    ایستا فرار LONG counter = 1;
    بازگشت (uint64_t) InterlockedIncrement(&counter) - 1;
#وگرنه
    ایستا فرار long long counter = 1;
    بازگشت (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#وگرنه
    ایستا uint64_t counter = 1;
    بازگشت __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#اگر defined(__AVX512BF16__)
#اگر defined(_MSC_VER)
#define m512i(p) p
#وگرنه
#شامل <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#اگر defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#شامل <unistd.h>
#شامل <sys/types.h>
#شامل <sys/stat.h>
#شامل <sys/wait.h>
#اگر defined(__linux__)
#شامل <sys/prctl.h>
#endif

#اگر defined(__ANDROID__)
#شامل <unwind.h>
#شامل <dlfcn.h>
#شامل <stdio.h>

ساختار backtrace_state {
    تهی ** current;
    تهی ** end;
};

ایستا _Unwind_Reason_Code unwind_callback(ساختار _Unwind_Context* context, تهی* arg) {
    ساختار backtrace_state * state = (ساختار backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    اگر (pc) {
        اگر (state->current == state->end) {
            بازگشت URC_END_OF_پشته;
        } وگرنه {
            *state->current++ = (تهی*)pc;
        }
    }
    بازگشت _URC_NO_REASON;
}

ایستا تهی ggml_print_backtrace_symbols(تهی) {
    ثابت صحیح max = 100;
    تهی* buffer[max];

    ساختار backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    صحیح شمار = state.current - buffer;

    برای (صحیح idx = 0; idx < شمار; ++idx) {
        ثابت تهی * addr = buffer[idx];
        ثابت نویسه * نماد = "";

        Dl_info info;
        اگر (dladdr(addr, &info) && info.dli_sname) {
            نماد = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, نماد);
    }
}
#وگرنهاگر defined(__linux__) && defined(__GLIBC__)
#شامل <execinfo.h>
ایستا تهی ggml_print_backtrace_symbols(تهی) {
    تهی * trace[100];
    صحیح nptrs = backtrace(trace, اندازه(trace)/اندازه(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وگرنهاگر defined(__APPLE__)
#شامل <execinfo.h>
ایستا تهی ggml_print_backtrace_symbols(تهی) {
    تهی * trace[100];
    صحیح nptrs = backtrace(trace, اندازه(trace)/اندازه(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#وگرنه
ایستا تهی ggml_print_backtrace_symbols(تهی) {
    // platform not supported
}
#endif

تهی ggml_print_backtrace(تهی) {
    ثابت نویسه * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    اگر (GGML_NO_BACKTRACE) {
        بازگشت;
    }
#اگر defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ثابت نویسه * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    اگر (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        بازگشت;
    }
#endif
#اگر defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    نویسه * سطر = پوچ;
    ssize_t length = 0;
    وقتی ((length = getline(&سطر, &size, f)) > 0) {
        اگر (!strncmp(سطر, "TracerPid:", اندازه("TracerPid:") - 1) &&
            (length != اندازه("TracerPid:\t0\n") - 1 || سطر[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            آزاد(سطر);
            fclose(f);
            بازگشت;
        }
    }
    آزاد(سطر);
    fclose(f);
    صحیح lock[2] = { -1, -1 };
    (تهی) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ثابت صحیح parent_pid = getpid();
    ثابت صحیح child_pid = fork();
    اگر (child_pid < 0) { // error
#اگر defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        بازگشت;
    } وگرنه اگر (child_pid == 0) { // child
        نویسه attach[32];
        snprintf(attach, اندازه(attach), "attach %d", parent_pid);
#اگر defined(__linux__)
        close(lock[1]);
        (تهی) !اقرأ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (نویسه *) پوچ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[اندازه("attach ") - 1],
            (نویسه *) پوچ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } وگرنه { // parent
#اگر defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, پوچ, 0);
    }
}
#وگرنه
تهی ggml_print_backtrace(تهی) {
    // platform not supported
}
#endif

ایستا ggml_abort_callback_t g_abort_callback = پوچ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    بازگشت ret_val;
}

تهی ggml_abort(ثابت نویسه * file, صحیح سطر, ثابت نویسه * fmt, ...) {
    fflush(stdout);

    نویسه message[2048];
    صحیح offset = snprintf(message, اندازه(message), "%s:%d: ", file, سطر);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, اندازه(message) - offset, fmt, args);
    va_end(args);

    اگر (g_abort_callback) {
        g_abort_callback(message);
    } وگرنه {
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

ساختار ggml_logger_state {
    ggml_log_callback log_callback;
    تهی * log_callback_user_data;
};
ایستا ساختار ggml_logger_state g_logger_state = {ggml_log_callback_default, پوچ};

ایستا تهی ggml_log_internal_v(شمارش ggml_log_level level, ثابت نویسه * format, va_list args) {
    اگر (format == پوچ) {
        بازگشت;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    نویسه buffer[128];
    صحیح len = vsnprintf(buffer, 128, format, args);
    اگر (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } وگرنه {
        نویسه * buffer2 = (نویسه *) calloc(len + 1, اندازه(نویسه));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        آزاد(buffer2);
    }
    va_end(args_copy);
}

تهی ggml_log_internal(شمارش ggml_log_level level, ثابت نویسه * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

تهی ggml_log_callback_default(شمارش ggml_log_level level, ثابت نویسه * text, تهی * user_data) {
    (تهی) level;
    (تهی) user_data;
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
#اگر defined(__has_feature)
#اگر __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#وگرنه
#اگر defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

تهی * ggml_aligned_malloc(size_t size) {
#اگر defined(__s390x__)
    ثابت صحیح alignment = 256;
#وگرنه
    ثابت صحیح alignment = 64;
#endif

#اگر defined(_MSC_VER) || defined(__MINGW32__)
    بازگشت _aligned_malloc(size, alignment);
#وگرنه
    اگر (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        بازگشت پوچ;
    }
    تهی * aligned_memory = پوچ;
  #ifdef GGML_USE_CPU_HBM
    صحیح result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #وگرنهاگر TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    صحیح result = EFAULT;
    انتخاب (alloc_status) {
        حالت KERN_SUCCESS:
            result = 0;
            بشکن;
        حالت KERN_INVALID_ADDRESS:
            result = EINVAL;
            بشکن;
        حالت KERN_NO_SPACE:
            result = ENOMEM;
            بشکن;
        پیشفرض:
            result = EFAULT;
            بشکن;
    }
  #وگرنه
    صحیح result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    اگر (result != 0) {
        // Handle allocation failure
        ثابت نویسه *error_desc = "unknown allocation error";
        انتخاب (result) {
            حالت EINVAL:
                error_desc = "invalid alignment value";
                بشکن;
            حالت ENOMEM:
                error_desc = "insufficient memory";
                بشکن;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        بازگشت پوچ;
    }
    بازگشت aligned_memory;
#endif
}

تهی ggml_aligned_free(تهی * ptr, size_t size) {
    GGML_UNUSED(size);
#اگر defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#وگرنهاگر GGML_USE_CPU_HBM
    اگر (ptr != پوچ) {
        hbw_free(ptr);
    }
#وگرنهاگر TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    اگر (ptr != پوچ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#وگرنه
    آزاد(ptr);
#endif
}


درونخطی ایستا تهی * ggml_malloc(size_t size) {
    اگر (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        بازگشت پوچ;
    }
    تهی * result = malloc(size);
    اگر (result == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    بازگشت result;
}

// calloc
درونخطی ایستا تهی * ggml_calloc(size_t رقم, size_t size) {
    اگر (رقم == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        بازگشت پوچ;
    }
    تهی * result = calloc(رقم, size);
    اگر (result == پوچ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    بازگشت result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(رقم, size) ggml_calloc(رقم, size)

#define GGML_FREE(ptr) آزاد(ptr)

ثابت نویسه * ggml_status_to_string(شمارش ggml_status status) {
    انتخاب (status) {
        حالت GGML_STATUS_ALLOC_FAILED: بازگشت "GGML status: error (failed to allocate memory)";
        حالت GGML_STATUS_FAILED:       بازگشت "GGML status: error (operation failed)";
        حالت GGML_STATUS_SUCCESS:      بازگشت "GGML status: success";
        حالت GGML_STATUS_ABORTED:      بازگشت "GGML status: warning (operation aborted)";
    }

    بازگشت "GGML status: unknown";
}

اعشاری ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    بازگشت GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(اعشاری x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    بازگشت GGML_FP32_TO_FP16(x);
}

اعشاری ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    بازگشت GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(اعشاری x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    بازگشت GGML_FP32_TO_BF16(x);
}

تهی ggml_fp16_to_fp32_row(ثابت ggml_fp16_t * x, اعشاری * y, int64_t n) {
    برای (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

تهی ggml_fp32_to_fp16_row(ثابت اعشاری * x, ggml_fp16_t * y, int64_t n) {
    صحیح i = 0;
    برای (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

تهی ggml_bf16_to_fp32_row(ثابت ggml_bf16_t * x, اعشاری * y, int64_t n) {
    صحیح i = 0;
    برای (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

تهی ggml_fp32_to_bf16_row_ref(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
    برای (صحیح i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

تهی ggml_fp32_to_bf16_row(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
  صحیح i = 0;
#اگر defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  برای (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    برای (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    بازگشت memcmp(guid_a, guid_b, اندازه(ggml_guid)) == 0;
}

ثابت نویسه * ggml_version(تهی) {
    بازگشت GGML_VERSION;
}

ثابت نویسه * ggml_commit(تهی) {
    بازگشت GGML_COMMIT;
}

//
// timing
//

#اگر defined(_MSC_VER) || defined(__MINGW32__)
ایستا int64_t timer_freq, timer_start;
ایستا BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    بازگشت TRUE;
}
تهی ggml_time_init(تهی) {
    ایستا INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, پوچ, پوچ);
}
int64_t ggml_time_ms(تهی) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    بازگشت ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(تهی) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    بازگشت ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#وگرنه
تهی ggml_time_init(تهی) {}
int64_t ggml_time_ms(تهی) {
    ساختار timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    بازگشت (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(تهی) {
    ساختار timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    بازگشت (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(تهی) {
    بازگشت clock();
}

int64_t ggml_cycles_per_ms(تهی) {
    بازگشت CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
ایستا wchar_t * ggml_mbstowcs(ثابت نویسه * mbs) {
    صحیح wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, پوچ, 0);
    اگر (!wlen) {
        errno = EINVAL;
        بازگشت پوچ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * اندازه(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    اگر (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        بازگشت پوچ;
    }

    بازگشت wbuf;
}
#endif

FILE * ggml_fopen(ثابت نویسه * fname, ثابت نویسه * mode) {
#ifdef _WIN32
    FILE * file = پوچ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    اگر (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        اگر (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    بازگشت file;
#وگرنه
    بازگشت fopen(fname, mode);
#endif

}

ایستا ثابت ساختار ggml_type_traits type_traits[GGML_TYPE_شمار] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = اندازه(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = اندازه(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = اندازه(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = اندازه(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = اندازه(دوگانه),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = اندازه(اعشاری),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = اندازه(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = اندازه(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = اندازه(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = اندازه(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = اندازه(block_q4_1),
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
        .type_size                = اندازه(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = اندازه(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = اندازه(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = اندازه(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = اندازه(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = اندازه(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = پوچ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = پوچ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = پوچ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = پوچ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = اندازه(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = اندازه(ggml_bf16_t),
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
        .type_size                = اندازه(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = اندازه(block_tq2_0),
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

ثابت ساختار ggml_type_traits * ggml_get_type_traits(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت &type_traits[type];
}

//
// ggml object
//

ساختار ggml_object {
    size_t offs;
    size_t size;

    ساختار ggml_object * next;

    شمارش ggml_object_type type;

    نویسه padding[4];
};

ایستا ثابت size_t GGML_OBJECT_SIZE = اندازه(ساختار ggml_object);

//
// ggml context
//

ساختار ggml_context {
    size_t mem_size;
    تهی * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    صحیح    n_objects;

    ساختار ggml_object * objects_begin;
    ساختار ggml_object * objects_end;
};

//
// data types
//

ایستا ثابت نویسه * GGML_OP_NAME[GGML_OP_شمار] = {
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

static_assert(GGML_OP_شمار == 102, "GGML_OP_COUNT != 102");

ایستا ثابت نویسه * GGML_OP_نماد[GGML_OP_شمار] = {
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

static_assert(GGML_OP_شمار == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_شمار == 2, "GGML_OP_POOL_COUNT != 2");

ایستا ثابت نویسه * GGML_UNARY_OP_NAME[GGML_UNARY_OP_شمار] = {
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

static_assert(GGML_UNARY_OP_شمار == 22, "GGML_UNARY_OP_COUNT != 22");

ایستا ثابت نویسه * GGML_GLU_OP_NAME[GGML_GLU_OP_شمار] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_شمار == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(اندازه(ساختار ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(اندازه(ساختار ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

تهی ggml_print_object(ثابت ساختار ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ثابت تهی *) obj->next);
}

تهی ggml_print_objects(ثابت ساختار ggml_context * ctx) {
    ساختار ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ثابت تهی *) ctx);

    وقتی (obj != پوچ) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ثابت ساختار ggml_tensor * tensor) {
    برای (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        اگر (tensor->ne[i] <= 0) {
            بازگشت 0;
        }
    }

    size_t nbytes;
    ثابت size_t blck_size = ggml_blck_size(tensor->type);
    اگر (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        برای (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    وگرنه {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        برای (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    بازگشت nbytes;
}

size_t ggml_nbytes_pad(ثابت ساختار ggml_tensor * tensor) {
    بازگشت GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت type_traits[type].blck_size;
}

size_t ggml_type_size(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت type_traits[type].type_size;
}

size_t ggml_row_size(شمارش ggml_type type, int64_t ne) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    أكد(ne % ggml_blck_size(type) == 0);
    بازگشت ggml_type_size(type)*ne/ggml_blck_size(type);
}

دوگانه ggml_type_sizef(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت ((دوگانه)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ثابت نویسه * ggml_type_name(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت type_traits[type].type_name;
}

bool ggml_is_quantized(شمارش ggml_type type) {
    أكد(type >= 0);
    أكد(type < GGML_TYPE_شمار);
    بازگشت type_traits[type].is_quantized;
}

ثابت نویسه * ggml_op_name(شمارش ggml_op op) {
    بازگشت GGML_OP_NAME[op];
}

ثابت نویسه * ggml_op_نماد(شمارش ggml_op op) {
    بازگشت GGML_OP_نماد[op];
}

ثابت نویسه * ggml_unary_op_name(شمارش ggml_unary_op op) {
    بازگشت GGML_UNARY_OP_NAME[op];
}

ثابت نویسه * ggml_glu_op_name(شمارش ggml_glu_op op) {
    بازگشت GGML_GLU_OP_NAME[op];
}

ثابت نویسه * ggml_op_desc(ثابت ساختار ggml_tensor * t) {
    اگر (t->op == GGML_OP_UNARY) {
        شمارش ggml_unary_op uop = ggml_get_unary_op(t);
        بازگشت ggml_unary_op_name(uop);
    }
    اگر (t->op == GGML_OP_GLU) {
        شمارش ggml_glu_op gop = ggml_get_glu_op(t);
        بازگشت ggml_glu_op_name(gop);
    }
    بازگشت ggml_op_name(t->op);
}

size_t ggml_element_size(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ثابت ساختار ggml_tensor * tensor) {
    بازگشت tensor->ne[3] == 1;
}

صحیح ggml_n_dims(ثابت ساختار ggml_tensor * tensor) {
    برای (صحیح i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        اگر (tensor->ne[i] > 1) {
            بازگشت i + 1;
        }
    }
    بازگشت 1;
}

شمارش ggml_type ggml_ftype_to_ggml_type(شمارش ggml_ftype ftype) {
    شمارش ggml_type wtype = GGML_TYPE_شمار;

    انتخاب (ftype) {
        حالت GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   بشکن;
        حالت GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   بشکن;
        حالت GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  بشکن;
        حالت GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; بشکن;
        حالت GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; بشکن;
        حالت GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  بشکن;
        حالت GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    بشکن;
        حالت GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    بشکن;
        حالت GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_شمار; بشکن;
        حالت GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_شمار; بشکن;
    }

    GGML_ASSERT(wtype != GGML_TYPE_شمار);

    بازگشت wtype;
}

size_t ggml_tensor_overhead(تهی) {
    بازگشت GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ثابت ساختار ggml_tensor * tensor) {
    بازگشت tensor->nb[0] > tensor->nb[1];
}

ایستا bool ggml_is_contiguous_m_n(ثابت ساختار ggml_tensor * tensor, صحیح m, صحیح n) {
    size_t next_nb = ggml_type_size(tensor->type);
    اگر (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        بازگشت false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    برای (صحیح i = 1; i < n; i++) {
        اگر (i > m) {
            اگر (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                بازگشت false;
            }
            next_nb *= tensor->ne[i];
        } وگرنه {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    بازگشت true;
}

bool ggml_is_contiguous(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ثابت ساختار ggml_tensor * tensor) {
    بازگشت ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ثابت ساختار ggml_tensor * tensor) {
    بازگشت
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ثابت ساختار ggml_tensor * tensor) {
    بازگشت
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

ایستا درونخطی bool ggml_is_padded_1d(ثابت ساختار ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ثابت ساختار ggml_tensor * tensor) {
    برای (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        اگر (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            بازگشت true;
        }
    }
    بازگشت false;
}

bool ggml_are_same_shape(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ثابت ساختار ggml_tensor * t) {
    بازگشت ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

ایستا درونخطی bool ggml_can_repeat_rows(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

ساختار ggml_context * ggml_init(ساختار ggml_init_params params) {
    ایستا bool is_first_call = true;

    ggml_critical_section_start();

    اگر (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    ساختار ggml_context * ctx = GGML_MALLOC(اندازه(ساختار ggml_context));

    // allow to call ggml_init with 0 size
    اگر (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ثابت size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (ساختار ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ پوچ,
        /*.objects_end        =*/ پوچ,
    };

    GGML_ASSERT(ctx->mem_buffer != پوچ);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    بازگشت ctx;
}

تهی ggml_reset(ساختار ggml_context * ctx) {
    اگر (ctx == پوچ) {
        بازگشت;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = پوچ;
    ctx->objects_end   = پوچ;
}

تهی ggml_free(ساختار ggml_context * ctx) {
    اگر (ctx == پوچ) {
        بازگشت;
    }

    اگر (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ثابت ساختار ggml_context * ctx) {
    بازگشت ctx->objects_end == پوچ ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(ساختار ggml_context * ctx) {
    بازگشت ctx->no_alloc;
}

تهی ggml_set_no_alloc(ساختار ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

تهی * ggml_get_mem_buffer(ثابت ساختار ggml_context * ctx) {
    بازگشت ctx->mem_buffer;
}

size_t ggml_get_mem_size(ثابت ساختار ggml_context * ctx) {
    بازگشت ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ثابت ساختار ggml_context * ctx) {
    size_t max_size = 0;

    برای (ساختار ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != پوچ; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    بازگشت max_size;
}

////////////////////////////////////////////////////////////////////////////////

ایستا ساختار ggml_object * ggml_new_object(ساختار ggml_context * ctx, شمارش ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    ساختار ggml_object * obj_cur = ctx->objects_end;

    ثابت size_t cur_offs = obj_cur == پوچ ? 0 : obj_cur->offs;
    ثابت size_t cur_size = obj_cur == پوچ ? 0 : obj_cur->size;
    ثابت size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    نویسه * ثابت mem_buffer = ctx->mem_buffer;
    ساختار ggml_object * ثابت obj_new = (ساختار ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    اگر (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        بازگشت پوچ;
    }
    اگر (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        بازگشت پوچ;
    }

    اگر (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        بازگشت پوچ;
    }

    *obj_new = (ساختار ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = پوچ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    اگر (obj_cur != پوچ) {
        obj_cur->next = obj_new;
    } وگرنه {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    بازگشت obj_new;
}

ایستا ساختار ggml_tensor * ggml_new_tensor_impl(
        ساختار ggml_context * ctx,
        شمارش   ggml_type      type,
        صحیح                   n_dims,
        ثابت int64_t       * ne,
        ساختار ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_شمار);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    اگر (view_src != پوچ && view_src->view_src != پوچ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    برای (صحیح i = 1; i < n_dims; i++) {
        اگر (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    برای (صحیح i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == پوچ || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    تهی * data = view_src != پوچ ? view_src->data : پوچ;
    اگر (data != پوچ) {
        data = (نویسه *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    اگر (view_src == پوچ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    ساختار ggml_object * ثابت obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    ساختار ggml_tensor * ثابت result = (ساختار ggml_tensor *)((نویسه *)ctx->mem_buffer + obj_new->offs);

    *result = (ساختار ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ پوچ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { پوچ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (تهی *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ پوچ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    برای (صحیح i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    برای (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    بازگشت result;
}

ساختار ggml_tensor * ggml_new_tensor(
        ساختار ggml_context * ctx,
        شمارش   ggml_type      type,
        صحیح                   n_dims,
        ثابت int64_t       * ne) {
    بازگشت ggml_new_tensor_impl(ctx, type, n_dims, ne, پوچ, 0);
}

ساختار ggml_tensor * ggml_new_tensor_1d(
        ساختار ggml_context * ctx,
        شمارش   ggml_type      type,
        int64_t ne0) {
    بازگشت ggml_new_tensor(ctx, type, 1, &ne0);
}

ساختار ggml_tensor * ggml_new_tensor_2d(
        ساختار ggml_context * ctx,
        شمارش   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ثابت int64_t ne[2] = { ne0, ne1 };
    بازگشت ggml_new_tensor(ctx, type, 2, ne);
}

ساختار ggml_tensor * ggml_new_tensor_3d(
        ساختار ggml_context * ctx,
        شمارش   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    بازگشت ggml_new_tensor(ctx, type, 3, ne);
}

ساختار ggml_tensor * ggml_new_tensor_4d(
        ساختار ggml_context * ctx,
        شمارش   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    بازگشت ggml_new_tensor(ctx, type, 4, ne);
}

تهی * ggml_new_buffer(ساختار ggml_context * ctx, size_t nbytes) {
    ساختار ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    بازگشت (uint8_t *)ctx->mem_buffer + obj->offs;
}

ساختار ggml_tensor * ggml_dup_tensor(ساختار ggml_context * ctx, ثابت ساختار ggml_tensor * src) {
    بازگشت ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

تهی ggml_unravel_index(ثابت ساختار ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ثابت int64_t ne2 = tensor->ne[2];
    ثابت int64_t ne1 = tensor->ne[1];
    ثابت int64_t ne0 = tensor->ne[0];

    ثابت int64_t i3_ = (i/(ne2*ne1*ne0));
    ثابت int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ثابت int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ثابت int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    اگر (i0) {
        * i0 = i0_;
    }
    اگر (i1) {
        * i1 = i1_;
    }
    اگر (i2) {
        * i2 = i2_;
    }
    اگر (i3) {
        * i3 = i3_;
    }
}

تهی * ggml_get_data(ثابت ساختار ggml_tensor * tensor) {
    بازگشت tensor->data;
}

اعشاری * ggml_get_data_f32(ثابت ساختار ggml_tensor * tensor) {
    أكد(tensor->type == GGML_TYPE_F32);
    بازگشت (اعشاری *)(tensor->data);
}

شمارش ggml_unary_op ggml_get_unary_op(ثابت ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    بازگشت (شمارش ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

شمارش ggml_glu_op ggml_get_glu_op(ثابت ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    بازگشت (شمارش ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ثابت نویسه * ggml_get_name(ثابت ساختار ggml_tensor * tensor) {
    بازگشت tensor->name;
}

ساختار ggml_tensor * ggml_set_name(ساختار ggml_tensor * tensor, ثابت نویسه * name) {
    size_t i;
    برای (i = 0; i < اندازه(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    بازگشت tensor;
}

ساختار ggml_tensor * ggml_format_name(ساختار ggml_tensor * tensor, ثابت نویسه * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, اندازه(tensor->name), fmt, args);
    va_end(args);
    بازگشت tensor;
}

ساختار ggml_tensor * ggml_view_tensor(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * src) {
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    بازگشت result;
}

ساختار ggml_tensor * ggml_get_first_tensor(ثابت ساختار ggml_context * ctx) {
    ساختار ggml_object * obj = ctx->objects_begin;

    نویسه * ثابت mem_buffer = ctx->mem_buffer;

    وقتی (obj != پوچ) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            بازگشت (ساختار ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    بازگشت پوچ;
}

ساختار ggml_tensor * ggml_get_next_tensor(ثابت ساختار ggml_context * ctx, ساختار ggml_tensor * tensor) {
    ساختار ggml_object * obj = (ساختار ggml_object *) ((نویسه *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    نویسه * ثابت mem_buffer = ctx->mem_buffer;

    وقتی (obj != پوچ) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            بازگشت (ساختار ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    بازگشت پوچ;
}

ساختار ggml_tensor * ggml_get_tensor(ساختار ggml_context * ctx, ثابت نویسه * name) {
    ساختار ggml_object * obj = ctx->objects_begin;

    نویسه * ثابت mem_buffer = ctx->mem_buffer;

    وقتی (obj != پوچ) {
        اگر (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            ساختار ggml_tensor * cur = (ساختار ggml_tensor *)(mem_buffer + obj->offs);
            اگر (strcmp(cur->name, name) == 0) {
                بازگشت cur;
            }
        }

        obj = obj->next;
    }

    بازگشت پوچ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

ایستا ساختار ggml_tensor * ggml_dup_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_dup(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_dup_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_dup_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_dup_impl(ctx, a, true);
}

// ggml_add

ایستا ساختار ggml_tensor * ggml_add_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_add(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_add_impl(ctx, a, b, false);
}

ساختار ggml_tensor * ggml_add_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

ایستا ساختار ggml_tensor * ggml_add_cast_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        شمارش   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_add_cast(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        شمارش   ggml_type      type) {
    بازگشت ggml_add_cast_impl(ctx, a, b, type);
}

ساختار ggml_tensor * ggml_add_id(
            ساختار ggml_context * ctx,
            ساختار ggml_tensor  * a,
            ساختار ggml_tensor  * b,
            ساختار ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    بازگشت result;
}

// ggml_add1

ایستا ساختار ggml_tensor * ggml_add1_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_add1(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_add1_impl(ctx, a, b, false);
}

ساختار ggml_tensor * ggml_add1_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

ایستا ساختار ggml_tensor * ggml_acc_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_acc(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    بازگشت ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ساختار ggml_tensor * ggml_acc_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    بازگشت ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

ایستا ساختار ggml_tensor * ggml_sub_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_sub(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_sub_impl(ctx, a, b, false);
}

ساختار ggml_tensor * ggml_sub_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

ایستا ساختار ggml_tensor * ggml_mul_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_mul(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_mul_impl(ctx, a, b, false);
}

ساختار ggml_tensor * ggml_mul_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

ایستا ساختار ggml_tensor * ggml_div_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_div(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_div_impl(ctx, a, b, false);
}

ساختار ggml_tensor * ggml_div_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

ایستا ساختار ggml_tensor * ggml_sqr_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_sqr(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sqr_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_sqr_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

ایستا ساختار ggml_tensor * ggml_sqrt_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_sqrt(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sqrt_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_sqrt_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

ایستا ساختار ggml_tensor * ggml_log_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_log(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_log_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_log_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_log_impl(ctx, a, true);
}

ساختار ggml_tensor * ggml_expm1(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

ساختار ggml_tensor * ggml_expm1_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

ساختار ggml_tensor * ggml_softplus(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

ساختار ggml_tensor * ggml_softplus_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

ایستا ساختار ggml_tensor * ggml_sin_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_sin(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sin_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_sin_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_sin_impl(ctx, a, true);
}

// ggml_cos

ایستا ساختار ggml_tensor * ggml_cos_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_cos(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_cos_impl(ctx, a, false);
}

ساختار ggml_tensor * ggml_cos_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_cos_impl(ctx, a, true);
}

// ggml_sum

ساختار ggml_tensor * ggml_sum(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    بازگشت result;
}

// ggml_sum_rows

ساختار ggml_tensor * ggml_sum_rows(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    برای (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    بازگشت result;
}

// ggml_cumsum

ساختار ggml_tensor * ggml_cumsum(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    بازگشت result;
}

// ggml_mean

ساختار ggml_tensor * ggml_mean(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    بازگشت result;
}

// ggml_argmax

ساختار ggml_tensor * ggml_argmax(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    بازگشت result;
}

// ggml_count_equal

ساختار ggml_tensor * ggml_شمار_equal(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_شمار_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_repeat

ساختار ggml_tensor * ggml_repeat(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_repeat_4d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ثابت bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    بازگشت result;
}

// ggml_repeat_back

ساختار ggml_tensor * ggml_repeat_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    بازگشت result;
}

// ggml_concat

ساختار ggml_tensor * ggml_concat(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    ساختار ggml_tensor  * b,
    صحیح                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    برای (صحیح d = 0; d < GGML_MAX_DIMS; ++d) {
        اگر (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            ادامه;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_abs

ساختار ggml_tensor * ggml_abs(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ساختار ggml_tensor * ggml_abs_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

ساختار ggml_tensor * ggml_sgn(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ساختار ggml_tensor * ggml_sgn_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

ساختار ggml_tensor * ggml_neg(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ساختار ggml_tensor * ggml_neg_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

ساختار ggml_tensor * ggml_step(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

ساختار ggml_tensor * ggml_step_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

ساختار ggml_tensor * ggml_tanh(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

ساختار ggml_tensor * ggml_tanh_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

ساختار ggml_tensor * ggml_elu(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

ساختار ggml_tensor * ggml_elu_inplace(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

ساختار ggml_tensor * ggml_relu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

ساختار ggml_tensor * ggml_relu_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

ساختار ggml_tensor * ggml_leaky_relu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 negative_slope,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, اندازه(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    بازگشت result;
}

// ggml_sigmoid

ساختار ggml_tensor * ggml_sigmoid(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ساختار ggml_tensor * ggml_sigmoid_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

ساختار ggml_tensor * ggml_gelu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

ساختار ggml_tensor * ggml_gelu_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

ساختار ggml_tensor * ggml_gelu_erf(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ساختار ggml_tensor * ggml_gelu_erf_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

ساختار ggml_tensor * ggml_gelu_quick(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

ساختار ggml_tensor * ggml_gelu_quick_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

ساختار ggml_tensor * ggml_silu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ساختار ggml_tensor * ggml_silu_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

ساختار ggml_tensor * ggml_xielu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری alpha_n,
        اعشاری alpha_p,
        اعشاری beta,
        اعشاری eps) {
    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    بازگشت result;
}

// ggml_silu_back

ساختار ggml_tensor * ggml_silu_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml hardswish

ساختار ggml_tensor * ggml_hardswish(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

ساختار ggml_tensor * ggml_hardsigmoid(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

ساختار ggml_tensor * ggml_exp(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ساختار ggml_tensor * ggml_exp_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

ایستا ساختار ggml_tensor * ggml_glu_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        شمارش ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    اگر (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; برای (صحیح i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, پوچ, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_floor

ساختار ggml_tensor * ggml_floor(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

ساختار ggml_tensor * ggml_floor_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

ساختار ggml_tensor * ggml_ceil(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

ساختار ggml_tensor * ggml_ceil_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

ساختار ggml_tensor * ggml_round(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

ساختار ggml_tensor * ggml_round_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

ساختار ggml_tensor * ggml_trunc(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

ساختار ggml_tensor * ggml_trunc_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

ساختار ggml_tensor * ggml_glu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_glu_op      op,
        bool                  swapped) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, op, swapped);
}

ساختار ggml_tensor * ggml_glu_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        شمارش ggml_glu_op      op) {
    بازگشت ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

ساختار ggml_tensor * ggml_reglu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_REGLU, false);
}

ساختار ggml_tensor * ggml_reglu_swapped(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_REGLU, true);
}

ساختار ggml_tensor * ggml_reglu_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

ساختار ggml_tensor * ggml_geglu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU, false);
}

ساختار ggml_tensor * ggml_geglu_swapped(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU, true);
}

ساختار ggml_tensor * ggml_geglu_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

ساختار ggml_tensor * ggml_swiglu(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_SWIGLU, false);
}

ساختار ggml_tensor * ggml_swiglu_swapped(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_SWIGLU, true);
}

ساختار ggml_tensor * ggml_swiglu_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

ساختار ggml_tensor * ggml_geglu_erf(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU_ERF, false);
}

ساختار ggml_tensor * ggml_geglu_erf_swapped(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU_ERF, true);
}

ساختار ggml_tensor * ggml_geglu_erf_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

ساختار ggml_tensor * ggml_geglu_quick(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU_QUICK, false);
}

ساختار ggml_tensor * ggml_geglu_quick_swapped(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_glu_impl(ctx, a, پوچ, GGML_GLU_OP_GEGLU_QUICK, true);
}

ساختار ggml_tensor * ggml_geglu_quick_split(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

ساختار ggml_tensor * ggml_swiglu_oai(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 alpha,
        اعشاری                 limit) {
    ساختار ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    بازگشت result;
}

ساختار ggml_tensor * ggml_swiglu_clamp(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 limit) {
    ساختار ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    بازگشت result;
}

// ggml_norm

ایستا ساختار ggml_tensor * ggml_norm_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, اندازه(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_norm(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_norm_impl(ctx, a, eps, false);
}

ساختار ggml_tensor * ggml_norm_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

ایستا ساختار ggml_tensor * ggml_rms_norm_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, اندازه(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_rms_norm(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_rms_norm_impl(ctx, a, eps, false);
}

ساختار ggml_tensor * ggml_rms_norm_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

ساختار ggml_tensor * ggml_rms_norm_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 eps) {
    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, اندازه(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_group_norm

ایستا ساختار ggml_tensor * ggml_group_norm_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_group_norm(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps) {
    بازگشت ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

ساختار ggml_tensor * ggml_group_norm_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_groups,
        اعشاری                 eps) {
    بازگشت ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

ایستا ساختار ggml_tensor * ggml_l2_norm_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_l2_norm(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_l2_norm_impl(ctx, a, eps, false);
}

ساختار ggml_tensor * ggml_l2_norm_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 eps) {
    بازگشت ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        ساختار ggml_tensor * a,
        شمارش ggml_prec       prec) {
    انتخاب (a->op) {
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            بشکن;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            بشکن;
        پیشفرض:
            بازگشت false;
    };

    بازگشت true;
}

bool ggml_prec_set_src(
        ساختار ggml_tensor * a,
        شمارش ggml_prec       prec,
        صحیح                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    انتخاب (a->op) {
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
            {
                اگر (idx != 1) {
                    بازگشت false;
                }

                ثابت int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            بشکن;
        پیشفرض:
            بازگشت false;
    };

    بازگشت true;
}

// ggml_mul_mat

ایستا درونخطی bool ggml_can_mul_mat(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ساختار ggml_tensor * ggml_mul_mat(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ثابت int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

تهی ggml_mul_mat_set_prec(
        ساختار ggml_tensor * a,
        شمارش ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

تهی ggml_mul_mat_set_hint(
        ساختار ggml_tensor * a,
        شمارش ggml_op_hint    hint) {
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
ساختار ggml_tensor * ggml_mul_mat_id(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * as,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ثابت int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    بازگشت result;
}

// ggml_out_prod

ایستا درونخطی bool ggml_can_out_prod(ثابت ساختار ggml_tensor * t0, ثابت ساختار ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    بازگشت (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

ساختار ggml_tensor * ggml_out_prod(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ثابت int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_scale

ایستا ساختار ggml_tensor * ggml_scale_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[2] = { s, b };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_scale(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 s) {
    بازگشت ggml_scale_impl(ctx, a, s, 0.0, false);
}

ساختار ggml_tensor * ggml_scale_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 s) {
    بازگشت ggml_scale_impl(ctx, a, s, 0.0, true);
}

ساختار ggml_tensor * ggml_scale_bias(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b) {
    بازگشت ggml_scale_impl(ctx, a, s, b, false);
}

ساختار ggml_tensor * ggml_scale_bias_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 s,
        اعشاری                 b) {
    بازگشت ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

ایستا ساختار ggml_tensor * ggml_set_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_set(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ساختار ggml_tensor * ggml_set_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

ساختار ggml_tensor * ggml_set_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

ساختار ggml_tensor * ggml_set_1d_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

ساختار ggml_tensor * ggml_set_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

ساختار ggml_tensor * ggml_set_2d_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    بازگشت ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

ایستا ساختار ggml_tensor * ggml_cpy_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    ساختار ggml_tensor * result = ggml_view_tensor(ctx, b);
    اگر (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } وگرنه {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_cpy(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor * a,
        ساختار ggml_tensor * b) {
    بازگشت ggml_cpy_impl(ctx, a, b);
}

ساختار ggml_tensor * ggml_cast(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش   ggml_type      type) {
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    بازگشت result;
}

// ggml_cont

ایستا ساختار ggml_tensor * ggml_cont_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_cont(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor * a) {
    بازگشت ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API ساختار ggml_tensor * ggml_cont_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0) {
    بازگشت ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API ساختار ggml_tensor * ggml_cont_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    بازگشت ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API ساختار ggml_tensor * ggml_cont_3d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    بازگشت ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

ساختار ggml_tensor * ggml_cont_4d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    بازگشت result;
}

// ggml_reshape

ساختار ggml_tensor * ggml_reshape(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor * a,
        ساختار ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_reshape_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ثابت int64_t ne[1] = { ne0 };
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_reshape_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ثابت int64_t ne[2] = { ne0, ne1 };
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_reshape_3d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ثابت int64_t ne[3] = { ne0, ne1, ne2 };
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_reshape_4d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    بازگشت result;
}

ایستا ساختار ggml_tensor * ggml_view_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_dims,
        ثابت int64_t       * ne,
        size_t                offset) {
    ساختار ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, اندازه(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    بازگشت result;
}

// ggml_view_1d

ساختار ggml_tensor * ggml_view_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    ساختار ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    بازگشت result;
}

// ggml_view_2d

ساختار ggml_tensor * ggml_view_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ثابت int64_t ne[2] = { ne0, ne1 };

    ساختار ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    بازگشت result;
}

// ggml_view_3d

ساختار ggml_tensor * ggml_view_3d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ثابت int64_t ne[3] = { ne0, ne1, ne2 };

    ساختار ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    بازگشت result;
}

// ggml_view_4d

ساختار ggml_tensor * ggml_view_4d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ثابت int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    ساختار ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    بازگشت result;
}

// ggml_permute

ساختار ggml_tensor * ggml_permute(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   axis0,
        صحیح                   axis1,
        صحیح                   axis2,
        صحیح                   axis3) {
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

    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, اندازه(params));

    بازگشت result;
}

// ggml_transpose

ساختار ggml_tensor * ggml_transpose(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    بازگشت result;
}

// ggml_get_rows

ساختار ggml_tensor * ggml_get_rows(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    شمارش ggml_type type = GGML_TYPE_F32;
    اگر (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_get_rows_back

ساختار ggml_tensor * ggml_get_rows_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ساختار ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_set_rows

ساختار ggml_tensor * ggml_set_rows(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c) {
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

    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    بازگشت result;
}

// ggml_diag

ساختار ggml_tensor * ggml_diag(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ثابت int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    بازگشت result;
}

// ggml_diag_mask_inf

ایستا ساختار ggml_tensor * ggml_diag_mask_inf_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_diag_mask_inf(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past) {
    بازگشت ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

ساختار ggml_tensor * ggml_diag_mask_inf_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past) {
    بازگشت ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

ایستا ساختار ggml_tensor * ggml_diag_mask_zero_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_diag_mask_zero(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past) {
    بازگشت ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

ساختار ggml_tensor * ggml_diag_mask_zero_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   n_past) {
    بازگشت ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

ایستا ساختار ggml_tensor * ggml_clamp_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        اعشاری                 min,
        اعشاری                 max,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[] = { min, max };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_clamp(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    اعشاری                 min,
    اعشاری                 max) {
    بازگشت ggml_clamp_impl(ctx, a, min, max, false);
}

ساختار ggml_tensor * ggml_clamp_inplace(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    اعشاری                 min,
    اعشاری                 max) {
    بازگشت ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

ایستا ساختار ggml_tensor * ggml_soft_max_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * mask,
        اعشاری                 scale,
        اعشاری                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    اگر (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    اگر (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    اعشاری params[] = { scale, max_bias };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    بازگشت result;
}

ساختار ggml_tensor * ggml_soft_max(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_soft_max_impl(ctx, a, پوچ, 1.0f, 0.0f, false);
}

ساختار ggml_tensor * ggml_soft_max_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a) {
    بازگشت ggml_soft_max_impl(ctx, a, پوچ, 1.0f, 0.0f, true);
}

ساختار ggml_tensor * ggml_soft_max_ext(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * mask,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    بازگشت ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

ساختار ggml_tensor * ggml_soft_max_ext_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * mask,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    بازگشت ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

تهی ggml_soft_max_add_sinks(
        ساختار ggml_tensor * a,
        ساختار ggml_tensor * sinks) {
    اگر (!sinks) {
        a->src[2] = پوچ;
        بازگشت;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == پوچ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

ایستا ساختار ggml_tensor * ggml_soft_max_ext_back_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias,
        bool                  inplace) {
    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((اعشاری *) result->op_params + 0, &scale,    اندازه(اعشاری));
    memcpy((اعشاری *) result->op_params + 1, &max_bias, اندازه(اعشاری));

    بازگشت result;
}

ساختار ggml_tensor * ggml_soft_max_ext_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    بازگشت ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

ساختار ggml_tensor * ggml_soft_max_ext_back_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        اعشاری                 scale,
        اعشاری                 max_bias) {
    بازگشت ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

ایستا ساختار ggml_tensor * ggml_rope_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    اگر (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } وگرنه {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    اگر (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    اندازه(اعشاری));
    memcpy(params +  6, &freq_scale,   اندازه(اعشاری));
    memcpy(params +  7, &ext_factor,   اندازه(اعشاری));
    memcpy(params +  8, &attn_factor,  اندازه(اعشاری));
    memcpy(params +  9, &beta_fast,    اندازه(اعشاری));
    memcpy(params + 10, &beta_slow,    اندازه(اعشاری));
    اگر (mrope_used && sections) {
        memcpy(params + 11, sections,  اندازه(int32_t) * GGML_MROPE_SECTIONS);
    } وگرنه {
        memset(params + 11, 0,         اندازه(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    بازگشت result;
}

ساختار ggml_tensor * ggml_rope(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   mode) {
    بازگشت ggml_rope_impl(
        ctx, a, b, پوچ, n_dims, پوچ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

ساختار ggml_tensor * ggml_rope_multi(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ساختار ggml_tensor * ggml_rope_multi_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[GGML_MROPE_SECTIONS],
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ساختار ggml_tensor * ggml_rope_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   mode) {
    بازگشت ggml_rope_impl(
        ctx, a, b, پوچ, n_dims, پوچ, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

ساختار ggml_tensor * ggml_rope_ext(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, c, n_dims, پوچ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ساختار ggml_tensor * ggml_rope_ext_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, c, n_dims, پوچ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

ساختار ggml_tensor * ggml_rope_custom(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, پوچ, n_dims, پوچ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

ساختار ggml_tensor * ggml_rope_custom_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   n_dims,
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    بازگشت ggml_rope_impl(
        ctx, a, b, پوچ, n_dims, پوچ, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
ایستا اعشاری ggml_rope_yarn_corr_dim(صحیح n_dims, صحیح n_ctx_orig, اعشاری n_rot, اعشاری base) {
    بازگشت n_dims * logf(n_ctx_orig / (n_rot * 2 * (اعشاری)M_PI)) / (2 * logf(base));
}

تهی ggml_rope_yarn_corr_dims(
    صحیح n_dims, صحیح n_ctx_orig, اعشاری freq_base, اعشاری beta_fast, اعشاری beta_slow, اعشاری dims[2]
) {
    // start and end correction dims
    اعشاری start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    اعشاری end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

ساختار ggml_tensor * ggml_rope_ext_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    ساختار ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    بازگشت result;
}

ساختار ggml_tensor * ggml_rope_multi_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c,
        صحیح                   n_dims,
        صحیح                   sections[4],
        صحیح                   mode,
        صحیح                   n_ctx_orig,
        اعشاری                 freq_base,
        اعشاری                 freq_scale,
        اعشاری                 ext_factor,
        اعشاری                 attn_factor,
        اعشاری                 beta_fast,
        اعشاری                 beta_slow) {
    ساختار ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    بازگشت result;
}

ساختار ggml_tensor * ggml_rope_set_offset(
        ساختار ggml_tensor  * a,
        صحیح                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ثابت int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    بازگشت a;
}

ایستا int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, صحیح s, صحیح p, صحیح d) {
    بازگشت (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ساختار ggml_tensor * ggml_im2col(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1,
        bool                  is_2D,
        شمارش ggml_type        dst_type) {
    اگر (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } وگرنه {
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

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_im2col_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        int64_t             * ne,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1,
        bool                  is_2D) {
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_conv_1d

ساختار ggml_tensor * ggml_conv_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    ساختار ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    ساختار ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    بازگشت result;
}

// ggml_conv_1d_ph

ساختار ggml_tensor* ggml_conv_1d_ph(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s,
        صحیح                   d) {
    بازگشت ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

ساختار ggml_tensor * ggml_conv_1d_dw(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    ساختار ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    ساختار ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    ساختار ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    بازگشت result;
}

// ggml_conv_1d_dw_ph

ساختار ggml_tensor * ggml_conv_1d_dw_ph(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   d0) {
    بازگشت ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

ساختار ggml_tensor * ggml_col2im_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   s0,
        صحیح                   oc,
        صحیح                   p0) {
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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    بازگشت result;
}

// ggml_conv_transpose_1d

ایستا int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, صحیح s, صحیح p, صحیح d) {
    بازگشت (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API ساختار ggml_tensor * ggml_conv_transpose_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   p0,
        صحیح                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
ساختار ggml_tensor * ggml_conv_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1) {
    ساختار ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    ساختار ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    بازگشت result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ساختار ggml_tensor * ggml_im2col_3d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        int64_t               IC,
        صحیح                   s0, // stride width
        صحیح                   s1, // stride height
        صحیح                   s2, // stride depth
        صحیح                   p0, // padding width
        صحیح                   p1, // padding height
        صحیح                   p2, // padding depth
        صحیح                   d0, // dilation width
        صحیح                   d1, // dilation height
        صحیح                   d2, // dilation depth
        شمارش ggml_type        dst_type) {
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

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
ساختار ggml_tensor * ggml_conv_3d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        int64_t               IC,
        صحیح                   s0, // stride width
        صحیح                   s1, // stride height
        صحیح                   s2, // stride depth
        صحیح                   p0, // padding width
        صحیح                   p1, // padding height
        صحیح                   p2, // padding depth
        صحیح                   d0, // dilation width
        صحیح                   d1, // dilation height
        صحیح                   d2  // dilation depth
        ) {
    ساختار ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    ساختار ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    بازگشت result;
}

// ggml_conv_2d_sk_p0

ساختار ggml_tensor * ggml_conv_2d_sk_p0(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

ساختار ggml_tensor * ggml_conv_2d_s1_ph(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    بازگشت ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

ساختار ggml_tensor * ggml_conv_2d_dw(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   d0,
        صحیح                   d1) {
    ساختار ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ساختار ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    ساختار ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ساختار ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    بازگشت result;
}

// ggml_conv_2d_dw_direct

ساختار ggml_tensor * ggml_conv_2d_dw_direct(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   stride0,
        صحیح                   stride1,
        صحیح                   pad0,
        صحیح                   pad1,
        صحیح                   dilation0,
        صحیح                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    اگر (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ثابت int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    بازگشت result;
}

// ggml_conv_2d_direct

ساختار ggml_tensor * ggml_conv_2d_direct(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        ساختار ggml_tensor  * b,   // input data [W, H, C, N]
        صحیح                   s0,  // stride dimension 0
        صحیح                   s1,  // stride dimension 1
        صحیح                   p0,  // padding dimension 0
        صحیح                   p1,  // padding dimension 1
        صحیح                   d0,  // dilation dimension 0
        صحیح                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_conv_3d_direct

ساختار ggml_tensor * ggml_conv_3d_direct(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   s0,
        صحیح                   s1,
        صحیح                   s2,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   d0,
        صحیح                   d1,
        صحیح                   d2,
        صحیح                   c,
        صحیح                   n,
        صحیح                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    بازگشت result;
}

// ggml_conv_transpose_2d_p0

ایستا int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, صحیح s, صحیح p) {
    بازگشت (ins - 1) * s - 2 * p + ks;
}

ساختار ggml_tensor * ggml_conv_transpose_2d_p0(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        صحیح                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ثابت int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    ساختار ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_pool_*

ایستا int64_t ggml_calc_pool_output_size(int64_t ins, صحیح ks, صحیح s, اعشاری p) {
    بازگشت (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

ساختار ggml_tensor * ggml_pool_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   s0,
        صحیح                   p0) {
    ثابت int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    بازگشت result;
}

// ggml_pool_2d

ساختار ggml_tensor * ggml_pool_2d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   k1,
        صحیح                   s0,
        صحیح                   s1,
        اعشاری                 p0,
        اعشاری                 p1) {
    ساختار ggml_tensor * result;
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
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_pool_2d_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * af,
        شمارش ggml_op_pool     op,
        صحیح                   k0,
        صحیح                   k1,
        صحیح                   s0,
        صحیح                   s1,
        اعشاری                 p0,
        اعشاری                 p1) {
    ساختار ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    بازگشت result;
}

// ggml_upscale / ggml_interpolate

ایستا ساختار ggml_tensor * ggml_interpolate_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_شمار);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_upscale(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   scale_factor,
        شمارش ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    بازگشت ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

ساختار ggml_tensor * ggml_upscale_ext(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   ne0,
        صحیح                   ne1,
        صحیح                   ne2,
        صحیح                   ne3,
        شمارش ggml_scale_mode  mode) {
    بازگشت ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

ساختار ggml_tensor * ggml_interpolate(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    بازگشت ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

ساختار ggml_tensor * ggml_pad(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   p3) {
    بازگشت ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

ساختار ggml_tensor * ggml_pad_circular(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1,
        صحیح                   p2,
        صحیح                   p3) {
    بازگشت ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

ساختار ggml_tensor * ggml_pad_ext(
            ساختار ggml_context * ctx,
            ساختار ggml_tensor  * a,
            صحیح                  lp0,
            صحیح                  rp0,
            صحیح                  lp1,
            صحیح                  rp1,
            صحیح                  lp2,
            صحیح                  rp2,
            صحیح                  lp3,
            صحیح                  rp3
            ) {
    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    بازگشت result;
}

// ggml_pad_ext_circular

ساختار ggml_tensor * ggml_pad_ext_circular(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                  lp0,
        صحیح                  rp0,
        صحیح                  lp1,
        صحیح                  rp1,
        صحیح                  lp2,
        صحیح                  rp2,
        صحیح                  lp3,
        صحیح                  rp3
        ) {
    ساختار ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    بازگشت result;
}

// ggml_pad_reflect_1d

ساختار ggml_tensor * ggml_pad_reflect_1d(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   p0,
        صحیح                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    بازگشت result;
}

// ggml_roll

ساختار ggml_tensor * ggml_roll(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   shift0,
        صحیح                   shift1,
        صحیح                   shift2,
        صحیح                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    بازگشت result;
}

// ggml_timestep_embedding

ساختار ggml_tensor * ggml_timestep_embedding(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * timesteps,
        صحیح                   dim,
        صحیح                   max_period) {

    ساختار ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    بازگشت result;
}

// ggml_tri

ساختار ggml_tensor * ggml_tri(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    شمارش ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    بازگشت result;
}

// ggml_fill

ایستا ساختار ggml_tensor * ggml_fill_impl(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    اعشاری                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_fill(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    اعشاری                 c) {
    بازگشت ggml_fill_impl(ctx, a, c, false);
}

ساختار ggml_tensor * ggml_fill_inplace(
    ساختار ggml_context * ctx,
    ساختار ggml_tensor  * a,
    اعشاری                 c) {
    بازگشت ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

ساختار ggml_tensor * ggml_argsort(
        ساختار ggml_context  * ctx,
        ساختار ggml_tensor   * a,
        شمارش ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    بازگشت result;
}

// ggml_argsort_top_k

ساختار ggml_tensor * ggml_argsort_بالاترین_k(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ساختار ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    بازگشت result;
}

// ggml_top_k

ساختار ggml_tensor * ggml_بالاترین_k(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_بالاترین_K;
    result->src[0] = a;

    بازگشت result;
}

// ggml_arange

ساختار ggml_tensor * ggml_arange(
        ساختار ggml_context * ctx,
        اعشاری                 start,
        اعشاری                 stop,
        اعشاری                 step) {
    GGML_ASSERT(stop > start);

    ثابت int64_t steps = (int64_t) ceilf((stop - start) / step);

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    بازگشت result;
}

// ggml_flash_attn_ext

ساختار ggml_tensor * ggml_flash_attn_ext(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * q,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * mask,
        اعشاری                 scale,
        اعشاری                 max_bias,
        اعشاری                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    اگر (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    اگر (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    اعشاری params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    بازگشت result;
}


تهی ggml_flash_attn_ext_set_prec(
        ساختار ggml_tensor * a,
        شمارش ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

شمارش ggml_prec ggml_flash_attn_ext_get_prec(
        ثابت ساختار ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ثابت int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    بازگشت (شمارش ggml_prec) prec_i32;
}

تهی ggml_flash_attn_ext_set_n_kv_max(
        ساختار ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

تهی ggml_flash_attn_ext_add_sinks(
        ساختار ggml_tensor * a,
        ساختار ggml_tensor * sinks) {
    اگر (!sinks) {
        a->src[4] = پوچ;
        بازگشت;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == پوچ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

ساختار ggml_tensor * ggml_flash_attn_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * q,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * d,
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

    شمارش ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ثابت size_t tsize = ggml_type_size(result_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ثابت size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ثابت size_t nelements = (end + tsize - 1)/tsize;

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, اندازه(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    بازگشت result;
}

// ggml_ssm_conv

ساختار ggml_tensor * ggml_ssm_conv(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * sx,
        ساختار ggml_tensor  * c) {
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

    ساختار ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    بازگشت result;
}

// ggml_ssm_scan

ساختار ggml_tensor * ggml_ssm_scan(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * s,
        ساختار ggml_tensor  * x,
        ساختار ggml_tensor  * dt,
        ساختار ggml_tensor  * A,
        ساختار ggml_tensor  * B,
        ساختار ggml_tensor  * C,
        ساختار ggml_tensor  * ids,
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

        اگر (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    بازگشت result;
}

// ggml_win_part

ساختار ggml_tensor * ggml_win_part(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ثابت صحیح px = (w - a->ne[1]%w)%w;
    ثابت صحیح py = (w - a->ne[2]%w)%w;

    ثابت صحیح npx = (px + a->ne[1])/w;
    ثابت صحیح npy = (py + a->ne[2])/w;
    ثابت صحیح np  = npx*npy;

    ثابت int64_t ne[4] = { a->ne[0], w, w, np, };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    بازگشت result;
}

// ggml_win_unpart

ساختار ggml_tensor * ggml_win_unpart(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   w0,
        صحیح                   h0,
        صحیح                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ثابت int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, اندازه(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    بازگشت result;
}

// ggml_get_rel_pos

ساختار ggml_tensor * ggml_get_rel_pos(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        صحیح                   qh,
        صحیح                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ثابت int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    بازگشت result;
}

// ggml_add_rel_pos

ایستا ساختار ggml_tensor * ggml_add_rel_pos_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * pw,
        ساختار ggml_tensor  * ph,
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

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    بازگشت result;
}

ساختار ggml_tensor * ggml_add_rel_pos(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * pw,
        ساختار ggml_tensor  * ph) {
    بازگشت ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

ساختار ggml_tensor * ggml_add_rel_pos_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * pw,
        ساختار ggml_tensor  * ph) {
    بازگشت ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

ساختار ggml_tensor * ggml_rwkv_wkv6(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * r,
        ساختار ggml_tensor  * tf,
        ساختار ggml_tensor  * td,
        ساختار ggml_tensor  * state) {
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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    بازگشت result;
}

// ggml_gated_linear_attn

ساختار ggml_tensor * ggml_gated_linear_attn(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * q,
        ساختار ggml_tensor  * g,
        ساختار ggml_tensor  * state,
        اعشاری scale) {
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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    بازگشت result;
}

// ggml_rwkv_wkv7

ساختار ggml_tensor * ggml_rwkv_wkv7(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * r,
        ساختار ggml_tensor  * w,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * state) {
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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    بازگشت result;
}

// ggml_unary

ایستا ساختار ggml_tensor * ggml_unary_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_unary(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_unary_op    op) {
    بازگشت ggml_unary_impl(ctx, a, op, false);
}

ساختار ggml_tensor * ggml_unary_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        شمارش ggml_unary_op    op) {
    بازگشت ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

ایستا ساختار ggml_tensor * ggml_map_custom1_impl(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساختار ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    بازگشت result;
}

ساختار ggml_tensor * ggml_map_custom1(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ساختار ggml_tensor * ggml_map_custom1_inplace(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ثابت  ggml_custom1_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

ایستا ساختار ggml_tensor * ggml_map_custom2_impl(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساختار ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

ساختار ggml_tensor * ggml_map_custom2(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ساختار ggml_tensor * ggml_map_custom2_inplace(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ثابت  ggml_custom2_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

ایستا ساختار ggml_tensor * ggml_map_custom3_impl(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ساختار ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    ساختار ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ساختار ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    بازگشت result;
}

ساختار ggml_tensor * ggml_map_custom3(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ساختار ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ساختار ggml_tensor * ggml_map_custom3_inplace(
        ساختار ggml_context      * ctx,
        ساختار ggml_tensor       * a,
        ساختار ggml_tensor       * b,
        ساختار ggml_tensor       * c,
        ثابت  ggml_custom3_op_t   fun,
        صحیح                        n_tasks,
        تهی                     * userdata) {
    بازگشت ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ساختار ggml_tensor * ggml_custom_4d(
        ساختار ggml_context * ctx,
        شمارش ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        ساختار ggml_tensor ** args,
        صحیح                   n_args,
        ggml_custom_op_t      fun,
        صحیح                   n_tasks,
        تهی                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    ساختار ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op = GGML_OP_CUSTOM;
    برای (صحیح i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    بازگشت result;
}

ساختار ggml_tensor * ggml_custom_inplace(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor ** args,
        صحیح                   n_args,
        ggml_custom_op_t      fun,
        صحیح                   n_tasks,
        تهی                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);

    ساختار ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, اندازه(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    برای (صحیح i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    بازگشت result;
}
// ggml_cross_entropy_loss

ساختار ggml_tensor * ggml_cross_entropy_loss(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_cross_entropy_loss_back

ساختار ggml_tensor * ggml_cross_entropy_loss_back(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        ساختار ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ساختار ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    بازگشت result;
}

// opt_step_adamw

ساختار ggml_tensor * ggml_opt_step_adamw(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * grad,
        ساختار ggml_tensor  * m,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    بازگشت result;
}

// opt_step_sgd

ساختار ggml_tensor * ggml_opt_step_sgd(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * grad,
        ساختار ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    ساختار ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    بازگشت result;
}

// solve_tri

ساختار ggml_tensor * ggml_solve_tri(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * a,
        ساختار ggml_tensor  * b,
        bool                  چپ,
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

    GGML_ASSERT(lower && چپ && !uni); // TODO: support other variants

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    بازگشت result;
}

// ggml_gated_delta_net

ساختار ggml_tensor * ggml_gated_delta_net(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * q,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * v,
        ساختار ggml_tensor  * g,
        ساختار ggml_tensor  * beta,
        ساختار ggml_tensor  * state,
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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    بازگشت result;
}

// ggml_lightning_indexer

ساختار ggml_tensor * ggml_lightning_indexer(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * q,
        ساختار ggml_tensor  * k,
        ساختار ggml_tensor  * weights,
        ساختار ggml_tensor  * mask) {

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
    ساختار ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    بازگشت result;
}

// ggml_dsv4_hc_comb

ساختار ggml_tensor * ggml_dsv4_hc_comb(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * mixes,
        ساختار ggml_tensor  * scale,
        ساختار ggml_tensor  * base,
        اعشاری                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ثابت int64_t hc_mix_dim = mixes->ne[0];
    ثابت int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    برای (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        اگر ((2 + i)*i == hc_mix_dim) {
            hc = i;
            بشکن;
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

    ساختار ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    بازگشت result;
}

// ggml_dsv4_hc_pre

ایستا ساختار ggml_tensor * ggml_dsv4_hc_pre_impl(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * x,
        ساختار ggml_tensor  * weights,
        اعشاری                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    اگر (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } وگرنه {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    ساختار ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    بازگشت result;
}

ساختار ggml_tensor * ggml_dsv4_hc_pre(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * x,
        ساختار ggml_tensor  * weights) {
    بازگشت ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

ساختار ggml_tensor * ggml_dsv4_hc_pre_gated(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * x,
        ساختار ggml_tensor  * gate,
        اعشاری                 scale) {
    بازگشت ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

ساختار ggml_tensor * ggml_dsv4_hc_post(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * x,
        ساختار ggml_tensor  * residual,
        ساختار ggml_tensor  * post,
        ساختار ggml_tensor  * comb) {
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

    اگر (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    ساختار ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    بازگشت result;
}

// ggml_top_k_mask

ساختار ggml_tensor * ggml_بالاترین_k_mask(
        ساختار ggml_context * ctx,
        ساختار ggml_tensor  * mask,
        ساختار ggml_tensor  * بالاترین_k,
        ساختار ggml_tensor  * score,
        ساختار ggml_tensor  * blocks,
        ساختار ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(بالاترین_k->type == GGML_TYPE_I32);
    GGML_ASSERT(بالاترین_k->ne[1] % mask->ne[1] == 0 && بالاترین_k->ne[2] % mask->ne[2] == 0 && بالاترین_k->ne[3] % mask->ne[3] == 0);

    اگر (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == بالاترین_k->ne[1] && score->ne[2] == بالاترین_k->ne[2] && score->ne[3] == بالاترین_k->ne[3]);
    }

    اگر (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    اگر (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == بالاترین_k->ne[1] && extra->ne[2] == بالاترین_k->ne[2] && extra->ne[3] == بالاترین_k->ne[3]);
    }

    ساختار ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], بالاترین_k->ne[1], بالاترین_k->ne[2], بالاترین_k->ne[3]);

    result->op     = GGML_OP_بالاترین_K_MASK;
    result->src[0] = mask;
    result->src[1] = بالاترین_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    بازگشت result;
}

////////////////////////////////////////////////////////////////////////////////

ساختار ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    ساختار ggml_hash_set result;
    result.size = size;
    result.کلیدها = GGML_MALLOC(اندازه(ساختار ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), اندازه(ggml_bitset_t));
    بازگشت result;
}

تهی ggml_hash_set_reset(ساختار ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, اندازه(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

تهی ggml_hash_set_free(ساختار ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->کلیدها);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    ایستا ثابت size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    ایستا ثابت size_t n_primes = اندازه(primes)/اندازه(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    وقتی (l < r) {
        size_t m = (l + r)/2;
        اگر (primes[m] < min_sz) {
            l = m + 1;
        } وگرنه {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    بازگشت sz;
}

ساختار hash_map {
    ساختار ggml_hash_set set;
    ساختار ggml_tensor ** vals;
};

ایستا ساختار hash_map * ggml_new_hash_map(size_t size) {
    ساختار hash_map * result = GGML_MALLOC(اندازه(ساختار hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, اندازه(ساختار ggml_tensor *));
    بازگشت result;
}

ایستا تهی ggml_hash_map_free(ساختار hash_map * map) {
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

ایستا تهی ggml_add_or_set(
        ساختار ggml_context * ctx,
        ساختار ggml_cgraph  * cgraph,
        size_t                isrc,
        ساختار ggml_tensor  * tensor) {
    ساختار ggml_tensor * src = cgraph->visited_hash_set.کلیدها[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } وگرنه {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ایستا تهی ggml_acc_or_set(
        ساختار ggml_context * ctx,
        ساختار ggml_cgraph  * cgraph,
        size_t                isrc,
        ساختار ggml_tensor  * tensor,
        ثابت  size_t         nb1,
        ثابت  size_t         nb2,
        ثابت  size_t         nb3,
        ثابت  size_t         offset) {
    ساختار ggml_tensor * src = cgraph->visited_hash_set.کلیدها[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } وگرنه {
        ساختار ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.کلیدها[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ایستا تهی ggml_add1_or_set(
        ساختار ggml_context * ctx,
        ساختار ggml_cgraph  * cgraph,
        size_t                isrc,
        ساختار ggml_tensor  * tensor) {
    ساختار ggml_tensor * src = cgraph->visited_hash_set.کلیدها[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } وگرنه {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ایستا تهی ggml_sub_or_set(
        ساختار ggml_context * ctx,
        ساختار ggml_cgraph  * cgraph,
        size_t                isrc,
        ساختار ggml_tensor  * tensor) {
    ساختار ggml_tensor * src = cgraph->visited_hash_set.کلیدها[isrc];
    GGML_ASSERT(src);
    اگر (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } وگرنه {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

ایستا تهی ggml_compute_backward(
        ساختار ggml_context * ctx, ساختار ggml_cgraph * cgraph, صحیح i, ثابت bool * grads_needed) {
    ساختار ggml_tensor * tensor = cgraph->nodes[i];
    ساختار ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    اگر (!grad) {
        بازگشت;
    }

    ساختار ggml_tensor * src0 = tensor->src[0];
    ساختار ggml_tensor * src1 = tensor->src[1];
    ساختار ggml_tensor * src2 = tensor->src[2];
    ساختار ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ثابت size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ثابت size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ثابت size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ثابت bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ثابت bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ثابت bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    انتخاب (tensor->op) {
        حالت GGML_OP_DUP: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } بشکن;
        حالت GGML_OP_ADD: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ساختار ggml_tensor * tmp = grad;
                اگر (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } بشکن;
        حالت GGML_OP_ADD1: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } بشکن;
        حالت GGML_OP_ACC: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ثابت size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ثابت size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ثابت size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ثابت size_t offset = ((int32_t *) tensor->op_params)[3];

                ساختار ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } بشکن;
        حالت GGML_OP_SUB: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            اگر (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } بشکن;
        حالت GGML_OP_MUL: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            اگر (src1_needs_grads) {
                ساختار ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                اگر (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } بشکن;
        حالت GGML_OP_DIV: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            اگر (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } بشکن;
        حالت GGML_OP_SQR: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } بشکن;
        حالت GGML_OP_SQRT: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } بشکن;
        حالت GGML_OP_LOG: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } بشکن;
        حالت GGML_OP_SIN: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } بشکن;
        حالت GGML_OP_COS: {
            اگر (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } بشکن;
        حالت GGML_OP_SUM: {
            اگر (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } بشکن;
        حالت GGML_OP_SUM_ROWS: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } بشکن;
        حالت GGML_OP_MEAN: {
            اگر (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } بشکن;
        حالت GGML_OP_REPEAT: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } بشکن;
        حالت GGML_OP_REPEAT_BACK: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } بشکن;
        حالت GGML_OP_RMS_NORM: {
            اگر (src0_needs_grads) {
                اعشاری eps;
                memcpy(&eps, tensor->op_params, اندازه(اعشاری));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } بشکن;
        حالت GGML_OP_MUL_MAT: {
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

            اگر (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                ساختار ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                اگر (!ggml_are_same_shape(tmp, src0)) {
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
            اگر (src1_needs_grads) {
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
        } بشکن;
        حالت GGML_OP_SCALE: {
            اگر (src0_needs_grads) {
                اعشاری s;
                memcpy(&s, tensor->op_params, اندازه(اعشاری));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } بشکن;
        حالت GGML_OP_SET: {
            ثابت size_t nb1    = ((ثابت int32_t *) tensor->op_params)[0];
            ثابت size_t nb2    = ((ثابت int32_t *) tensor->op_params)[1];
            ثابت size_t nb3    = ((ثابت int32_t *) tensor->op_params)[2];
            ثابت size_t offset = ((ثابت int32_t *) tensor->op_params)[3];

            ساختار ggml_tensor * tensor_grad_view = پوچ;

            اگر (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            اگر (src0_needs_grads) {
                ساختار ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            اگر (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } بشکن;
        حالت GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            اگر (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            اگر (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } بشکن;
        حالت GGML_OP_CONT: {
            // same as cpy
            اگر (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } بشکن;
        حالت GGML_OP_RESHAPE: {
            اگر (src0_needs_grads) {
                ساختار ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } بشکن;
        حالت GGML_OP_VIEW: {
            اگر (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, اندازه(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                اگر (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } بشکن;
        حالت GGML_OP_PERMUTE: {
            اگر (src0_needs_grads) {
                ثابت int32_t * axes = (ثابت int32_t *) tensor->op_params;
                ثابت صحیح axis0 = axes[0] & 0x3;
                ثابت صحیح axis1 = axes[1] & 0x3;
                ثابت صحیح axis2 = axes[2] & 0x3;
                ثابت صحیح axis3 = axes[3] & 0x3;
                صحیح axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } بشکن;
        حالت GGML_OP_TRANSPOSE: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } بشکن;
        حالت GGML_OP_GET_ROWS: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            اگر (src1_needs_grads) {
                // noop
            }
        } بشکن;
        حالت GGML_OP_DIAG_MASK_INF: {
            اگر (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ثابت صحیح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } بشکن;
        حالت GGML_OP_DIAG_MASK_ZERO: {
            اگر (src0_needs_grads) {
                ثابت صحیح n_past = ((ثابت int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } بشکن;
        حالت GGML_OP_SOFT_MAX: {
            اگر (src0_needs_grads) {
                اعشاری scale    = 1.0f;
                اعشاری max_bias = 0.0f;

                memcpy(&scale,    (ثابت اعشاری *) tensor->op_params + 0, اندازه(اعشاری));
                memcpy(&max_bias, (ثابت اعشاری *) tensor->op_params + 1, اندازه(اعشاری));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } بشکن;
        حالت GGML_OP_ROPE: {
            اگر (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ثابت صحیح n_dims     = ((ثابت int32_t *) tensor->op_params)[1];
                ثابت صحیح mode       = ((ثابت int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ثابت صحیح n_ctx_orig = ((ثابت int32_t *) tensor->op_params)[4];
                اعشاری freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                صحیح sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ثابت اعشاری *) tensor->op_params +  5, اندازه(اعشاری));
                memcpy(&freq_scale,  (ثابت اعشاری *) tensor->op_params +  6, اندازه(اعشاری));
                memcpy(&ext_factor,  (ثابت اعشاری *) tensor->op_params +  7, اندازه(اعشاری));
                memcpy(&attn_factor, (ثابت اعشاری *) tensor->op_params +  8, اندازه(اعشاری));
                memcpy(&beta_fast,   (ثابت اعشاری *) tensor->op_params +  9, اندازه(اعشاری));
                memcpy(&beta_slow,   (ثابت اعشاری *) tensor->op_params + 10, اندازه(اعشاری));
                memcpy(&sections,                    tensor->op_params + 11, اندازه(sections));

                ساختار ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } بشکن;
        حالت GGML_OP_IM2COL: {
            اگر (src1_needs_grads) {
                ثابت int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ثابت int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ثابت int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ثابت int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ثابت int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ثابت int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ثابت bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } بشکن;
        حالت GGML_OP_POOL_2D: {
            اگر (src0_needs_grads) {
                ثابت شمارش ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ثابت      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ثابت      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ثابت      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ثابت      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ثابت      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ثابت      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } بشکن;
        حالت GGML_OP_WIN_PART:
        حالت GGML_OP_WIN_UNPART:
        حالت GGML_OP_UNARY: {
            انتخاب (ggml_get_unary_op(tensor)) {
                حالت GGML_UNARY_OP_ABS: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } بشکن;
                حالت GGML_UNARY_OP_SGN: {
                    // noop
                } بشکن;
                حالت GGML_UNARY_OP_NEG: {
                    اگر (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } بشکن;
                حالت GGML_UNARY_OP_STEP: {
                    // noop
                } بشکن;
                حالت GGML_UNARY_OP_RELU: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } بشکن;
                حالت GGML_UNARY_OP_SILU: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } بشکن;
                حالت GGML_UNARY_OP_EXP: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } بشکن;
                حالت GGML_UNARY_OP_EXPM1: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } بشکن;
                حالت GGML_UNARY_OP_SOFTPLUS: {
                    اگر (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } بشکن;
                پیشفرض: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } بشکن;
        حالت GGML_OP_CROSS_ENTROPY_LOSS: {
            اگر (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } بشکن;
        حالت GGML_OP_GLU: {
            انتخاب (ggml_get_glu_op(tensor)) {
                حالت GGML_GLU_OP_SWIGLU: {
                    اگر (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    اگر (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } بشکن;
                پیشفرض: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } بشکن;
        حالت GGML_OP_NONE: {
            // noop
        } بشکن;
        حالت GGML_OP_شمار:
        پیشفرض: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

ایستا size_t ggml_visit_parents_graph(ساختار ggml_cgraph * cgraph, ساختار ggml_tensor * گره, bool compute) {
    اگر (گره->op != GGML_OP_NONE && compute) {
        گره->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ثابت size_t گره_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, گره);
    GGML_ASSERT(گره_hash_pos != GGML_HASHSET_FULL);

    اگر (ggml_bitset_get(cgraph->visited_hash_set.used, گره_hash_pos)) {
        // already visited

        اگر (compute) {
            // update the compute flag regardless
            برای (صحیح i = 0; i < GGML_MAX_SRC; ++i) {
                ساختار ggml_tensor * src = گره->src[i];
                اگر (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        بازگشت گره_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.کلیدها[گره_hash_pos] = گره;
    ggml_bitset_set(cgraph->visited_hash_set.used, گره_hash_pos);
    cgraph->use_counts[گره_hash_pos] = 0;

    برای (صحیح i = 0; i < GGML_MAX_SRC; ++i) {
        ثابت صحیح k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_چپ_TO_راست) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_راست_TO_چپ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        ساختار ggml_tensor * src = گره->src[k];
        اگر (src) {
            ثابت size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    اگر (گره->op == GGML_OP_NONE && !(گره->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        اگر (strlen(گره->name) == 0) {
            ggml_format_name(گره, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = گره;
        cgraph->n_leafs++;
    } وگرنه {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        اگر (strlen(گره->name) == 0) {
            ggml_format_name(گره, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = گره;
        cgraph->n_nodes++;
    }

    بازگشت گره_hash_pos;
}

ایستا تهی ggml_build_forward_impl(ساختار ggml_cgraph * cgraph, ساختار ggml_tensor * tensor, bool expand, bool compute) {
    اگر (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ثابت صحیح n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ثابت صحیح n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    اگر (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

ساختار ggml_tensor * ggml_build_forward_select(
        ساختار ggml_cgraph  * cgraph,
        ساختار ggml_tensor ** tensors,
        صحیح                   n_tensors,
        صحیح                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    برای (صحیح i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    بازگشت tensors[idx];
}

تهی ggml_build_forward_expand(ساختار ggml_cgraph * cgraph, ساختار ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

تهی ggml_build_forward_order(ساختار ggml_cgraph * cgraph, ساختار ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

تهی ggml_build_backward_expand(
        ساختار ggml_context *  ctx,
        ساختار ggml_cgraph  *  cgraph,
        ساختار ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ثابت صحیح n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*اندازه(ساختار ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*اندازه(ساختار ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, اندازه(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        برای (صحیح i = 0; i < n_nodes_f; ++i) {
            ساختار ggml_tensor * گره = cgraph->nodes[i];
            any_params = any_params || (گره->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (گره->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    برای (صحیح i = 0; i < n_nodes_f; ++i) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        اگر (گره->type == GGML_TYPE_I32) {
            ادامه;
        }

        bool گره_needs_grad = (گره->flags & GGML_TENSOR_FLAG_PARAM) || (گره->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        انتخاب (گره->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            حالت GGML_OP_IM2COL:      // only used for its shape
            حالت GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                بشکن;
            حالت GGML_OP_UNARY: {
                ثابت شمارش ggml_unary_op uop = ggml_get_unary_op(گره);
                // SGN and STEP unary ops are piecewise constant
                اگر (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } بشکن;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            حالت GGML_OP_CPY:           // gradients in CPY target are irrelevant
            حالت GGML_OP_GET_ROWS:      // row indices not differentiable
            حالت GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            حالت GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                بشکن;

            پیشفرض:
                بشکن;
        }
        برای (صحیح j = 0; j < GGML_MAX_SRC; ++j) {
            اگر (!گره->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, گره->src[j])]) {
                ادامه;
            }
            GGML_ASSERT(گره->src[j]->type == GGML_TYPE_F32 || گره->src[j]->type == GGML_TYPE_F16);
            گره_needs_grad = true;
            بشکن;
        }
        اگر (!گره_needs_grad) {
            ادامه;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!گره->view_src || گره->op == GGML_OP_CPY || گره->op == GGML_OP_SET_ROWS || گره->op == GGML_OP_VIEW ||
            گره->op == GGML_OP_RESHAPE || گره->op == GGML_OP_PERMUTE || گره->op == GGML_OP_TRANSPOSE);

        ثابت size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, گره);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        اگر (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } وگرنه اگر (گره->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, گره->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    برای (صحیح i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    آزاد(grads_needed);
}

ایستا تهی * incr_ptr_aligned(تهی ** p, size_t size, size_t align) {
    تهی * ptr = *p;
    ptr = (تهی *) GGML_PAD((uintptr_t) ptr, align);
    *p = (تهی *) ((نویسه *) ptr + size);
    بازگشت ptr;
}

ایستا size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    تهی * p = (نویسه *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, اندازه(ساختار ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * اندازه(int32_t), اندازه(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)); // hash keys
    اگر (grads) {
        incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * اندازه(ggml_bitset_t), اندازه(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    بازگشت nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    بازگشت GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(تهی) {
    بازگشت ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

ساختار ggml_cgraph * ggml_new_graph_custom(ساختار ggml_context * ctx, size_t size, bool grads) {
    ثابت size_t obj_size = ggml_graph_nbytes(size, grads);
    ساختار ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    ساختار ggml_cgraph * cgraph = (ساختار ggml_cgraph *) ((نویسه *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    تهی * p = cgraph + 1;

    ساختار ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *));
    ساختار ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * اندازه(int32_t), اندازه(int32_t));
    ساختار ggml_tensor ** hash_کلیدها_ptr  =         incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *));
    ساختار ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)) : پوچ;
    ساختار ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * اندازه(ساختار ggml_tensor *), اندازه(ساختار ggml_tensor *)) : پوچ;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * اندازه(ggml_bitset_t), اندازه(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    أكد(obj_size == (size_t)((نویسه *)p - (نویسه *)cgraph));

    *cgraph = (ساختار ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_کلیدها_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_چپ_TO_راست,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    اگر (grads) {
        memset(cgraph->grads,     0, hash_size*اندازه(ساختار ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*اندازه(ساختار ggml_tensor *));
    }

    بازگشت cgraph;
}

ساختار ggml_cgraph * ggml_new_graph(ساختار ggml_context * ctx) {
    بازگشت ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

ساختار ggml_cgraph ggml_graph_view(ساختار ggml_cgraph * cgraph0, صحیح i0, صحیح i1) {
    ساختار ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ پوچ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ پوچ,
        /*.leafs            =*/ پوچ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    بازگشت cgraph;
}

تهی ggml_graph_cpy(ساختار ggml_cgraph * src, ساختار ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    برای (صحیح i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    برای (صحیح i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    برای (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        اگر (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_درج(&dst->visited_hash_set, src->visited_hash_set.کلیدها[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    اگر (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*اندازه(ساختار ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*اندازه(ساختار ggml_tensor *));
    }
    اگر (src->grads) {
        GGML_ASSERT(dst->grads     != پوچ);
        GGML_ASSERT(dst->grad_accs != پوچ);
        برای (صحیح i = 0; i < src->n_nodes; ++i) {
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

ساختار ggml_cgraph * ggml_graph_dup(ساختار ggml_context * ctx, ساختار ggml_cgraph * cgraph, bool force_grads) {
    ساختار ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    بازگشت result;
}

ساختار ggml_tensor * ggml_set_zero(ساختار ggml_tensor * tensor) {
    اگر (ggml_is_empty(tensor)) {
        بازگشت tensor;
    }
    اگر (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } وگرنه {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    بازگشت tensor;
}

تهی ggml_graph_reset(ساختار ggml_cgraph * cgraph) {
    اگر (!cgraph) {
        بازگشت;
    }
    GGML_ASSERT(cgraph->grads != پوچ);

    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره     = cgraph->nodes[i];
        ساختار ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, گره);

        اگر (گره->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(گره->src[2]);
            ggml_set_zero(گره->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        اگر (grad_acc) {
            اگر (گره->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ثابت اعشاری onef = 1.0f;
                اگر (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, اندازه(اعشاری));
                } وگرنه {
                    GGML_ASSERT(grad_acc->data);
                    *((اعشاری *) grad_acc->data) = onef;
                }
            } وگرنه {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

تهی ggml_graph_clear(ساختار ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

صحیح ggml_graph_size(ساختار ggml_cgraph * cgraph) {
    بازگشت cgraph->size;
}

ساختار ggml_tensor * ggml_graph_گره(ساختار ggml_cgraph * cgraph, صحیح i) {
    اگر (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        بازگشت cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    بازگشت cgraph->nodes[i];
}

ساختار ggml_tensor ** ggml_graph_nodes(ساختار ggml_cgraph * cgraph) {
    بازگشت cgraph->nodes;
}

صحیح ggml_graph_n_nodes(ساختار ggml_cgraph * cgraph) {
    بازگشت cgraph->n_nodes;
}

تهی ggml_graph_add_گره(ساختار ggml_cgraph * cgraph, ساختار ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

ساختار ggml_tensor * ggml_graph_get_tensor(ثابت ساختار ggml_cgraph * cgraph, ثابت نویسه * name) {
    برای (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ساختار ggml_tensor * leaf = cgraph->leafs[i];

        اگر (strcmp(leaf->name, name) == 0) {
            بازگشت leaf;
        }
    }

    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        اگر (strcmp(گره->name, name) == 0) {
            بازگشت گره;
        }
    }

    بازگشت پوچ;
}

ساختار ggml_tensor * ggml_graph_get_grad(ثابت ساختار ggml_cgraph * cgraph, ثابت ساختار ggml_tensor * گره) {
    ثابت size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, گره);
    بازگشت igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : پوچ;
}

ساختار ggml_tensor * ggml_graph_get_grad_acc(ثابت ساختار ggml_cgraph * cgraph, ثابت ساختار ggml_tensor * گره) {
    ثابت size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, گره);
    بازگشت igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : پوچ;
}

تهی ggml_graph_print(ثابت ساختار ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                گره->ne[0], گره->ne[1], گره->ne[2],
                ggml_op_name(گره->op), (گره->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, گره) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    برای (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ساختار ggml_tensor * گره = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                گره->ne[0], گره->ne[1],
                ggml_op_name(گره->op),
                ggml_get_name(گره));
    }

    GGML_LOG_INFO("========================================\n");
}

ایستا صحیح ggml_گره_list_find_tensor(ثابت ساختار ggml_cgraph * cgraph,
                                      ثابت صحیح *                idxs,
                                      صحیح                        شمار,
                                      ثابت ساختار ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    برای (صحیح i = 0; i < شمار; ++i) {
        ثابت صحیح گره_idx = idxs[i];

        اگر (گره_idx >= cgraph->n_nodes) {
            بازگشت -1;
        }
        اگر (cgraph->nodes[گره_idx] == tensor) {
            بازگشت i;
        }
    }
    بازگشت -1;
}

ایستا bool ggml_is_constant(ثابت ساختار ggml_tensor * tensor) {
    بازگشت tensor->buffer != پوچ && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ثابت ساختار ggml_cgraph * cgraph,
                                ثابت صحیح *                گره_idxs,
                                صحیح                        شمار,
                                ثابت شمارش ggml_op *       ops,
                                ثابت صحیح *                outputs,
                                صحیح                        رقم_outputs) {
    GGML_ASSERT(outputs && رقم_outputs > 0);

    برای (صحیح i = 0; i < شمار; ++i) {
        اگر (گره_idxs[i] >= cgraph->n_nodes) {
            بازگشت false;
        }

        ثابت ساختار ggml_tensor * گره = cgraph->nodes[گره_idxs[i]];

        اگر (گره->op != ops[i]) {
            بازگشت false;
        }

        اگر ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            بازگشت false;
        }

        اگر (ggml_گره_list_find_tensor(cgraph, outputs, رقم_outputs, گره) != -1) {
            ادامه;
        }

        اگر (گره->flags & GGML_TENSOR_FLAG_OUTPUT) {
            بازگشت false;
        }

        صحیح subgraph_uses = 0;
        برای (صحیح j = i + 1; j < شمار; ++j) {
            ثابت ساختار ggml_tensor * other_گره = cgraph->nodes[گره_idxs[j]];
            برای (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                اگر (other_گره->src[src_idx] == گره) {
                    subgraph_uses++;
                }
            }
        }

        اگر (subgraph_uses != ggml_گره_get_use_شمار(cgraph, گره_idxs[i])) {
            بازگشت false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        ساختار ggml_tensor * view_src = گره->view_src;
        وقتی (view_src) {
            اگر (ggml_گره_list_find_tensor(cgraph, گره_idxs, شمار, view_src) == -1 && !ggml_is_constant(view_src)) {
                بازگشت false;
            }
            view_src = view_src->view_src;
        }
    }

    بازگشت true;
}

// check if node is part of the graph
ایستا bool ggml_graph_find(ثابت ساختار ggml_cgraph * cgraph, ثابت ساختار ggml_tensor * گره) {
    اگر (cgraph == پوچ) {
        بازگشت true;
    }

    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        اگر (cgraph->nodes[i] == گره) {
            بازگشت true;
        }
    }

    بازگشت false;
}

ایستا ساختار ggml_tensor * ggml_graph_get_parent(ثابت ساختار ggml_cgraph * cgraph, ثابت ساختار ggml_tensor * گره) {
    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * parent = cgraph->nodes[i];
        ساختار ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        اگر (grad == گره) {
            بازگشت parent;
        }
    }

    بازگشت پوچ;
}

ایستا تهی ggml_graph_dump_dot_گره_edge(FILE * fp, ثابت ساختار ggml_cgraph * gb, ساختار ggml_tensor * گره, ساختار ggml_tensor * parent, ثابت نویسه * label)  {
    ساختار ggml_tensor * gparent = ggml_graph_get_parent(gb, گره);
    ساختار ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (تهی *) gparent0 : (تهی *) parent,
            gparent ? (تهی *) gparent : (تهی *) گره,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

ایستا تهی ggml_graph_dump_dot_leaf_edge(FILE * fp, ساختار ggml_tensor * گره, ساختار ggml_tensor * parent, ثابت نویسه * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (تهی *) parent,
            (تهی *) گره,
            label);
}

تهی ggml_graph_dump_dot(ثابت ساختار ggml_cgraph * gb, ثابت ساختار ggml_cgraph * cgraph, ثابت نویسه * filename) {
    نویسه color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    برای (صحیح i = 0; i < gb->n_nodes; i++) {
        ساختار ggml_tensor * گره = gb->nodes[i];
        ساختار ggml_tensor * grad = ggml_graph_get_grad(gb, گره);

        اگر (ggml_graph_get_parent(gb, گره) != پوچ) {
            ادامه;
        }

        اگر (گره->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, اندازه(color), "yellow");
        } وگرنه اگر (grad) {
            اگر (ggml_graph_find(cgraph, گره)) {
                snprintf(color, اندازه(color), "green");
            } وگرنه {
                snprintf(color, اندازه(color), "lightblue");
            }
        } وگرنه {
            snprintf(color, اندازه(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (تهی *) گره, color);

        اگر (strlen(گره->name) > 0) {
            fprintf(fp, "%s (%s)|", گره->name, ggml_type_name(گره->type));
        } وگرنه {
            fprintf(fp, "(%s)|", ggml_type_name(گره->type));
        }

        اگر (ggml_is_matrix(گره)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, گره->ne[0], گره->ne[1], ggml_op_نماد(گره->op));
        } وگرنه {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, گره->ne[0], گره->ne[1], گره->ne[2], ggml_op_نماد(گره->op));
        }

        اگر (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_نماد(grad->op));
        } وگرنه {
            fprintf(fp, "\"; ]\n");
        }
    }

    برای (صحیح i = 0; i < gb->n_leafs; i++) {
        ساختار ggml_tensor * گره = gb->leafs[i];

        snprintf(color, اندازه(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (تهی *) گره, color);

        اگر (strlen(گره->name) > 0) {
            fprintf(fp, "%s (%s)|", گره->name, ggml_type_name(گره->type));
        } وگرنه {
            fprintf(fp, "(%s)|", ggml_type_name(گره->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, گره->ne[0], گره->ne[1]);
        اگر (ggml_nelements(گره) < 5 && گره->data != پوچ) {
            fprintf(fp, " | (");
            برای (صحیح j = 0; j < ggml_nelements(گره); j++) {
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
                اگر (j < ggml_nelements(گره) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    برای (صحیح i = 0; i < gb->n_nodes; i++) {
        ساختار ggml_tensor * گره = gb->nodes[i];

        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گره->src[j]) {
                نویسه label[16];
                snprintf(label, اندازه(label), "src %d", j);
                ggml_graph_dump_dot_گره_edge(fp, gb, گره, گره->src[j], label);
            }
        }
    }

    برای (صحیح i = 0; i < gb->n_leafs; i++) {
        ساختار ggml_tensor * گره = gb->leafs[i];

        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گره->src[j]) {
                نویسه label[16];
                snprintf(label, اندازه(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, گره, گره->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

تهی ggml_set_درون_داد(ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_درون_داد;
}

تهی ggml_set_output(ساختار ggml_tensor * tensor) {
    برای (ساختار ggml_tensor * cur = tensor; cur != پوچ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

تهی ggml_set_param(ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

تهی ggml_set_loss(ساختار ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

تهی ggml_quantize_init(شمارش ggml_type type) {
    ggml_critical_section_start();

    انتخاب (type) {
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); بشکن;
        حالت GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); بشکن;
        حالت GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); بشکن;
        پیشفرض: // nothing
            بشکن;
    }

    ggml_critical_section_end();
}

تهی ggml_quantize_free(تهی) {
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

bool ggml_quantize_requires_imatrix(شمارش ggml_type type) {
    بازگشت
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        شمارش ggml_type   type,
           ثابت اعشاری * src,
                  تهی * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ثابت اعشاری * imatrix) {
    ثابت int64_t n = nrows * n_per_row;

    اگر (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != پوچ);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ثابت size_t start_row = start / n_per_row;
    ثابت size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (نویسه *) dst + start_row * row_size, nrows, n_per_row, imatrix); بشکن;
        حالت GGML_TYPE_F16:
            {
                size_t elemsize = اندازه(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                size_t elemsize = اندازه(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                size_t elemsize = اندازه(اعشاری);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } بشکن;
        پیشفرض:
            أكد(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    بازگشت result;
}

////////////////////////////////////////////////////////////////////////////////

تهی ggml_log_get(ggml_log_callback * log_callback, تهی ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

تهی ggml_log_set(ggml_log_callback log_callback, تهی * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

تهی ggml_threadpool_params_init(ساختار ggml_threadpool_params * p, صحیح n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

ساختار ggml_threadpool_params ggml_threadpool_params_default(صحیح n_threads) {
    ساختار ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    بازگشت p;
}

bool ggml_threadpool_params_match(ثابت ساختار ggml_threadpool_params * p0, ثابت ساختار ggml_threadpool_params * p1) {
    اگر (p0->n_threads  != p1->n_threads  ) بازگشت false;
    اگر (p0->prio       != p1->prio       ) بازگشت false;
    اگر (p0->poll       != p1->poll       ) بازگشت false;
    اگر (p0->strict_cpu != p1->strict_cpu ) بازگشت false;
    بازگشت memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
