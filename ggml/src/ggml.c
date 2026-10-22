#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#अन्तर्भाव "ggml-version.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-threading.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml.h"

// FIXME: required here for quantization functions
#अन्तर्भाव "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#अन्तर्भाव <hbwmalloc.h>
#endif

#यदि defined(_MSC_VER) || defined(__MINGW32__)
#अन्तर्भाव <malloc.h> // using malloc.h with MSC/MINGW
#नोचेद्यदि !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#अन्तर्भाव <alloca.h>
#endif

#अन्तर्भाव <assert.h>
#अन्तर्भाव <errno.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <inttypes.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <float.h>
#अन्तर्भाव <limits.h>
#अन्तर्भाव <stdarg.h>
#अन्तर्भाव <signal.h>
#यदि defined(__gnu_linux__)
#अन्तर्भाव <syscall.h>
#endif

#यदि defined(__APPLE__)
#अन्तर्भाव <unistd.h>
#अन्तर्भाव <mach/mach.h>
#अन्तर्भाव <TargetConditionals.h>
#endif

#यदि defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#अन्तर्भाव <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(शून्य) {
#ifdef _MSC_VER
#यदि defined(_WIN32)
    स्थिर चञ्चल LONG counter = 1;
    प्रत्यागम (uint64_t) InterlockedIncrement(&counter) - 1;
#अन्यथा
    स्थिर चञ्चल long long counter = 1;
    प्रत्यागम (uint64_t) _InterlockedIncrement64(&counter) - 1;
#endif
#अन्यथा
    स्थिर uint64_t counter = 1;
    प्रत्यागम __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#यदि defined(__AVX512BF16__)
#यदि defined(_MSC_VER)
#define m512i(p) p
#अन्यथा
#अन्तर्भाव <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#यदि defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#अन्तर्भाव <unistd.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/wait.h>
#यदि defined(__linux__)
#अन्तर्भाव <sys/prctl.h>
#endif

#यदि defined(__ANDROID__)
#अन्तर्भाव <unwind.h>
#अन्तर्भाव <dlfcn.h>
#अन्तर्भाव <stdio.h>

संरचना backtrace_state {
    शून्य ** current;
    शून्य ** end;
};

स्थिर _Unwind_Reason_Code unwind_callback(संरचना _Unwind_Context* context, शून्य* arg) {
    संरचना backtrace_state * state = (संरचना backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    यदि (pc) {
        यदि (state->current == state->end) {
            प्रत्यागम URC_END_OF_चळत;
        } अन्यथा {
            *state->current++ = (शून्य*)pc;
        }
    }
    प्रत्यागम _URC_NO_REASON;
}

स्थिर शून्य ggml_print_backtrace_symbols(शून्य) {
    ध्रुव पूर्णाङ्क max = 100;
    शून्य* buffer[max];

    संरचना backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    पूर्णाङ्क मोजणी = state.current - buffer;

    प्रति (पूर्णाङ्क idx = 0; idx < मोजणी; ++idx) {
        ध्रुव शून्य * addr = buffer[idx];
        ध्रुव वर्ण * चिन्ह = "";

        Dl_info info;
        यदि (dladdr(addr, &info) && info.dli_sname) {
            चिन्ह = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, चिन्ह);
    }
}
#नोचेद्यदि defined(__linux__) && defined(__GLIBC__)
#अन्तर्भाव <execinfo.h>
स्थिर शून्य ggml_print_backtrace_symbols(शून्य) {
    शून्य * trace[100];
    पूर्णाङ्क nptrs = backtrace(trace, परिमाण(trace)/परिमाण(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#नोचेद्यदि defined(__APPLE__)
#अन्तर्भाव <execinfo.h>
स्थिर शून्य ggml_print_backtrace_symbols(शून्य) {
    शून्य * trace[100];
    पूर्णाङ्क nptrs = backtrace(trace, परिमाण(trace)/परिमाण(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#अन्यथा
स्थिर शून्य ggml_print_backtrace_symbols(शून्य) {
    // platform not supported
}
#endif

शून्य ggml_print_backtrace(शून्य) {
    ध्रुव वर्ण * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    यदि (GGML_NO_BACKTRACE) {
        प्रत्यागम;
    }
#यदि defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    ध्रुव वर्ण * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    यदि (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        प्रत्यागम;
    }
#endif
#यदि defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t size = 0;
    वर्ण * ओळ = रिक्त;
    ssize_t length = 0;
    यावत् ((length = getline(&ओळ, &size, f)) > 0) {
        यदि (!strncmp(ओळ, "TracerPid:", परिमाण("TracerPid:") - 1) &&
            (length != परिमाण("TracerPid:\t0\n") - 1 || ओळ[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            मुक्त(ओळ);
            fclose(f);
            प्रत्यागम;
        }
    }
    मुक्त(ओळ);
    fclose(f);
    पूर्णाङ्क lock[2] = { -1, -1 };
    (शून्य) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#endif
    ध्रुव पूर्णाङ्क parent_pid = getpid();
    ध्रुव पूर्णाङ्क child_pid = fork();
    यदि (child_pid < 0) { // error
#यदि defined(__linux__)
        close(lock[1]);
        close(lock[0]);
#endif
        प्रत्यागम;
    } अन्यथा यदि (child_pid == 0) { // child
        वर्ण attach[32];
        snprintf(attach, परिमाण(attach), "attach %d", parent_pid);
#यदि defined(__linux__)
        close(lock[1]);
        (शून्य) !पठ(lock[0], lock, 1);
        close(lock[0]);
#endif
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (वर्ण *) रिक्त);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[परिमाण("attach ") - 1],
            (वर्ण *) रिक्त);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } अन्यथा { // parent
#यदि defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        close(lock[1]);
        close(lock[0]);
#endif
        waitpid(child_pid, रिक्त, 0);
    }
}
#अन्यथा
शून्य ggml_print_backtrace(शून्य) {
    // platform not supported
}
#endif

स्थिर ggml_abort_callback_t g_abort_callback = रिक्त;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    प्रत्यागम ret_val;
}

शून्य ggml_abort(ध्रुव वर्ण * file, पूर्णाङ्क ओळ, ध्रुव वर्ण * fmt, ...) {
    fflush(stdout);

    वर्ण message[2048];
    पूर्णाङ्क offset = snprintf(message, परिमाण(message), "%s:%d: ", file, ओळ);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, परिमाण(message) - offset, fmt, args);
    va_end(args);

    यदि (g_abort_callback) {
        g_abort_callback(message);
    } अन्यथा {
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

संरचना ggml_logger_state {
    ggml_log_callback log_callback;
    शून्य * log_callback_user_data;
};
स्थिर संरचना ggml_logger_state g_logger_state = {ggml_log_callback_default, रिक्त};

स्थिर शून्य ggml_log_internal_v(गणना ggml_log_level level, ध्रुव वर्ण * format, va_list args) {
    यदि (format == रिक्त) {
        प्रत्यागम;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    वर्ण buffer[128];
    पूर्णाङ्क len = vsnprintf(buffer, 128, format, args);
    यदि (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } अन्यथा {
        वर्ण * buffer2 = (वर्ण *) calloc(len + 1, परिमाण(वर्ण));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        मुक्त(buffer2);
    }
    va_end(args_copy);
}

शून्य ggml_log_internal(गणना ggml_log_level level, ध्रुव वर्ण * format, ...) {
    va_list args;
    va_start(args, format);
    ggml_log_internal_v(level, format, args);
    va_end(args);
}

शून्य ggml_log_callback_default(गणना ggml_log_level level, ध्रुव वर्ण * text, शून्य * user_data) {
    (शून्य) level;
    (शून्य) user_data;
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
#यदि defined(__has_feature)
#यदि __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#अन्यथा
#यदि defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif

शून्य * ggml_aligned_malloc(size_t size) {
#यदि defined(__s390x__)
    ध्रुव पूर्णाङ्क alignment = 256;
#अन्यथा
    ध्रुव पूर्णाङ्क alignment = 64;
#endif

#यदि defined(_MSC_VER) || defined(__MINGW32__)
    प्रत्यागम _aligned_malloc(size, alignment);
#अन्यथा
    यदि (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * aligned_memory = रिक्त;
  #ifdef GGML_USE_CPU_HBM
    पूर्णाङ्क result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #नोचेद्यदि TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    पूर्णाङ्क result = EFAULT;
    विकल्प (alloc_status) {
        पक्ष KERN_SUCCESS:
            result = 0;
            भङ्ग;
        पक्ष KERN_INVALID_ADDRESS:
            result = EINVAL;
            भङ्ग;
        पक्ष KERN_NO_SPACE:
            result = ENOMEM;
            भङ्ग;
        स्वभाव:
            result = EFAULT;
            भङ्ग;
    }
  #अन्यथा
    पूर्णाङ्क result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    यदि (result != 0) {
        // Handle allocation failure
        ध्रुव वर्ण *error_desc = "unknown allocation error";
        विकल्प (result) {
            पक्ष EINVAL:
                error_desc = "invalid alignment value";
                भङ्ग;
            पक्ष ENOMEM:
                error_desc = "insufficient memory";
                भङ्ग;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        प्रत्यागम रिक्त;
    }
    प्रत्यागम aligned_memory;
#endif
}

शून्य ggml_aligned_free(शून्य * ptr, size_t size) {
    GGML_UNUSED(size);
#यदि defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#नोचेद्यदि GGML_USE_CPU_HBM
    यदि (ptr != रिक्त) {
        hbw_free(ptr);
    }
#नोचेद्यदि TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    यदि (ptr != रिक्त) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#अन्यथा
    मुक्त(ptr);
#endif
}


अन्तःस्थ स्थिर शून्य * ggml_malloc(size_t size) {
    यदि (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * result = malloc(size);
    यदि (result == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    प्रत्यागम result;
}

// calloc
अन्तःस्थ स्थिर शून्य * ggml_calloc(size_t अंक, size_t size) {
    यदि (अंक == 0 || size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * result = calloc(अंक, size);
    यदि (result == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    प्रत्यागम result;
}

#define GGML_MALLOC(size)      ggml_malloc(size)
#define GGML_CALLOC(अंक, size) ggml_calloc(अंक, size)

#define GGML_FREE(ptr) मुक्त(ptr)

ध्रुव वर्ण * ggml_status_to_string(गणना ggml_status status) {
    विकल्प (status) {
        पक्ष GGML_STATUS_ALLOC_FAILED: प्रत्यागम "GGML status: error (failed to allocate memory)";
        पक्ष GGML_STATUS_FAILED:       प्रत्यागम "GGML status: error (operation failed)";
        पक्ष GGML_STATUS_SUCCESS:      प्रत्यागम "GGML status: success";
        पक्ष GGML_STATUS_ABORTED:      प्रत्यागम "GGML status: warning (operation aborted)";
    }

    प्रत्यागम "GGML status: unknown";
}

भिन्न ggml_fp16_to_fp32(ggml_fp16_t x) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    प्रत्यागम GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(भिन्न x) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    प्रत्यागम GGML_FP32_TO_FP16(x);
}

भिन्न ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    प्रत्यागम GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(भिन्न x) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    प्रत्यागम GGML_FP32_TO_BF16(x);
}

शून्य ggml_fp16_to_fp32_row(ध्रुव ggml_fp16_t * x, भिन्न * y, int64_t n) {
    प्रति (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

शून्य ggml_fp32_to_fp16_row(ध्रुव भिन्न * x, ggml_fp16_t * y, int64_t n) {
    पूर्णाङ्क i = 0;
    प्रति (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

शून्य ggml_bf16_to_fp32_row(ध्रुव ggml_bf16_t * x, भिन्न * y, int64_t n) {
    पूर्णाङ्क i = 0;
    प्रति (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

शून्य ggml_fp32_to_bf16_row_ref(ध्रुव भिन्न * x, ggml_bf16_t * y, int64_t n) {
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

शून्य ggml_fp32_to_bf16_row(ध्रुव भिन्न * x, ggml_bf16_t * y, int64_t n) {
  पूर्णाङ्क i = 0;
#यदि defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  प्रति (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    प्रति (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    प्रत्यागम memcmp(guid_a, guid_b, परिमाण(ggml_guid)) == 0;
}

ध्रुव वर्ण * ggml_version(शून्य) {
    प्रत्यागम GGML_VERSION;
}

ध्रुव वर्ण * ggml_commit(शून्य) {
    प्रत्यागम GGML_COMMIT;
}

//
// timing
//

#यदि defined(_MSC_VER) || defined(__MINGW32__)
स्थिर int64_t timer_freq, timer_start;
स्थिर BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
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

    प्रत्यागम TRUE;
}
शून्य ggml_time_init(शून्य) {
    स्थिर INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, रिक्त, रिक्त);
}
int64_t ggml_time_ms(शून्य) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    प्रत्यागम ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(शून्य) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    प्रत्यागम ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#अन्यथा
शून्य ggml_time_init(शून्य) {}
int64_t ggml_time_ms(शून्य) {
    संरचना timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    प्रत्यागम (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(शून्य) {
    संरचना timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    प्रत्यागम (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(शून्य) {
    प्रत्यागम clock();
}

int64_t ggml_cycles_per_ms(शून्य) {
    प्रत्यागम CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
स्थिर wchar_t * ggml_mbstowcs(ध्रुव वर्ण * mbs) {
    पूर्णाङ्क wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, रिक्त, 0);
    यदि (!wlen) {
        errno = EINVAL;
        प्रत्यागम रिक्त;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * परिमाण(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    यदि (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        प्रत्यागम रिक्त;
    }

    प्रत्यागम wbuf;
}
#endif

FILE * ggml_fopen(ध्रुव वर्ण * fname, ध्रुव वर्ण * mode) {
#ifdef _WIN32
    FILE * file = रिक्त;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    यदि (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(mode);
        यदि (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    प्रत्यागम file;
#अन्यथा
    प्रत्यागम fopen(fname, mode);
#endif

}

स्थिर ध्रुव संरचना ggml_type_traits type_traits[GGML_TYPE_मोजणी] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = परिमाण(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = परिमाण(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = परिमाण(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = परिमाण(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = परिमाण(द्विगुण),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = परिमाण(भिन्न),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = परिमाण(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_size                = QK1_0,
        .type_size                = परिमाण(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_size                = QK2_0,
        .type_size                = परिमाण(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = परिमाण(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = परिमाण(block_q4_1),
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
        .type_size                = परिमाण(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = परिमाण(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = परिमाण(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = परिमाण(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_size                = QK_MXFP4,
        .type_size                = परिमाण(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,
        .type_size                = परिमाण(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = परिमाण(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = परिमाण(ggml_bf16_t),
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
        .type_size                = परिमाण(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = परिमाण(block_tq2_0),
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

ध्रुव संरचना ggml_type_traits * ggml_get_type_traits(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम &type_traits[type];
}

//
// ggml object
//

संरचना ggml_object {
    size_t offs;
    size_t size;

    संरचना ggml_object * next;

    गणना ggml_object_type type;

    वर्ण padding[4];
};

स्थिर ध्रुव size_t GGML_OBJECT_SIZE = परिमाण(संरचना ggml_object);

//
// ggml context
//

संरचना ggml_context {
    size_t mem_size;
    शून्य * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    पूर्णाङ्क    n_objects;

    संरचना ggml_object * objects_begin;
    संरचना ggml_object * objects_end;
};

//
// data types
//

स्थिर ध्रुव वर्ण * GGML_OP_NAME[GGML_OP_मोजणी] = {
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

static_assert(GGML_OP_मोजणी == 102, "GGML_OP_COUNT != 102");

स्थिर ध्रुव वर्ण * GGML_OP_चिन्ह[GGML_OP_मोजणी] = {
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

static_assert(GGML_OP_मोजणी == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_मोजणी == 2, "GGML_OP_POOL_COUNT != 2");

स्थिर ध्रुव वर्ण * GGML_UNARY_OP_NAME[GGML_UNARY_OP_मोजणी] = {
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

static_assert(GGML_UNARY_OP_मोजणी == 22, "GGML_UNARY_OP_COUNT != 22");

स्थिर ध्रुव वर्ण * GGML_GLU_OP_NAME[GGML_GLU_OP_मोजणी] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_मोजणी == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(परिमाण(संरचना ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(परिमाण(संरचना ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

शून्य ggml_print_object(ध्रुव संरचना ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (ध्रुव शून्य *) obj->next);
}

शून्य ggml_print_objects(ध्रुव संरचना ggml_context * ctx) {
    संरचना ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ध्रुव शून्य *) ctx);

    यावत् (obj != रिक्त) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
        यदि (tensor->ne[i] <= 0) {
            प्रत्यागम 0;
        }
    }

    size_t nbytes;
    ध्रुव size_t blck_size = ggml_blck_size(tensor->type);
    यदि (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    अन्यथा {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    प्रत्यागम nbytes;
}

size_t ggml_nbytes_pad(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_size(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम type_traits[type].blck_size;
}

size_t ggml_type_size(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम type_traits[type].type_size;
}

size_t ggml_row_size(गणना ggml_type type, int64_t ne) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रतिज्ञा(ne % ggml_blck_size(type) == 0);
    प्रत्यागम ggml_type_size(type)*ne/ggml_blck_size(type);
}

द्विगुण ggml_type_sizef(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम ((द्विगुण)(type_traits[type].type_size))/type_traits[type].blck_size;
}

ध्रुव वर्ण * ggml_type_name(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम type_traits[type].type_name;
}

bool ggml_is_quantized(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_मोजणी);
    प्रत्यागम type_traits[type].is_quantized;
}

ध्रुव वर्ण * ggml_op_name(गणना ggml_op op) {
    प्रत्यागम GGML_OP_NAME[op];
}

ध्रुव वर्ण * ggml_op_चिन्ह(गणना ggml_op op) {
    प्रत्यागम GGML_OP_चिन्ह[op];
}

ध्रुव वर्ण * ggml_unary_op_name(गणना ggml_unary_op op) {
    प्रत्यागम GGML_UNARY_OP_NAME[op];
}

ध्रुव वर्ण * ggml_glu_op_name(गणना ggml_glu_op op) {
    प्रत्यागम GGML_GLU_OP_NAME[op];
}

ध्रुव वर्ण * ggml_op_desc(ध्रुव संरचना ggml_tensor * t) {
    यदि (t->op == GGML_OP_UNARY) {
        गणना ggml_unary_op uop = ggml_get_unary_op(t);
        प्रत्यागम ggml_unary_op_name(uop);
    }
    यदि (t->op == GGML_OP_GLU) {
        गणना ggml_glu_op gop = ggml_get_glu_op(t);
        प्रत्यागम ggml_glu_op_name(gop);
    }
    प्रत्यागम ggml_op_name(t->op);
}

size_t ggml_element_size(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_type_size(tensor->type);
}

bool ggml_is_scalar(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->ne[3] == 1;
}

पूर्णाङ्क ggml_n_dims(ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        यदि (tensor->ne[i] > 1) {
            प्रत्यागम i + 1;
        }
    }
    प्रत्यागम 1;
}

गणना ggml_type ggml_ftype_to_ggml_type(गणना ggml_ftype ftype) {
    गणना ggml_type wtype = GGML_TYPE_मोजणी;

    विकल्प (ftype) {
        पक्ष GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    भङ्ग;
        पक्ष GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_मोजणी; भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_मोजणी; भङ्ग;
    }

    GGML_ASSERT(wtype != GGML_TYPE_मोजणी);

    प्रत्यागम wtype;
}

size_t ggml_tensor_overhead(शून्य) {
    प्रत्यागम GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

bool ggml_is_transposed(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->nb[0] > tensor->nb[1];
}

स्थिर bool ggml_is_contiguous_m_n(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क m, पूर्णाङ्क n) {
    size_t next_nb = ggml_type_size(tensor->type);
    यदि (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        प्रत्यागम false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    प्रति (पूर्णाङ्क i = 1; i < n; i++) {
        यदि (i > m) {
            यदि (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                प्रत्यागम false;
            }
            next_nb *= tensor->ne[i];
        } अन्यथा {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    प्रत्यागम true;
}

bool ggml_is_contiguous(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

bool ggml_is_permuted(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_size(tensor->type);
}

bool ggml_is_contiguous_rows(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम
        tensor->ne[0] == ggml_blck_size(tensor->type) ||
        tensor->nb[0] == ggml_type_size(tensor->type);
}

स्थिर अन्तःस्थ bool ggml_is_padded_1d(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(ध्रुव संरचना ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
        यदि (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            प्रत्यागम true;
        }
    }
    प्रत्यागम false;
}

bool ggml_are_same_shape(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

स्थिर अन्तःस्थ bool ggml_can_repeat_rows(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

संरचना ggml_context * ggml_init(संरचना ggml_init_params params) {
    स्थिर bool is_first_call = true;

    ggml_critical_section_start();

    यदि (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    संरचना ggml_context * ctx = GGML_MALLOC(परिमाण(संरचना ggml_context));

    // allow to call ggml_init with 0 size
    यदि (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    ध्रुव size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (संरचना ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ रिक्त,
        /*.objects_end        =*/ रिक्त,
    };

    GGML_ASSERT(ctx->mem_buffer != रिक्त);

    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    प्रत्यागम ctx;
}

शून्य ggml_reset(संरचना ggml_context * ctx) {
    यदि (ctx == रिक्त) {
        प्रत्यागम;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = रिक्त;
    ctx->objects_end   = रिक्त;
}

शून्य ggml_free(संरचना ggml_context * ctx) {
    यदि (ctx == रिक्त) {
        प्रत्यागम;
    }

    यदि (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}

size_t ggml_used_mem(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->objects_end == रिक्त ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_get_no_alloc(संरचना ggml_context * ctx) {
    प्रत्यागम ctx->no_alloc;
}

शून्य ggml_set_no_alloc(संरचना ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

शून्य * ggml_get_mem_buffer(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->mem_buffer;
}

size_t ggml_get_mem_size(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->mem_size;
}

size_t ggml_get_max_tensor_size(ध्रुव संरचना ggml_context * ctx) {
    size_t max_size = 0;

    प्रति (संरचना ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != रिक्त; tensor = ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    प्रत्यागम max_size;
}

////////////////////////////////////////////////////////////////////////////////

स्थिर संरचना ggml_object * ggml_new_object(संरचना ggml_context * ctx, गणना ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    संरचना ggml_object * obj_cur = ctx->objects_end;

    ध्रुव size_t cur_offs = obj_cur == रिक्त ? 0 : obj_cur->offs;
    ध्रुव size_t cur_size = obj_cur == रिक्त ? 0 : obj_cur->size;
    ध्रुव size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    वर्ण * ध्रुव mem_buffer = ctx->mem_buffer;
    संरचना ggml_object * ध्रुव obj_new = (संरचना ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    यदि (cur_end > SIZE_MAX - size_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, size_needed);
        प्रत्यागम रिक्त;
    }
    यदि (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, size_needed, (size_t) GGML_OBJECT_SIZE);
        प्रत्यागम रिक्त;
    }

    यदि (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        प्रत्यागम रिक्त;
    }

    *obj_new = (संरचना ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = रिक्त,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    यदि (obj_cur != रिक्त) {
        obj_cur->next = obj_new;
    } अन्यथा {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    प्रत्यागम obj_new;
}

स्थिर संरचना ggml_tensor * ggml_new_tensor_impl(
        संरचना ggml_context * ctx,
        गणना   ggml_type      type,
        पूर्णाङ्क                   n_dims,
        ध्रुव int64_t       * ne,
        संरचना ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_मोजणी);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    यदि (view_src != रिक्त && view_src->view_src != रिक्त) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    प्रति (पूर्णाङ्क i = 1; i < n_dims; i++) {
        यदि (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    प्रति (पूर्णाङ्क i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == रिक्त || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    शून्य * data = view_src != रिक्त ? view_src->data : रिक्त;
    यदि (data != रिक्त) {
        data = (वर्ण *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    यदि (view_src == रिक्त && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    संरचना ggml_object * ध्रुव obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    संरचना ggml_tensor * ध्रुव result = (संरचना ggml_tensor *)((वर्ण *)ctx->mem_buffer + obj_new->offs);

    *result = (संरचना ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ रिक्त,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { रिक्त },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (शून्य *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ रिक्त,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    प्रति (पूर्णाङ्क i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_new_tensor(
        संरचना ggml_context * ctx,
        गणना   ggml_type      type,
        पूर्णाङ्क                   n_dims,
        ध्रुव int64_t       * ne) {
    प्रत्यागम ggml_new_tensor_impl(ctx, type, n_dims, ne, रिक्त, 0);
}

संरचना ggml_tensor * ggml_new_tensor_1d(
        संरचना ggml_context * ctx,
        गणना   ggml_type      type,
        int64_t ne0) {
    प्रत्यागम ggml_new_tensor(ctx, type, 1, &ne0);
}

संरचना ggml_tensor * ggml_new_tensor_2d(
        संरचना ggml_context * ctx,
        गणना   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    ध्रुव int64_t ne[2] = { ne0, ne1 };
    प्रत्यागम ggml_new_tensor(ctx, type, 2, ne);
}

संरचना ggml_tensor * ggml_new_tensor_3d(
        संरचना ggml_context * ctx,
        गणना   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ध्रुव int64_t ne[3] = { ne0, ne1, ne2 };
    प्रत्यागम ggml_new_tensor(ctx, type, 3, ne);
}

संरचना ggml_tensor * ggml_new_tensor_4d(
        संरचना ggml_context * ctx,
        गणना   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    ध्रुव int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    प्रत्यागम ggml_new_tensor(ctx, type, 4, ne);
}

शून्य * ggml_new_buffer(संरचना ggml_context * ctx, size_t nbytes) {
    संरचना ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    प्रत्यागम (uint8_t *)ctx->mem_buffer + obj->offs;
}

संरचना ggml_tensor * ggml_dup_tensor(संरचना ggml_context * ctx, ध्रुव संरचना ggml_tensor * src) {
    प्रत्यागम ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

शून्य ggml_unravel_index(ध्रुव संरचना ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    ध्रुव int64_t ne2 = tensor->ne[2];
    ध्रुव int64_t ne1 = tensor->ne[1];
    ध्रुव int64_t ne0 = tensor->ne[0];

    ध्रुव int64_t i3_ = (i/(ne2*ne1*ne0));
    ध्रुव int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    ध्रुव int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    ध्रुव int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    यदि (i0) {
        * i0 = i0_;
    }
    यदि (i1) {
        * i1 = i1_;
    }
    यदि (i2) {
        * i2 = i2_;
    }
    यदि (i3) {
        * i3 = i3_;
    }
}

शून्य * ggml_get_data(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->data;
}

भिन्न * ggml_get_data_f32(ध्रुव संरचना ggml_tensor * tensor) {
    प्रतिज्ञा(tensor->type == GGML_TYPE_F32);
    प्रत्यागम (भिन्न *)(tensor->data);
}

गणना ggml_unary_op ggml_get_unary_op(ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    प्रत्यागम (गणना ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

गणना ggml_glu_op ggml_get_glu_op(ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    प्रत्यागम (गणना ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

ध्रुव वर्ण * ggml_get_name(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->name;
}

संरचना ggml_tensor * ggml_set_name(संरचना ggml_tensor * tensor, ध्रुव वर्ण * name) {
    size_t i;
    प्रति (i = 0; i < परिमाण(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    प्रत्यागम tensor;
}

संरचना ggml_tensor * ggml_format_name(संरचना ggml_tensor * tensor, ध्रुव वर्ण * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, परिमाण(tensor->name), fmt, args);
    va_end(args);
    प्रत्यागम tensor;
}

संरचना ggml_tensor * ggml_view_tensor(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * src) {
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(result, "%s (view)", src->name);

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_get_first_tensor(ध्रुव संरचना ggml_context * ctx) {
    संरचना ggml_object * obj = ctx->objects_begin;

    वर्ण * ध्रुव mem_buffer = ctx->mem_buffer;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            प्रत्यागम (संरचना ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_get_next_tensor(ध्रुव संरचना ggml_context * ctx, संरचना ggml_tensor * tensor) {
    संरचना ggml_object * obj = (संरचना ggml_object *) ((वर्ण *)tensor - GGML_OBJECT_SIZE);
    obj = obj->next;

    वर्ण * ध्रुव mem_buffer = ctx->mem_buffer;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            प्रत्यागम (संरचना ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_get_tensor(संरचना ggml_context * ctx, ध्रुव वर्ण * name) {
    संरचना ggml_object * obj = ctx->objects_begin;

    वर्ण * ध्रुव mem_buffer = ctx->mem_buffer;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            संरचना ggml_tensor * cur = (संरचना ggml_tensor *)(mem_buffer + obj->offs);
            यदि (strcmp(cur->name, name) == 0) {
                प्रत्यागम cur;
            }
        }

        obj = obj->next;
    }

    प्रत्यागम रिक्त;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

स्थिर संरचना ggml_tensor * ggml_dup_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DUP;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_dup(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_dup_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_dup_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_dup_impl(ctx, a, true);
}

// ggml_add

स्थिर संरचना ggml_tensor * ggml_add_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_add(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_add_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_add_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

स्थिर संरचना ggml_tensor * ggml_add_cast_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    result->op     = GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_add_cast(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना   ggml_type      type) {
    प्रत्यागम ggml_add_cast_impl(ctx, a, b, type);
}

संरचना ggml_tensor * ggml_add_id(
            संरचना ggml_context * ctx,
            संरचना ggml_tensor  * a,
            संरचना ggml_tensor  * b,
            संरचना ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    प्रत्यागम result;
}

// ggml_add1

स्थिर संरचना ggml_tensor * ggml_add1_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_add1(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_add1_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_add1_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

स्थिर संरचना ggml_tensor * ggml_acc_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_acc(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    प्रत्यागम ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

संरचना ggml_tensor * ggml_acc_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    प्रत्यागम ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

स्थिर संरचना ggml_tensor * ggml_sub_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_sub(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_sub_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_sub_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

स्थिर संरचना ggml_tensor * ggml_mul_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_mul(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_mul_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_mul_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

स्थिर संरचना ggml_tensor * ggml_div_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_div(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_div_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_div_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

स्थिर संरचना ggml_tensor * ggml_sqr_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQR;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_sqr(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sqr_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_sqr_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

स्थिर संरचना ggml_tensor * ggml_sqrt_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SQRT;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_sqrt(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sqrt_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_sqrt_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

स्थिर संरचना ggml_tensor * ggml_log_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_LOG;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_log(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_log_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_log_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_log_impl(ctx, a, true);
}

संरचना ggml_tensor * ggml_expm1(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

संरचना ggml_tensor * ggml_expm1_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

संरचना ggml_tensor * ggml_softplus(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

संरचना ggml_tensor * ggml_softplus_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

स्थिर संरचना ggml_tensor * ggml_sin_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SIN;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_sin(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sin_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_sin_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_sin_impl(ctx, a, true);
}

// ggml_cos

स्थिर संरचना ggml_tensor * ggml_cos_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_COS;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_cos(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_cos_impl(ctx, a, false);
}

संरचना ggml_tensor * ggml_cos_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_cos_impl(ctx, a, true);
}

// ggml_sum

संरचना ggml_tensor * ggml_sum(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_SUM;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_sum_rows

संरचना ggml_tensor * ggml_sum_rows(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    result->op     = GGML_OP_SUM_ROWS;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_cumsum

संरचना ggml_tensor * ggml_cumsum(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_CUMSUM;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_mean

संरचना ggml_tensor * ggml_mean(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MEAN;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_argmax

संरचना ggml_tensor * ggml_argmax(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    result->op     = GGML_OP_ARGMAX;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_count_equal

संरचना ggml_tensor * ggml_मोजणी_equal(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    result->op     = GGML_OP_मोजणी_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_repeat

संरचना ggml_tensor * ggml_repeat(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_repeat_4d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ध्रुव bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = GGML_OP_REPEAT;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_repeat_back

संरचना ggml_tensor * ggml_repeat_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_concat

संरचना ggml_tensor * ggml_concat(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    संरचना ggml_tensor  * b,
    पूर्णाङ्क                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    प्रति (पूर्णाङ्क d = 0; d < GGML_MAX_DIMS; ++d) {
        यदि (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            अनुवर्त;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(result, 0, dim);

    result->op     = GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_abs

संरचना ggml_tensor * ggml_abs(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

संरचना ggml_tensor * ggml_abs_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

संरचना ggml_tensor * ggml_sgn(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

संरचना ggml_tensor * ggml_sgn_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

संरचना ggml_tensor * ggml_neg(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

संरचना ggml_tensor * ggml_neg_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

संरचना ggml_tensor * ggml_step(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

संरचना ggml_tensor * ggml_step_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

संरचना ggml_tensor * ggml_tanh(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

संरचना ggml_tensor * ggml_tanh_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

संरचना ggml_tensor * ggml_elu(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

संरचना ggml_tensor * ggml_elu_inplace(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

संरचना ggml_tensor * ggml_relu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

संरचना ggml_tensor * ggml_relu_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

संरचना ggml_tensor * ggml_leaky_relu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 negative_slope,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &negative_slope, परिमाण(negative_slope));

    result->op     = GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_sigmoid

संरचना ggml_tensor * ggml_sigmoid(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

संरचना ggml_tensor * ggml_sigmoid_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

संरचना ggml_tensor * ggml_gelu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

संरचना ggml_tensor * ggml_gelu_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

संरचना ggml_tensor * ggml_gelu_erf(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

संरचना ggml_tensor * ggml_gelu_erf_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

संरचना ggml_tensor * ggml_gelu_quick(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

संरचना ggml_tensor * ggml_gelu_quick_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

संरचना ggml_tensor * ggml_silu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

संरचना ggml_tensor * ggml_silu_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

संरचना ggml_tensor * ggml_xielu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न alpha_n,
        भिन्न alpha_p,
        भिन्न beta,
        भिन्न eps) {
    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(result, 3, beta);
    ggml_set_op_params_f32(result, 4, eps);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_silu_back

संरचना ggml_tensor * ggml_silu_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml hardswish

संरचना ggml_tensor * ggml_hardswish(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

संरचना ggml_tensor * ggml_hardsigmoid(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

संरचना ggml_tensor * ggml_exp(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

संरचना ggml_tensor * ggml_exp_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

स्थिर संरचना ggml_tensor * ggml_glu_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    यदि (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, रिक्त, 0);

    ggml_set_op_params_i32(result, 0, (int32_t) op);
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);

    result->op     = GGML_OP_GLU;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_floor

संरचना ggml_tensor * ggml_floor(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

संरचना ggml_tensor * ggml_floor_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

संरचना ggml_tensor * ggml_ceil(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

संरचना ggml_tensor * ggml_ceil_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

संरचना ggml_tensor * ggml_round(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

संरचना ggml_tensor * ggml_round_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

संरचना ggml_tensor * ggml_trunc(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

संरचना ggml_tensor * ggml_trunc_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

संरचना ggml_tensor * ggml_glu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_glu_op      op,
        bool                  swapped) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, op, swapped);
}

संरचना ggml_tensor * ggml_glu_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना ggml_glu_op      op) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

संरचना ggml_tensor * ggml_reglu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_REGLU, false);
}

संरचना ggml_tensor * ggml_reglu_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_REGLU, true);
}

संरचना ggml_tensor * ggml_reglu_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

संरचना ggml_tensor * ggml_geglu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU, false);
}

संरचना ggml_tensor * ggml_geglu_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU, true);
}

संरचना ggml_tensor * ggml_geglu_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

संरचना ggml_tensor * ggml_swiglu(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_SWIGLU, false);
}

संरचना ggml_tensor * ggml_swiglu_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_SWIGLU, true);
}

संरचना ggml_tensor * ggml_swiglu_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

संरचना ggml_tensor * ggml_geglu_erf(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_ERF, false);
}

संरचना ggml_tensor * ggml_geglu_erf_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_ERF, true);
}

संरचना ggml_tensor * ggml_geglu_erf_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

संरचना ggml_tensor * ggml_geglu_quick(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_QUICK, false);
}

संरचना ggml_tensor * ggml_geglu_quick_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_QUICK, true);
}

संरचना ggml_tensor * ggml_geglu_quick_split(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

संरचना ggml_tensor * ggml_swiglu_oai(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 alpha,
        भिन्न                 limit) {
    संरचना ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(result, 2, alpha);
    ggml_set_op_params_f32(result, 3, limit);

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_swiglu_clamp(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 limit) {
    संरचना ggml_tensor * result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(result, 3, limit);

    प्रत्यागम result;
}

// ggml_norm

स्थिर संरचना ggml_tensor * ggml_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, परिमाण(eps));

    result->op     = GGML_OP_NORM;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_norm(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_norm_impl(ctx, a, eps, false);
}

संरचना ggml_tensor * ggml_norm_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

स्थिर संरचना ggml_tensor * ggml_rms_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, परिमाण(eps));

    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_rms_norm(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_rms_norm_impl(ctx, a, eps, false);
}

संरचना ggml_tensor * ggml_rms_norm_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

संरचना ggml_tensor * ggml_rms_norm_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 eps) {
    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(result, &eps, परिमाण(eps));

    result->op     = GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_group_norm

स्थिर संरचना ggml_tensor * ggml_group_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, n_groups);
    ggml_set_op_params_f32(result, 1, eps);

    result->op     = GGML_OP_GROUP_NORM;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_group_norm(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps) {
    प्रत्यागम ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

संरचना ggml_tensor * ggml_group_norm_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps) {
    प्रत्यागम ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

स्थिर संरचना ggml_tensor * ggml_l2_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, eps);

    result->op     = GGML_OP_L2_NORM;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_l2_norm(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_l2_norm_impl(ctx, a, eps, false);
}

संरचना ggml_tensor * ggml_l2_norm_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps) {
    प्रत्यागम ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    विकल्प (a->op) {
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ध्रुव int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ध्रुव int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            भङ्ग;
        स्वभाव:
            प्रत्यागम false;
    };

    प्रत्यागम true;
}

bool ggml_prec_set_src(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec,
        पूर्णाङ्क                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    विकल्प (a->op) {
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                यदि (idx != 1) {
                    प्रत्यागम false;
                }

                ध्रुव int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            भङ्ग;
        स्वभाव:
            प्रत्यागम false;
    };

    प्रत्यागम true;
}

// ggml_mul_mat

स्थिर अन्तःस्थ bool ggml_can_mul_mat(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

संरचना ggml_tensor * ggml_mul_mat(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    ध्रुव int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

शून्य ggml_mul_mat_set_prec(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ध्रुव int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

शून्य ggml_mul_mat_set_hint(
        संरचना ggml_tensor * a,
        गणना ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ध्रुव int32_t hint_i32 = (int32_t) hint;

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
संरचना ggml_tensor * ggml_mul_mat_id(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * as,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    ध्रुव int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    प्रत्यागम result;
}

// ggml_out_prod

स्थिर अन्तःस्थ bool ggml_can_out_prod(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

संरचना ggml_tensor * ggml_out_prod(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ध्रुव int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_scale

स्थिर संरचना ggml_tensor * ggml_scale_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s,
        भिन्न                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[2] = { s, b };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op     = GGML_OP_SCALE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_scale(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s) {
    प्रत्यागम ggml_scale_impl(ctx, a, s, 0.0, false);
}

संरचना ggml_tensor * ggml_scale_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s) {
    प्रत्यागम ggml_scale_impl(ctx, a, s, 0.0, true);
}

संरचना ggml_tensor * ggml_scale_bias(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s,
        भिन्न                 b) {
    प्रत्यागम ggml_scale_impl(ctx, a, s, b, false);
}

संरचना ggml_tensor * ggml_scale_bias_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s,
        भिन्न                 b) {
    प्रत्यागम ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

स्थिर संरचना ggml_tensor * ggml_set_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_set(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

संरचना ggml_tensor * ggml_set_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

संरचना ggml_tensor * ggml_set_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

संरचना ggml_tensor * ggml_set_1d_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

संरचना ggml_tensor * ggml_set_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

संरचना ggml_tensor * ggml_set_2d_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    प्रत्यागम ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

स्थिर संरचना ggml_tensor * ggml_cpy_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    संरचना ggml_tensor * result = ggml_view_tensor(ctx, b);
    यदि (strlen(b->name) > 0) {
        ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } अन्यथा {
        ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_cpy(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a,
        संरचना ggml_tensor * b) {
    प्रत्यागम ggml_cpy_impl(ctx, a, b);
}

संरचना ggml_tensor * ggml_cast(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना   ggml_type      type) {
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(result, "%s (copy)", a->name);

    result->op     = GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    प्रत्यागम result;
}

// ggml_cont

स्थिर संरचना ggml_tensor * ggml_cont_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_cont(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a) {
    प्रत्यागम ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API संरचना ggml_tensor * ggml_cont_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0) {
    प्रत्यागम ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API संरचना ggml_tensor * ggml_cont_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    प्रत्यागम ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API संरचना ggml_tensor * ggml_cont_3d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    प्रत्यागम ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

संरचना ggml_tensor * ggml_cont_4d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(result, "%s (cont)", a->name);

    result->op     = GGML_OP_CONT;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_reshape

संरचना ggml_tensor * ggml_reshape(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a,
        संरचना ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_reshape_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ध्रुव int64_t ne[1] = { ne0 };
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_reshape_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ध्रुव int64_t ne[2] = { ne0, ne1 };
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_reshape_3d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    ध्रुव int64_t ne[3] = { ne0, ne1, ne2 };
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_reshape_4d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    ध्रुव int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;

    प्रत्यागम result;
}

स्थिर संरचना ggml_tensor * ggml_view_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_dims,
        ध्रुव int64_t       * ne,
        size_t                offset) {
    संरचना ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);

    ggml_set_op_params(result, &offset, परिमाण(offset));

    result->op     = GGML_OP_VIEW;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_view_1d

संरचना ggml_tensor * ggml_view_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    संरचना ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);

    प्रत्यागम result;
}

// ggml_view_2d

संरचना ggml_tensor * ggml_view_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    ध्रुव int64_t ne[2] = { ne0, ne1 };

    संरचना ggml_tensor * result = ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    प्रत्यागम result;
}

// ggml_view_3d

संरचना ggml_tensor * ggml_view_3d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    ध्रुव int64_t ne[3] = { ne0, ne1, ne2 };

    संरचना ggml_tensor * result = ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    प्रत्यागम result;
}

// ggml_view_4d

संरचना ggml_tensor * ggml_view_4d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    ध्रुव int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    संरचना ggml_tensor * result = ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    प्रत्यागम result;
}

// ggml_permute

संरचना ggml_tensor * ggml_permute(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   axis0,
        पूर्णाङ्क                   axis1,
        पूर्णाङ्क                   axis2,
        पूर्णाङ्क                   axis3) {
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

    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);
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
    ggml_set_op_params(result, params, परिमाण(params));

    प्रत्यागम result;
}

// ggml_transpose

संरचना ggml_tensor * ggml_transpose(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_get_rows

संरचना ggml_tensor * ggml_get_rows(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    गणना ggml_type type = GGML_TYPE_F32;
    यदि (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_get_rows_back

संरचना ggml_tensor * ggml_get_rows_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    संरचना ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_set_rows

संरचना ggml_tensor * ggml_set_rows(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c) {
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

    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_SET_ROWS;
    result->src[0] = b;
    result->src[1] = c;
    result->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    प्रत्यागम result;
}

// ggml_diag

संरचना ggml_tensor * ggml_diag(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ध्रुव int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = GGML_OP_DIAG;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_diag_mask_inf

स्थिर संरचना ggml_tensor * ggml_diag_mask_inf_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_diag_mask_inf(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

संरचना ggml_tensor * ggml_diag_mask_inf_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

स्थिर संरचना ggml_tensor * ggml_diag_mask_zero_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_diag_mask_zero(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

संरचना ggml_tensor * ggml_diag_mask_zero_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

स्थिर संरचना ggml_tensor * ggml_clamp_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 min,
        भिन्न                 max,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[] = { min, max };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_CLAMP;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_clamp(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 min,
    भिन्न                 max) {
    प्रत्यागम ggml_clamp_impl(ctx, a, min, max, false);
}

संरचना ggml_tensor * ggml_clamp_inplace(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 min,
    भिन्न                 max) {
    प्रत्यागम ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

स्थिर संरचना ggml_tensor * ggml_soft_max_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * mask,
        भिन्न                 scale,
        भिन्न                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    यदि (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    यदि (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[] = { scale, max_bias };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_soft_max(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, रिक्त, 1.0f, 0.0f, false);
}

संरचना ggml_tensor * ggml_soft_max_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, रिक्त, 1.0f, 0.0f, true);
}

संरचना ggml_tensor * ggml_soft_max_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * mask,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

संरचना ggml_tensor * ggml_soft_max_ext_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * mask,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

शून्य ggml_soft_max_add_sinks(
        संरचना ggml_tensor * a,
        संरचना ggml_tensor * sinks) {
    यदि (!sinks) {
        a->src[2] = रिक्त;
        प्रत्यागम;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == रिक्त);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

स्थिर संरचना ggml_tensor * ggml_soft_max_ext_back_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias,
        bool                  inplace) {
    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op     = GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    memcpy((भिन्न *) result->op_params + 0, &scale,    परिमाण(भिन्न));
    memcpy((भिन्न *) result->op_params + 1, &max_bias, परिमाण(भिन्न));

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_soft_max_ext_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

संरचना ggml_tensor * ggml_soft_max_ext_back_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

स्थिर संरचना ggml_tensor * ggml_rope_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    यदि (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } अन्यथा {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    यदि (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    परिमाण(भिन्न));
    memcpy(params +  6, &freq_scale,   परिमाण(भिन्न));
    memcpy(params +  7, &ext_factor,   परिमाण(भिन्न));
    memcpy(params +  8, &attn_factor,  परिमाण(भिन्न));
    memcpy(params +  9, &beta_fast,    परिमाण(भिन्न));
    memcpy(params + 10, &beta_slow,    परिमाण(भिन्न));
    यदि (mrope_used && sections) {
        memcpy(params + 11, sections,  परिमाण(int32_t) * GGML_MROPE_SECTIONS);
    } अन्यथा {
        memset(params + 11, 0,         परिमाण(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_rope(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

संरचना ggml_tensor * ggml_rope_multi(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_multi_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

संरचना ggml_tensor * ggml_rope_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

संरचना ggml_tensor * ggml_rope_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, रिक्त, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_ext_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, रिक्त, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

संरचना ggml_tensor * ggml_rope_custom(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_custom_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
स्थिर भिन्न ggml_rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न n_rot, भिन्न base) {
    प्रत्यागम n_dims * logf(n_ctx_orig / (n_rot * 2 * (भिन्न)M_PI)) / (2 * logf(base));
}

शून्य ggml_rope_yarn_corr_dims(
    पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न freq_base, भिन्न beta_fast, भिन्न beta_slow, भिन्न dims[2]
) {
    // start and end correction dims
    भिन्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    भिन्न end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

संरचना ggml_tensor * ggml_rope_ext_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    संरचना ggml_tensor * result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_rope_multi_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[4],
        पूर्णाङ्क                   mode,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_base,
        भिन्न                 freq_scale,
        भिन्न                 ext_factor,
        भिन्न                 attn_factor,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    संरचना ggml_tensor * result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_rope_set_offset(
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    ध्रुव int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    प्रत्यागम a;
}

स्थिर int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p, पूर्णाङ्क d) {
    प्रत्यागम (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
संरचना ggml_tensor * ggml_im2col(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   d0,
        पूर्णाङ्क                   d1,
        bool                  is_2D,
        गणना ggml_type        dst_type) {
    यदि (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } अन्यथा {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    ध्रुव int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ध्रुव int64_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ध्रुव int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_im2col_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        int64_t             * ne,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   d0,
        पूर्णाङ्क                   d1,
        bool                  is_2D) {
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_conv_1d

संरचना ggml_tensor * ggml_conv_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   d0) {
    संरचना ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    संरचना ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [N, OC, OL]

    प्रत्यागम result;
}

// ggml_conv_1d_ph

संरचना ggml_tensor* ggml_conv_1d_ph(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s,
        पूर्णाङ्क                   d) {
    प्रत्यागम ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

संरचना ggml_tensor * ggml_conv_1d_dw(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   d0) {
    संरचना ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    संरचना ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    संरचना ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);

    result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

    प्रत्यागम result;
}

// ggml_conv_1d_dw_ph

संरचना ggml_tensor * ggml_conv_1d_dw_ph(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   d0) {
    प्रत्यागम ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

संरचना ggml_tensor * ggml_col2im_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   oc,
        पूर्णाङ्क                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    ध्रुव int64_t K_OC = a->ne[0];
    ध्रुव int64_t T_in = a->ne[1];
    ध्रुव int64_t K = K_OC / oc;
    ध्रुव int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    ध्रुव int64_t ne[4] = { T_out, oc, 1, 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_COL2IM_1D;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_conv_transpose_1d

स्थिर int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p, पूर्णाङ्क d) {
    प्रत्यागम (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API संरचना ggml_tensor * ggml_conv_transpose_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    ध्रुव int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
संरचना ggml_tensor * ggml_conv_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   d0,
        पूर्णाङ्क                   d1) {
    संरचना ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    संरचना ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    प्रत्यागम result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
संरचना ggml_tensor * ggml_im2col_3d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        int64_t               IC,
        पूर्णाङ्क                   s0, // stride width
        पूर्णाङ्क                   s1, // stride height
        पूर्णाङ्क                   s2, // stride depth
        पूर्णाङ्क                   p0, // padding width
        पूर्णाङ्क                   p1, // padding height
        पूर्णाङ्क                   p2, // padding depth
        पूर्णाङ्क                   d0, // dilation width
        पूर्णाङ्क                   d1, // dilation height
        पूर्णाङ्क                   d2, // dilation depth
        गणना ggml_type        dst_type) {
    ध्रुव int64_t N = b->ne[3] / IC;
    ध्रुव int64_t ID = b->ne[2];
    ध्रुव int64_t IH = b->ne[1];
    ध्रुव int64_t IW = b->ne[0];

    ध्रुव int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    ध्रुव int64_t KD = a->ne[2];
    ध्रुव int64_t KH = a->ne[1];
    ध्रुव int64_t KW = a->ne[0];
    ध्रुव int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
    ध्रुव int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
    ध्रुव int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ध्रुव int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_IM2COL_3D;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
संरचना ggml_tensor * ggml_conv_3d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        int64_t               IC,
        पूर्णाङ्क                   s0, // stride width
        पूर्णाङ्क                   s1, // stride height
        पूर्णाङ्क                   s2, // stride depth
        पूर्णाङ्क                   p0, // padding width
        पूर्णाङ्क                   p1, // padding height
        पूर्णाङ्क                   p2, // padding depth
        पूर्णाङ्क                   d0, // dilation width
        पूर्णाङ्क                   d1, // dilation height
        पूर्णाङ्क                   d2  // dilation depth
        ) {
    संरचना ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    संरचना ggml_tensor * result =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    result = ggml_reshape_4d(ctx, result, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    प्रत्यागम result;
}

// ggml_conv_2d_sk_p0

संरचना ggml_tensor * ggml_conv_2d_sk_p0(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

संरचना ggml_tensor * ggml_conv_2d_s1_ph(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

संरचना ggml_tensor * ggml_conv_2d_dw(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   d0,
        पूर्णाङ्क                   d1) {
    संरचना ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    संरचना ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    संरचना ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    संरचना ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    प्रत्यागम result;
}

// ggml_conv_2d_dw_direct

संरचना ggml_tensor * ggml_conv_2d_dw_direct(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   stride0,
        पूर्णाङ्क                   stride1,
        पूर्णाङ्क                   pad0,
        पूर्णाङ्क                   pad1,
        पूर्णाङ्क                   dilation0,
        पूर्णाङ्क                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    यदि (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ध्रुव int64_t type_size = ggml_type_size(result->type);
        GGML_ASSERT(ggml_blck_size(result->type) == 1);
        result->nb[0] = result->ne[2] * type_size;
        result->nb[1] = result->ne[0] * result->nb[0];
        result->nb[2] = type_size;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_CONV_2D_DW;
    result->src[0] = a;
    result->src[1] = b;
    प्रत्यागम result;
}

// ggml_conv_2d_direct

संरचना ggml_tensor * ggml_conv_2d_direct(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        संरचना ggml_tensor  * b,   // input data [W, H, C, N]
        पूर्णाङ्क                   s0,  // stride dimension 0
        पूर्णाङ्क                   s1,  // stride dimension 1
        पूर्णाङ्क                   p0,  // padding dimension 0
        पूर्णाङ्क                   p1,  // padding dimension 1
        पूर्णाङ्क                   d0,  // dilation dimension 0
        पूर्णाङ्क                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(result, 0, s0);
    ggml_set_op_params_i32(result, 1, s1);
    ggml_set_op_params_i32(result, 2, p0);
    ggml_set_op_params_i32(result, 3, p1);
    ggml_set_op_params_i32(result, 4, d0);
    ggml_set_op_params_i32(result, 5, d1);

    result->op = GGML_OP_CONV_2D;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_conv_3d_direct

संरचना ggml_tensor * ggml_conv_3d_direct(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        पूर्णाङ्क                   s2,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   p2,
        पूर्णाङ्क                   d0,
        पूर्णाङ्क                   d1,
        पूर्णाङ्क                   d2,
        पूर्णाङ्क                   c,
        पूर्णाङ्क                   n,
        पूर्णाङ्क                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

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

    प्रत्यागम result;
}

// ggml_conv_transpose_2d_p0

स्थिर int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p) {
    प्रत्यागम (ins - 1) * s - 2 * p + ks;
}

संरचना ggml_tensor * ggml_conv_transpose_2d_p0(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ध्रुव int64_t ne[4] = {
        ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    संरचना ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, stride);

    result->op     = GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_pool_*

स्थिर int64_t ggml_calc_pool_output_size(int64_t ins, पूर्णाङ्क ks, पूर्णाङ्क s, भिन्न p) {
    प्रत्यागम (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

संरचना ggml_tensor * ggml_pool_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_op_pool     op,
        पूर्णाङ्क                   k0,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   p0) {
    ध्रुव int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_POOL_1D;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_pool_2d

संरचना ggml_tensor * ggml_pool_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_op_pool     op,
        पूर्णाङ्क                   k0,
        पूर्णाङ्क                   k1,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        भिन्न                 p0,
        भिन्न                 p1) {
    संरचना ggml_tensor * result;
    ध्रुव int64_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_POOL_2D;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_pool_2d_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * af,
        गणना ggml_op_pool     op,
        पूर्णाङ्क                   k0,
        पूर्णाङ्क                   k1,
        पूर्णाङ्क                   s0,
        पूर्णाङ्क                   s1,
        भिन्न                 p0,
        भिन्न                 p1) {
    संरचना ggml_tensor * result;
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    प्रत्यागम result;
}

// ggml_upscale / ggml_interpolate

स्थिर संरचना ggml_tensor * ggml_interpolate_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_मोजणी);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(result, 0, (int32_t)mode);

    result->op     = GGML_OP_UPSCALE;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_upscale(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   scale_factor,
        गणना ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    प्रत्यागम ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

संरचना ggml_tensor * ggml_upscale_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   ne0,
        पूर्णाङ्क                   ne1,
        पूर्णाङ्क                   ne2,
        पूर्णाङ्क                   ne3,
        गणना ggml_scale_mode  mode) {
    प्रत्यागम ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

संरचना ggml_tensor * ggml_interpolate(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    प्रत्यागम ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

संरचना ggml_tensor * ggml_pad(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   p2,
        पूर्णाङ्क                   p3) {
    प्रत्यागम ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

संरचना ggml_tensor * ggml_pad_circular(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1,
        पूर्णाङ्क                   p2,
        पूर्णाङ्क                   p3) {
    प्रत्यागम ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

संरचना ggml_tensor * ggml_pad_ext(
            संरचना ggml_context * ctx,
            संरचना ggml_tensor  * a,
            पूर्णाङ्क                  lp0,
            पूर्णाङ्क                  rp0,
            पूर्णाङ्क                  lp1,
            पूर्णाङ्क                  rp1,
            पूर्णाङ्क                  lp2,
            पूर्णाङ्क                  rp2,
            पूर्णाङ्क                  lp3,
            पूर्णाङ्क                  rp3
            ) {
    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
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

    प्रत्यागम result;
}

// ggml_pad_ext_circular

संरचना ggml_tensor * ggml_pad_ext_circular(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                  lp0,
        पूर्णाङ्क                  rp0,
        पूर्णाङ्क                  lp1,
        पूर्णाङ्क                  rp1,
        पूर्णाङ्क                  lp2,
        पूर्णाङ्क                  rp2,
        पूर्णाङ्क                  lp3,
        पूर्णाङ्क                  rp3
        ) {
    संरचना ggml_tensor * result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(result, 8, 1); // circular
    प्रत्यागम result;
}

// ggml_pad_reflect_1d

संरचना ggml_tensor * ggml_pad_reflect_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   p0,
        पूर्णाङ्क                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_PAD_REFLECT_1D;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_roll

संरचना ggml_tensor * ggml_roll(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   shift0,
        पूर्णाङ्क                   shift1,
        पूर्णाङ्क                   shift2,
        पूर्णाङ्क                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
    GGML_ASSERT(abs(shift0) < a->ne[0]);
    GGML_ASSERT(abs(shift1) < a->ne[1]);
    GGML_ASSERT(abs(shift2) < a->ne[2]);
    GGML_ASSERT(abs(shift3) < a->ne[3]);

    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, shift0);
    ggml_set_op_params_i32(result, 1, shift1);
    ggml_set_op_params_i32(result, 2, shift2);
    ggml_set_op_params_i32(result, 3, shift3);

    result->op     = GGML_OP_ROLL;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_timestep_embedding

संरचना ggml_tensor * ggml_timestep_embedding(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * timesteps,
        पूर्णाङ्क                   dim,
        पूर्णाङ्क                   max_period) {

    संरचना ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    प्रत्यागम result;
}

// ggml_tri

संरचना ggml_tensor * ggml_tri(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    गणना ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, type);

    result->op = GGML_OP_TRI;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_fill

स्थिर संरचना ggml_tensor * ggml_fill_impl(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(result, 0, c);

    result->op = GGML_OP_FILL;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_fill(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 c) {
    प्रत्यागम ggml_fill_impl(ctx, a, c, false);
}

संरचना ggml_tensor * ggml_fill_inplace(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 c) {
    प्रत्यागम ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

संरचना ggml_tensor * ggml_argsort(
        संरचना ggml_context  * ctx,
        संरचना ggml_tensor   * a,
        गणना ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_argsort_top_k

संरचना ggml_tensor * ggml_argsort_शीर्ष_k(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    संरचना ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    प्रत्यागम result;
}

// ggml_top_k

संरचना ggml_tensor * ggml_शीर्ष_k(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    result->op     = GGML_OP_शीर्ष_K;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_arange

संरचना ggml_tensor * ggml_arange(
        संरचना ggml_context * ctx,
        भिन्न                 start,
        भिन्न                 stop,
        भिन्न                 step) {
    GGML_ASSERT(stop > start);

    ध्रुव int64_t steps = (int64_t) ceilf((stop - start) / step);

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(result, 0, start);
    ggml_set_op_params_f32(result, 1, stop);
    ggml_set_op_params_f32(result, 2, step);

    result->op = GGML_OP_ARANGE;

    प्रत्यागम result;
}

// ggml_flash_attn_ext

संरचना ggml_tensor * ggml_flash_attn_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * mask,
        भिन्न                 scale,
        भिन्न                 max_bias,
        भिन्न                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    यदि (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    यदि (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    भिन्न params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    प्रत्यागम result;
}


शून्य ggml_flash_attn_ext_set_prec(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ध्रुव int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

गणना ggml_prec ggml_flash_attn_ext_get_prec(
        ध्रुव संरचना ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ध्रुव int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    प्रत्यागम (गणना ggml_prec) prec_i32;
}

शून्य ggml_flash_attn_ext_set_n_kv_max(
        संरचना ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

शून्य ggml_flash_attn_ext_add_sinks(
        संरचना ggml_tensor * a,
        संरचना ggml_tensor * sinks) {
    यदि (!sinks) {
        a->src[4] = रिक्त;
        प्रत्यागम;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == रिक्त);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

संरचना ggml_tensor * ggml_flash_attn_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    ध्रुव int64_t     D = q->ne[0];
    ध्रुव int64_t     N = q->ne[1];
    ध्रुव int64_t     M = k->ne[1];
    ध्रुव int64_t   ne2 = q->ne[2];
    ध्रुव int64_t   ne3 = q->ne[3];
    ध्रुव int64_t kvne2 = k->ne[2];

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
    ध्रुव int64_t elem_q = ggml_nelements(q);
    ध्रुव int64_t elem_k = ggml_nelements(k);
    ध्रुव int64_t elem_v = ggml_nelements(v);

    गणना ggml_type result_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ध्रुव size_t tsize = ggml_type_size(result_type);

    ध्रुव size_t offs_q = 0;
    ध्रुव size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ध्रुव size_t nelements = (end + tsize - 1)/tsize;

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(result, &masked_i, परिमाण(masked_i));

    result->op     = GGML_OP_FLASH_ATTN_BACK;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    प्रत्यागम result;
}

// ggml_ssm_conv

संरचना ggml_tensor * ggml_ssm_conv(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * sx,
        संरचना ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    ध्रुव int64_t d_conv  = c->ne[0];
    ध्रुव int64_t d_inner = c->ne[1];
    ध्रुव int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    ध्रुव int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    संरचना ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    प्रत्यागम result;
}

// ggml_ssm_scan

संरचना ggml_tensor * ggml_ssm_scan(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * s,
        संरचना ggml_tensor  * x,
        संरचना ggml_tensor  * dt,
        संरचना ggml_tensor  * A,
        संरचना ggml_tensor  * B,
        संरचना ggml_tensor  * C,
        संरचना ggml_tensor  * ids,
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
        ध्रुव int64_t d_state      = s->ne[0];
        ध्रुव int64_t head_dim     = x->ne[0];
        ध्रुव int64_t n_head       = x->ne[1];
        ध्रुव int64_t n_seq_tokens = x->ne[2];
        ध्रुव int64_t n_seqs       = x->ne[3];

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

        यदि (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    result->op   = GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = ids;

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    प्रत्यागम result;
}

// ggml_win_part

संरचना ggml_tensor * ggml_win_part(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    ध्रुव पूर्णाङ्क px = (w - a->ne[1]%w)%w;
    ध्रुव पूर्णाङ्क py = (w - a->ne[2]%w)%w;

    ध्रुव पूर्णाङ्क npx = (px + a->ne[1])/w;
    ध्रुव पूर्णाङ्क npy = (py + a->ne[2])/w;
    ध्रुव पूर्णाङ्क np  = npx*npy;

    ध्रुव int64_t ne[4] = { a->ne[0], w, w, np, };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_win_unpart

संरचना ggml_tensor * ggml_win_unpart(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   w0,
        पूर्णाङ्क                   h0,
        पूर्णाङ्क                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    ध्रुव int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(result, params, परिमाण(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_get_rel_pos

संरचना ggml_tensor * ggml_get_rel_pos(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   qh,
        पूर्णाङ्क                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    ध्रुव int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    प्रत्यागम result;
}

// ggml_add_rel_pos

स्थिर संरचना ggml_tensor * ggml_add_rel_pos_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * pw,
        संरचना ggml_tensor  * ph,
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

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_add_rel_pos(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * pw,
        संरचना ggml_tensor  * ph) {
    प्रत्यागम ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

संरचना ggml_tensor * ggml_add_rel_pos_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * pw,
        संरचना ggml_tensor  * ph) {
    प्रत्यागम ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

संरचना ggml_tensor * ggml_rwkv_wkv6(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * r,
        संरचना ggml_tensor  * tf,
        संरचना ggml_tensor  * td,
        संरचना ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    ध्रुव int64_t S = k->ne[0];
    ध्रुव int64_t H = k->ne[1];
    ध्रुव int64_t n_tokens = k->ne[2];
    ध्रुव int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ध्रुव int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    प्रत्यागम result;
}

// ggml_gated_linear_attn

संरचना ggml_tensor * ggml_gated_linear_attn(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * g,
        संरचना ggml_tensor  * state,
        भिन्न scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    ध्रुव int64_t S = k->ne[0];
    ध्रुव int64_t H = k->ne[1];
    ध्रुव int64_t n_tokens = k->ne[2];
    ध्रुव int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ध्रुव int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(result, 0, scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = q;
    result->src[3] = g;
    result->src[4] = state;

    प्रत्यागम result;
}

// ggml_rwkv_wkv7

संरचना ggml_tensor * ggml_rwkv_wkv7(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * r,
        संरचना ggml_tensor  * w,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    ध्रुव int64_t S = k->ne[0];
    ध्रुव int64_t H = k->ne[1];
    ध्रुव int64_t n_tokens = k->ne[2];
    ध्रुव int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    ध्रुव int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_RWKV_WKV7;
    result->src[0] = r;
    result->src[1] = w;
    result->src[2] = k;
    result->src[3] = v;
    result->src[4] = a;
    result->src[5] = b;
    result->src[6] = state;

    प्रत्यागम result;
}

// ggml_unary

स्थिर संरचना ggml_tensor * ggml_unary_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = GGML_OP_UNARY;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_unary(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_unary_op    op) {
    प्रत्यागम ggml_unary_impl(ctx, a, op, false);
}

संरचना ggml_tensor * ggml_unary_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_unary_op    op) {
    प्रत्यागम ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

स्थिर संरचना ggml_tensor * ggml_map_custom1_impl(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        ध्रुव  ggml_custom1_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op     = GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_map_custom1(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        ध्रुव  ggml_custom1_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

संरचना ggml_tensor * ggml_map_custom1_inplace(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        ध्रुव  ggml_custom1_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

स्थिर संरचना ggml_tensor * ggml_map_custom2_impl(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        ध्रुव  ggml_custom2_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op     = GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_map_custom2(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        ध्रुव  ggml_custom2_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

संरचना ggml_tensor * ggml_map_custom2_inplace(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        ध्रुव  ggml_custom2_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

स्थिर संरचना ggml_tensor * ggml_map_custom3_impl(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        संरचना ggml_tensor       * c,
        ध्रुव  ggml_custom3_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    संरचना ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op     = GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_map_custom3(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        संरचना ggml_tensor       * c,
        ध्रुव  ggml_custom3_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

संरचना ggml_tensor * ggml_map_custom3_inplace(
        संरचना ggml_context      * ctx,
        संरचना ggml_tensor       * a,
        संरचना ggml_tensor       * b,
        संरचना ggml_tensor       * c,
        ध्रुव  ggml_custom3_op_t   fun,
        पूर्णाङ्क                        n_tasks,
        शून्य                     * userdata) {
    प्रत्यागम ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

संरचना ggml_tensor * ggml_custom_4d(
        संरचना ggml_context * ctx,
        गणना ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        संरचना ggml_tensor ** args,
        पूर्णाङ्क                   n_args,
        ggml_custom_op_t      fun,
        पूर्णाङ्क                   n_tasks,
        शून्य                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    संरचना ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op = GGML_OP_CUSTOM;
    प्रति (पूर्णाङ्क i = 0; i < n_args; i++) {
        result->src[i] = args[i];
    }

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_custom_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor ** args,
        पूर्णाङ्क                   n_args,
        ggml_custom_op_t      fun,
        पूर्णाङ्क                   n_tasks,
        शून्य                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);

    संरचना ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(result, &params, परिमाण(params));

    result->op = GGML_OP_CUSTOM;
    result->src[0] = a;
    प्रति (पूर्णाङ्क i = 0; i < n_args; i++) {
        result->src[i + 1] = args[i];
    }

    प्रत्यागम result;
}
// ggml_cross_entropy_loss

संरचना ggml_tensor * ggml_cross_entropy_loss(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_cross_entropy_loss_back

संरचना ggml_tensor * ggml_cross_entropy_loss_back(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    संरचना ggml_tensor * result = ggml_dup_tensor(ctx, b);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    प्रत्यागम result;
}

// opt_step_adamw

संरचना ggml_tensor * ggml_opt_step_adamw(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * grad,
        संरचना ggml_tensor  * m,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;

    प्रत्यागम result;
}

// opt_step_sgd

संरचना ggml_tensor * ggml_opt_step_sgd(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * grad,
        संरचना ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    संरचना ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op     = GGML_OP_OPT_STEP_SGD;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = params;

    प्रत्यागम result;
}

// solve_tri

संरचना ggml_tensor * ggml_solve_tri(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  डावा,
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

    GGML_ASSERT(lower && डावा && !uni); // TODO: support other variants

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    result->op     = GGML_OP_SOLVE_TRI;
    result->src[0] = a;
    result->src[1] = b;

    प्रत्यागम result;
}

// ggml_gated_delta_net

संरचना ggml_tensor * ggml_gated_delta_net(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * g,
        संरचना ggml_tensor  * beta,
        संरचना ggml_tensor  * state,
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

    ध्रुव int64_t S_v      = v->ne[0];
    ध्रुव int64_t H        = v->ne[1];
    ध्रुव int64_t n_tokens = v->ne[2];
    ध्रुव int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    ध्रुव int64_t state_rows = K * S_v * n_seqs;
    ध्रुव int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(result, 0, (int32_t) K);

    result->op     = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g;
    result->src[4] = beta;
    result->src[5] = state;

    प्रत्यागम result;
}

// ggml_lightning_indexer

संरचना ggml_tensor * ggml_lightning_indexer(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * weights,
        संरचना ggml_tensor  * mask) {

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
    संरचना ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_LIGHTNING_INDEXER;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = weights;
    result->src[3] = mask;

    प्रत्यागम result;
}

// ggml_dsv4_hc_comb

संरचना ggml_tensor * ggml_dsv4_hc_comb(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * mixes,
        संरचना ggml_tensor  * scale,
        संरचना ggml_tensor  * base,
        भिन्न                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    ध्रुव int64_t hc_mix_dim = mixes->ne[0];
    ध्रुव int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    प्रति (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        यदि ((2 + i)*i == hc_mix_dim) {
            hc = i;
            भङ्ग;
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

    संरचना ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(result, 0, eps);
    ggml_set_op_params_i32(result, 1, n_iter);

    result->op     = GGML_OP_DSV4_HC_COMB;
    result->src[0] = mixes;
    result->src[1] = scale;
    result->src[2] = base;

    प्रत्यागम result;
}

// ggml_dsv4_hc_pre

स्थिर संरचना ggml_tensor * ggml_dsv4_hc_pre_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * x,
        संरचना ggml_tensor  * weights,
        भिन्न                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t hc       = x->ne[1];
    ध्रुव int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    यदि (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    संरचना ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_dsv4_hc_pre(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * x,
        संरचना ggml_tensor  * weights) {
    प्रत्यागम ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

संरचना ggml_tensor * ggml_dsv4_hc_pre_gated(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * x,
        संरचना ggml_tensor  * gate,
        भिन्न                 scale) {
    प्रत्यागम ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

संरचना ggml_tensor * ggml_dsv4_hc_post(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * x,
        संरचना ggml_tensor  * residual,
        संरचना ggml_tensor  * post,
        संरचना ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t n_tokens = x->ne[1];
    ध्रुव int64_t hc       = residual->ne[1];

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

    यदि (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    संरचना ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    result->op     = GGML_OP_DSV4_HC_POST;
    result->src[0] = x;
    result->src[1] = residual;
    result->src[2] = post;
    result->src[3] = comb;

    प्रत्यागम result;
}

// ggml_top_k_mask

संरचना ggml_tensor * ggml_शीर्ष_k_mask(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * mask,
        संरचना ggml_tensor  * शीर्ष_k,
        संरचना ggml_tensor  * score,
        संरचना ggml_tensor  * blocks,
        संरचना ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(शीर्ष_k->type == GGML_TYPE_I32);
    GGML_ASSERT(शीर्ष_k->ne[1] % mask->ne[1] == 0 && शीर्ष_k->ne[2] % mask->ne[2] == 0 && शीर्ष_k->ne[3] % mask->ne[3] == 0);

    यदि (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == शीर्ष_k->ne[1] && score->ne[2] == शीर्ष_k->ne[2] && score->ne[3] == शीर्ष_k->ne[3]);
    }

    यदि (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    यदि (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == शीर्ष_k->ne[1] && extra->ne[2] == शीर्ष_k->ne[2] && extra->ne[3] == शीर्ष_k->ne[3]);
    }

    संरचना ggml_tensor * result = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], शीर्ष_k->ne[1], शीर्ष_k->ne[2], शीर्ष_k->ne[3]);

    result->op     = GGML_OP_शीर्ष_K_MASK;
    result->src[0] = mask;
    result->src[1] = शीर्ष_k;
    result->src[2] = score;
    result->src[3] = blocks;
    result->src[4] = extra;

    प्रत्यागम result;
}

////////////////////////////////////////////////////////////////////////////////

संरचना ggml_hash_set ggml_hash_set_new(size_t size) {
    size = ggml_hash_size(size);
    संरचना ggml_hash_set result;
    result.size = size;
    result.किल्ल्या = GGML_MALLOC(परिमाण(संरचना ggml_tensor *) * size);
    result.used = GGML_CALLOC(ggml_bitset_size(size), परिमाण(ggml_bitset_t));
    प्रत्यागम result;
}

शून्य ggml_hash_set_reset(संरचना ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, परिमाण(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}

शून्य ggml_hash_set_free(संरचना ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->किल्ल्या);
}

size_t ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    स्थिर ध्रुव size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    स्थिर ध्रुव size_t n_primes = परिमाण(primes)/परिमाण(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    यावत् (l < r) {
        size_t m = (l + r)/2;
        यदि (primes[m] < min_sz) {
            l = m + 1;
        } अन्यथा {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    प्रत्यागम sz;
}

संरचना hash_map {
    संरचना ggml_hash_set set;
    संरचना ggml_tensor ** vals;
};

स्थिर संरचना hash_map * ggml_new_hash_map(size_t size) {
    संरचना hash_map * result = GGML_MALLOC(परिमाण(संरचना hash_map));
    result->set = ggml_hash_set_new(size);
    result->vals = GGML_CALLOC(result->set.size, परिमाण(संरचना ggml_tensor *));
    प्रत्यागम result;
}

स्थिर शून्य ggml_hash_map_free(संरचना hash_map * map) {
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

स्थिर शून्य ggml_add_or_set(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_set.किल्ल्या[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_acc_or_set(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor,
        ध्रुव  size_t         nb1,
        ध्रुव  size_t         nb2,
        ध्रुव  size_t         nb3,
        ध्रुव  size_t         offset) {
    संरचना ggml_tensor * src = cgraph->visited_hash_set.किल्ल्या[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } अन्यथा {
        संरचना ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.किल्ल्या[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_add1_or_set(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_set.किल्ल्या[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_sub_or_set(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_set.किल्ल्या[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_compute_backward(
        संरचना ggml_context * ctx, संरचना ggml_cgraph * cgraph, पूर्णाङ्क i, ध्रुव bool * grads_needed) {
    संरचना ggml_tensor * tensor = cgraph->nodes[i];
    संरचना ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    यदि (!grad) {
        प्रत्यागम;
    }

    संरचना ggml_tensor * src0 = tensor->src[0];
    संरचना ggml_tensor * src1 = tensor->src[1];
    संरचना ggml_tensor * src2 = tensor->src[2];
    संरचना ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    ध्रुव size_t isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (size_t) -1;
    ध्रुव size_t isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (size_t) -1;
    ध्रुव size_t isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (size_t) -1;
    ध्रुव bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    ध्रुव bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    ध्रुव bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    विकल्प (tensor->op) {
        पक्ष GGML_OP_DUP: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_ADD: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                संरचना ggml_tensor * tmp = grad;
                यदि (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } भङ्ग;
        पक्ष GGML_OP_ADD1: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } भङ्ग;
        पक्ष GGML_OP_ACC: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ध्रुव size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ध्रुव size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ध्रुव size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ध्रुव size_t offset = ((int32_t *) tensor->op_params)[3];

                संरचना ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } भङ्ग;
        पक्ष GGML_OP_SUB: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_MUL: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            यदि (src1_needs_grads) {
                संरचना ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                यदि (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } भङ्ग;
        पक्ष GGML_OP_DIV: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            यदि (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } भङ्ग;
        पक्ष GGML_OP_SQR: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } भङ्ग;
        पक्ष GGML_OP_SQRT: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } भङ्ग;
        पक्ष GGML_OP_LOG: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_SIN: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } भङ्ग;
        पक्ष GGML_OP_COS: {
            यदि (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } भङ्ग;
        पक्ष GGML_OP_SUM: {
            यदि (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_SUM_ROWS: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_MEAN: {
            यदि (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_REPEAT: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_REPEAT_BACK: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_RMS_NORM: {
            यदि (src0_needs_grads) {
                भिन्न eps;
                memcpy(&eps, tensor->op_params, परिमाण(भिन्न));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } भङ्ग;
        पक्ष GGML_OP_MUL_MAT: {
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

            यदि (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                संरचना ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                यदि (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    ध्रुव int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    ध्रुव size_t nb2 = tmp->nb[2] * nr2;
                    ध्रुव size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            यदि (src1_needs_grads) {
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
        } भङ्ग;
        पक्ष GGML_OP_SCALE: {
            यदि (src0_needs_grads) {
                भिन्न s;
                memcpy(&s, tensor->op_params, परिमाण(भिन्न));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_SET: {
            ध्रुव size_t nb1    = ((ध्रुव int32_t *) tensor->op_params)[0];
            ध्रुव size_t nb2    = ((ध्रुव int32_t *) tensor->op_params)[1];
            ध्रुव size_t nb3    = ((ध्रुव int32_t *) tensor->op_params)[2];
            ध्रुव size_t offset = ((ध्रुव int32_t *) tensor->op_params)[3];

            संरचना ggml_tensor * tensor_grad_view = रिक्त;

            यदि (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            यदि (src0_needs_grads) {
                संरचना ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            यदि (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } भङ्ग;
        पक्ष GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            यदि (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            यदि (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } भङ्ग;
        पक्ष GGML_OP_CONT: {
            // same as cpy
            यदि (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_RESHAPE: {
            यदि (src0_needs_grads) {
                संरचना ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_VIEW: {
            यदि (src0_needs_grads) {
                size_t offset;

                memcpy(&offset, tensor->op_params, परिमाण(offset));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                यदि (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
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
        } भङ्ग;
        पक्ष GGML_OP_PERMUTE: {
            यदि (src0_needs_grads) {
                ध्रुव int32_t * axes = (ध्रुव int32_t *) tensor->op_params;
                ध्रुव पूर्णाङ्क axis0 = axes[0] & 0x3;
                ध्रुव पूर्णाङ्क axis1 = axes[1] & 0x3;
                ध्रुव पूर्णाङ्क axis2 = axes[2] & 0x3;
                ध्रुव पूर्णाङ्क axis3 = axes[3] & 0x3;
                पूर्णाङ्क axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } भङ्ग;
        पक्ष GGML_OP_TRANSPOSE: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } भङ्ग;
        पक्ष GGML_OP_GET_ROWS: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            यदि (src1_needs_grads) {
                // noop
            }
        } भङ्ग;
        पक्ष GGML_OP_DIAG_MASK_INF: {
            यदि (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ध्रुव पूर्णाङ्क n_past = ((ध्रुव int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_DIAG_MASK_ZERO: {
            यदि (src0_needs_grads) {
                ध्रुव पूर्णाङ्क n_past = ((ध्रुव int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_SOFT_MAX: {
            यदि (src0_needs_grads) {
                भिन्न scale    = 1.0f;
                भिन्न max_bias = 0.0f;

                memcpy(&scale,    (ध्रुव भिन्न *) tensor->op_params + 0, परिमाण(भिन्न));
                memcpy(&max_bias, (ध्रुव भिन्न *) tensor->op_params + 1, परिमाण(भिन्न));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } भङ्ग;
        पक्ष GGML_OP_ROPE: {
            यदि (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ध्रुव पूर्णाङ्क n_dims     = ((ध्रुव int32_t *) tensor->op_params)[1];
                ध्रुव पूर्णाङ्क mode       = ((ध्रुव int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ध्रुव पूर्णाङ्क n_ctx_orig = ((ध्रुव int32_t *) tensor->op_params)[4];
                भिन्न freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                पूर्णाङ्क sections[4] = {0, 0, 0, 0};

                memcpy(&freq_base,   (ध्रुव भिन्न *) tensor->op_params +  5, परिमाण(भिन्न));
                memcpy(&freq_scale,  (ध्रुव भिन्न *) tensor->op_params +  6, परिमाण(भिन्न));
                memcpy(&ext_factor,  (ध्रुव भिन्न *) tensor->op_params +  7, परिमाण(भिन्न));
                memcpy(&attn_factor, (ध्रुव भिन्न *) tensor->op_params +  8, परिमाण(भिन्न));
                memcpy(&beta_fast,   (ध्रुव भिन्न *) tensor->op_params +  9, परिमाण(भिन्न));
                memcpy(&beta_slow,   (ध्रुव भिन्न *) tensor->op_params + 10, परिमाण(भिन्न));
                memcpy(&sections,                    tensor->op_params + 11, परिमाण(sections));

                संरचना ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } भङ्ग;
        पक्ष GGML_OP_IM2COL: {
            यदि (src1_needs_grads) {
                ध्रुव int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                ध्रुव int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                ध्रुव int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                ध्रुव int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                ध्रुव int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                ध्रुव int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                ध्रुव bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } भङ्ग;
        पक्ष GGML_OP_POOL_2D: {
            यदि (src0_needs_grads) {
                ध्रुव गणना ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                ध्रुव      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                ध्रुव      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                ध्रुव      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                ध्रुव      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                ध्रुव      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                ध्रुव      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } भङ्ग;
        पक्ष GGML_OP_WIN_PART:
        पक्ष GGML_OP_WIN_UNPART:
        पक्ष GGML_OP_UNARY: {
            विकल्प (ggml_get_unary_op(tensor)) {
                पक्ष GGML_UNARY_OP_ABS: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SGN: {
                    // noop
                } भङ्ग;
                पक्ष GGML_UNARY_OP_NEG: {
                    यदि (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_STEP: {
                    // noop
                } भङ्ग;
                पक्ष GGML_UNARY_OP_RELU: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SILU: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_EXP: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_EXPM1: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SOFTPLUS: {
                    यदि (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } भङ्ग;
                स्वभाव: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS: {
            यदि (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } भङ्ग;
        पक्ष GGML_OP_GLU: {
            विकल्प (ggml_get_glu_op(tensor)) {
                पक्ष GGML_GLU_OP_SWIGLU: {
                    यदि (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    यदि (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } भङ्ग;
                स्वभाव: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } भङ्ग;
        पक्ष GGML_OP_NONE: {
            // noop
        } भङ्ग;
        पक्ष GGML_OP_मोजणी:
        स्वभाव: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

स्थिर size_t ggml_visit_parents_graph(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * गाठ, bool compute) {
    यदि (गाठ->op != GGML_OP_NONE && compute) {
        गाठ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ध्रुव size_t गाठ_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, गाठ);
    GGML_ASSERT(गाठ_hash_pos != GGML_HASHSET_FULL);

    यदि (ggml_bitset_get(cgraph->visited_hash_set.used, गाठ_hash_pos)) {
        // already visited

        यदि (compute) {
            // update the compute flag regardless
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; ++i) {
                संरचना ggml_tensor * src = गाठ->src[i];
                यदि (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        प्रत्यागम गाठ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.किल्ल्या[गाठ_hash_pos] = गाठ;
    ggml_bitset_set(cgraph->visited_hash_set.used, गाठ_hash_pos);
    cgraph->use_counts[गाठ_hash_pos] = 0;

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; ++i) {
        ध्रुव पूर्णाङ्क k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_डावा_TO_उजवा) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_उजवा_TO_डावा) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        संरचना ggml_tensor * src = गाठ->src[k];
        यदि (src) {
            ध्रुव size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    यदि (गाठ->op == GGML_OP_NONE && !(गाठ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        यदि (strlen(गाठ->name) == 0) {
            ggml_format_name(गाठ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = गाठ;
        cgraph->n_leafs++;
    } अन्यथा {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        यदि (strlen(गाठ->name) == 0) {
            ggml_format_name(गाठ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = गाठ;
        cgraph->n_nodes++;
    }

    प्रत्यागम गाठ_hash_pos;
}

स्थिर शून्य ggml_build_forward_impl(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor, bool expand, bool compute) {
    यदि (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ध्रुव पूर्णाङ्क n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    ध्रुव पूर्णाङ्क n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    यदि (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

संरचना ggml_tensor * ggml_build_forward_select(
        संरचना ggml_cgraph  * cgraph,
        संरचना ggml_tensor ** tensors,
        पूर्णाङ्क                   n_tensors,
        पूर्णाङ्क                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    प्रति (पूर्णाङ्क i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    प्रत्यागम tensors[idx];
}

शून्य ggml_build_forward_expand(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

शून्य ggml_build_forward_order(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

शून्य ggml_build_backward_expand(
        संरचना ggml_context *  ctx,
        संरचना ggml_cgraph  *  cgraph,
        संरचना ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    ध्रुव पूर्णाङ्क n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*परिमाण(संरचना ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*परिमाण(संरचना ggml_tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, परिमाण(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        प्रति (पूर्णाङ्क i = 0; i < n_nodes_f; ++i) {
            संरचना ggml_tensor * गाठ = cgraph->nodes[i];
            any_params = any_params || (गाठ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (गाठ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    प्रति (पूर्णाङ्क i = 0; i < n_nodes_f; ++i) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        यदि (गाठ->type == GGML_TYPE_I32) {
            अनुवर्त;
        }

        bool गाठ_needs_grad = (गाठ->flags & GGML_TENSOR_FLAG_PARAM) || (गाठ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        विकल्प (गाठ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            पक्ष GGML_OP_IM2COL:      // only used for its shape
            पक्ष GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                भङ्ग;
            पक्ष GGML_OP_UNARY: {
                ध्रुव गणना ggml_unary_op uop = ggml_get_unary_op(गाठ);
                // SGN and STEP unary ops are piecewise constant
                यदि (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } भङ्ग;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            पक्ष GGML_OP_CPY:           // gradients in CPY target are irrelevant
            पक्ष GGML_OP_GET_ROWS:      // row indices not differentiable
            पक्ष GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            पक्ष GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                भङ्ग;

            स्वभाव:
                भङ्ग;
        }
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; ++j) {
            यदि (!गाठ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, गाठ->src[j])]) {
                अनुवर्त;
            }
            GGML_ASSERT(गाठ->src[j]->type == GGML_TYPE_F32 || गाठ->src[j]->type == GGML_TYPE_F16);
            गाठ_needs_grad = true;
            भङ्ग;
        }
        यदि (!गाठ_needs_grad) {
            अनुवर्त;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!गाठ->view_src || गाठ->op == GGML_OP_CPY || गाठ->op == GGML_OP_SET_ROWS || गाठ->op == GGML_OP_VIEW ||
            गाठ->op == GGML_OP_RESHAPE || गाठ->op == GGML_OP_PERMUTE || गाठ->op == GGML_OP_TRANSPOSE);

        ध्रुव size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, गाठ);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        यदि (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } अन्यथा यदि (गाठ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, गाठ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    प्रति (पूर्णाङ्क i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    मुक्त(grads_needed);
}

स्थिर शून्य * incr_ptr_aligned(शून्य ** p, size_t size, size_t align) {
    शून्य * ptr = *p;
    ptr = (शून्य *) GGML_PAD((uintptr_t) ptr, align);
    *p = (शून्य *) ((वर्ण *) ptr + size);
    प्रत्यागम ptr;
}

स्थिर size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    शून्य * p = (वर्ण *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, परिमाण(संरचना ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * परिमाण(int32_t), परिमाण(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // hash keys
    यदि (grads) {
        incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * परिमाण(ggml_bitset_t), परिमाण(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    प्रत्यागम nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t size, bool grads) {
    प्रत्यागम GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(शून्य) {
    प्रत्यागम ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}

संरचना ggml_cgraph * ggml_new_graph_custom(संरचना ggml_context * ctx, size_t size, bool grads) {
    ध्रुव size_t obj_size = ggml_graph_nbytes(size, grads);
    संरचना ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    संरचना ggml_cgraph * cgraph = (संरचना ggml_cgraph *) ((वर्ण *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    शून्य * p = cgraph + 1;

    संरचना ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    संरचना ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * परिमाण(int32_t), परिमाण(int32_t));
    संरचना ggml_tensor ** hash_किल्ल्या_ptr  =         incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    संरचना ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)) : रिक्त;
    संरचना ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)) : रिक्त;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * परिमाण(ggml_bitset_t), परिमाण(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    प्रतिज्ञा(obj_size == (size_t)((वर्ण *)p - (वर्ण *)cgraph));

    *cgraph = (संरचना ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_किल्ल्या_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_डावा_TO_उजवा,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    यदि (grads) {
        memset(cgraph->grads,     0, hash_size*परिमाण(संरचना ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_size*परिमाण(संरचना ggml_tensor *));
    }

    प्रत्यागम cgraph;
}

संरचना ggml_cgraph * ggml_new_graph(संरचना ggml_context * ctx) {
    प्रत्यागम ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

संरचना ggml_cgraph ggml_graph_view(संरचना ggml_cgraph * cgraph0, पूर्णाङ्क i0, पूर्णाङ्क i1) {
    संरचना ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ रिक्त, // gradients would need visited_hash_set
        /*.grad_accs        =*/ रिक्त,
        /*.leafs            =*/ रिक्त,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    प्रत्यागम cgraph;
}

शून्य ggml_graph_cpy(संरचना ggml_cgraph * src, संरचना ggml_cgraph * dst) {
    GGML_ASSERT(dst->size >= src->n_leafs);
    GGML_ASSERT(dst->size >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    प्रति (पूर्णाङ्क i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    प्रति (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        यदि (ggml_bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = ggml_hash_घाला(&dst->visited_hash_set, src->visited_hash_set.किल्ल्या[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    यदि (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*परिमाण(संरचना ggml_tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*परिमाण(संरचना ggml_tensor *));
    }
    यदि (src->grads) {
        GGML_ASSERT(dst->grads     != रिक्त);
        GGML_ASSERT(dst->grad_accs != रिक्त);
        प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; ++i) {
            ध्रुव size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            ध्रुव size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

संरचना ggml_cgraph * ggml_graph_dup(संरचना ggml_context * ctx, संरचना ggml_cgraph * cgraph, bool force_grads) {
    संरचना ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    प्रत्यागम result;
}

संरचना ggml_tensor * ggml_set_zero(संरचना ggml_tensor * tensor) {
    यदि (ggml_is_empty(tensor)) {
        प्रत्यागम tensor;
    }
    यदि (tensor->buffer) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } अन्यथा {
        GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    प्रत्यागम tensor;
}

शून्य ggml_graph_reset(संरचना ggml_cgraph * cgraph) {
    यदि (!cgraph) {
        प्रत्यागम;
    }
    GGML_ASSERT(cgraph->grads != रिक्त);

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ     = cgraph->nodes[i];
        संरचना ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, गाठ);

        यदि (गाठ->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(गाठ->src[2]);
            ggml_set_zero(गाठ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        यदि (grad_acc) {
            यदि (गाठ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ध्रुव भिन्न onef = 1.0f;
                यदि (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, परिमाण(भिन्न));
                } अन्यथा {
                    GGML_ASSERT(grad_acc->data);
                    *((भिन्न *) grad_acc->data) = onef;
                }
            } अन्यथा {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

शून्य ggml_graph_clear(संरचना ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

पूर्णाङ्क ggml_graph_size(संरचना ggml_cgraph * cgraph) {
    प्रत्यागम cgraph->size;
}

संरचना ggml_tensor * ggml_graph_गाठ(संरचना ggml_cgraph * cgraph, पूर्णाङ्क i) {
    यदि (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        प्रत्यागम cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    प्रत्यागम cgraph->nodes[i];
}

संरचना ggml_tensor ** ggml_graph_nodes(संरचना ggml_cgraph * cgraph) {
    प्रत्यागम cgraph->nodes;
}

पूर्णाङ्क ggml_graph_n_nodes(संरचना ggml_cgraph * cgraph) {
    प्रत्यागम cgraph->n_nodes;
}

शून्य ggml_graph_add_गाठ(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

संरचना ggml_tensor * ggml_graph_get_tensor(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव वर्ण * name) {
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
        संरचना ggml_tensor * leaf = cgraph->leafs[i];

        यदि (strcmp(leaf->name, name) == 0) {
            प्रत्यागम leaf;
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        यदि (strcmp(गाठ->name, name) == 0) {
            प्रत्यागम गाठ;
        }
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_graph_get_grad(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * गाठ) {
    ध्रुव size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, गाठ);
    प्रत्यागम igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : रिक्त;
}

संरचना ggml_tensor * ggml_graph_get_grad_acc(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * गाठ) {
    ध्रुव size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, गाठ);
    प्रत्यागम igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : रिक्त;
}

शून्य ggml_graph_print(ध्रुव संरचना ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                गाठ->ne[0], गाठ->ne[1], गाठ->ne[2],
                ggml_op_name(गाठ->op), (गाठ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, गाठ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
        संरचना ggml_tensor * गाठ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                गाठ->ne[0], गाठ->ne[1],
                ggml_op_name(गाठ->op),
                ggml_get_name(गाठ));
    }

    GGML_LOG_INFO("========================================\n");
}

स्थिर पूर्णाङ्क ggml_गाठ_list_find_tensor(ध्रुव संरचना ggml_cgraph * cgraph,
                                      ध्रुव पूर्णाङ्क *                idxs,
                                      पूर्णाङ्क                        मोजणी,
                                      ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    प्रति (पूर्णाङ्क i = 0; i < मोजणी; ++i) {
        ध्रुव पूर्णाङ्क गाठ_idx = idxs[i];

        यदि (गाठ_idx >= cgraph->n_nodes) {
            प्रत्यागम -1;
        }
        यदि (cgraph->nodes[गाठ_idx] == tensor) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम -1;
}

स्थिर bool ggml_is_constant(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->buffer != रिक्त && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ध्रुव संरचना ggml_cgraph * cgraph,
                                ध्रुव पूर्णाङ्क *                गाठ_idxs,
                                पूर्णाङ्क                        मोजणी,
                                ध्रुव गणना ggml_op *       ops,
                                ध्रुव पूर्णाङ्क *                outputs,
                                पूर्णाङ्क                        अंक_outputs) {
    GGML_ASSERT(outputs && अंक_outputs > 0);

    प्रति (पूर्णाङ्क i = 0; i < मोजणी; ++i) {
        यदि (गाठ_idxs[i] >= cgraph->n_nodes) {
            प्रत्यागम false;
        }

        ध्रुव संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_idxs[i]];

        यदि (गाठ->op != ops[i]) {
            प्रत्यागम false;
        }

        यदि ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            प्रत्यागम false;
        }

        यदि (ggml_गाठ_list_find_tensor(cgraph, outputs, अंक_outputs, गाठ) != -1) {
            अनुवर्त;
        }

        यदि (गाठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            प्रत्यागम false;
        }

        पूर्णाङ्क subgraph_uses = 0;
        प्रति (पूर्णाङ्क j = i + 1; j < मोजणी; ++j) {
            ध्रुव संरचना ggml_tensor * other_गाठ = cgraph->nodes[गाठ_idxs[j]];
            प्रति (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                यदि (other_गाठ->src[src_idx] == गाठ) {
                    subgraph_uses++;
                }
            }
        }

        यदि (subgraph_uses != ggml_गाठ_get_use_मोजणी(cgraph, गाठ_idxs[i])) {
            प्रत्यागम false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        संरचना ggml_tensor * view_src = गाठ->view_src;
        यावत् (view_src) {
            यदि (ggml_गाठ_list_find_tensor(cgraph, गाठ_idxs, मोजणी, view_src) == -1 && !ggml_is_constant(view_src)) {
                प्रत्यागम false;
            }
            view_src = view_src->view_src;
        }
    }

    प्रत्यागम true;
}

// check if node is part of the graph
स्थिर bool ggml_graph_find(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * गाठ) {
    यदि (cgraph == रिक्त) {
        प्रत्यागम true;
    }

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        यदि (cgraph->nodes[i] == गाठ) {
            प्रत्यागम true;
        }
    }

    प्रत्यागम false;
}

स्थिर संरचना ggml_tensor * ggml_graph_get_parent(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * गाठ) {
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * parent = cgraph->nodes[i];
        संरचना ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        यदि (grad == गाठ) {
            प्रत्यागम parent;
        }
    }

    प्रत्यागम रिक्त;
}

स्थिर शून्य ggml_graph_dump_dot_गाठ_edge(FILE * fp, ध्रुव संरचना ggml_cgraph * gb, संरचना ggml_tensor * गाठ, संरचना ggml_tensor * parent, ध्रुव वर्ण * label)  {
    संरचना ggml_tensor * gparent = ggml_graph_get_parent(gb, गाठ);
    संरचना ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (शून्य *) gparent0 : (शून्य *) parent,
            gparent ? (शून्य *) gparent : (शून्य *) गाठ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

स्थिर शून्य ggml_graph_dump_dot_leaf_edge(FILE * fp, संरचना ggml_tensor * गाठ, संरचना ggml_tensor * parent, ध्रुव वर्ण * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (शून्य *) parent,
            (शून्य *) गाठ,
            label);
}

शून्य ggml_graph_dump_dot(ध्रुव संरचना ggml_cgraph * gb, ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव वर्ण * filename) {
    वर्ण color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    प्रति (पूर्णाङ्क i = 0; i < gb->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = gb->nodes[i];
        संरचना ggml_tensor * grad = ggml_graph_get_grad(gb, गाठ);

        यदि (ggml_graph_get_parent(gb, गाठ) != रिक्त) {
            अनुवर्त;
        }

        यदि (गाठ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, परिमाण(color), "yellow");
        } अन्यथा यदि (grad) {
            यदि (ggml_graph_find(cgraph, गाठ)) {
                snprintf(color, परिमाण(color), "green");
            } अन्यथा {
                snprintf(color, परिमाण(color), "lightblue");
            }
        } अन्यथा {
            snprintf(color, परिमाण(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (शून्य *) गाठ, color);

        यदि (strlen(गाठ->name) > 0) {
            fprintf(fp, "%s (%s)|", गाठ->name, ggml_type_name(गाठ->type));
        } अन्यथा {
            fprintf(fp, "(%s)|", ggml_type_name(गाठ->type));
        }

        यदि (ggml_is_matrix(गाठ)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, गाठ->ne[0], गाठ->ne[1], ggml_op_चिन्ह(गाठ->op));
        } अन्यथा {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, गाठ->ne[0], गाठ->ne[1], गाठ->ne[2], ggml_op_चिन्ह(गाठ->op));
        }

        यदि (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_चिन्ह(grad->op));
        } अन्यथा {
            fprintf(fp, "\"; ]\n");
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_leafs; i++) {
        संरचना ggml_tensor * गाठ = gb->leafs[i];

        snprintf(color, परिमाण(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (शून्य *) गाठ, color);

        यदि (strlen(गाठ->name) > 0) {
            fprintf(fp, "%s (%s)|", गाठ->name, ggml_type_name(गाठ->type));
        } अन्यथा {
            fprintf(fp, "(%s)|", ggml_type_name(गाठ->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, गाठ->ne[0], गाठ->ne[1]);
        यदि (ggml_nelements(गाठ) < 5 && गाठ->data != रिक्त) {
            fprintf(fp, " | (");
            प्रति (पूर्णाङ्क j = 0; j < ggml_nelements(गाठ); j++) {
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
                यदि (j < ggml_nelements(गाठ) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = gb->nodes[i];

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (गाठ->src[j]) {
                वर्ण label[16];
                snprintf(label, परिमाण(label), "src %d", j);
                ggml_graph_dump_dot_गाठ_edge(fp, gb, गाठ, गाठ->src[j], label);
            }
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_leafs; i++) {
        संरचना ggml_tensor * गाठ = gb->leafs[i];

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (गाठ->src[j]) {
                वर्ण label[16];
                snprintf(label, परिमाण(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, गाठ, गाठ->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

शून्य ggml_set_आदान(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_आदान;
}

शून्य ggml_set_output(संरचना ggml_tensor * tensor) {
    प्रति (संरचना ggml_tensor * cur = tensor; cur != रिक्त; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

शून्य ggml_set_param(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

शून्य ggml_set_loss(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

शून्य ggml_quantize_init(गणना ggml_type type) {
    ggml_critical_section_start();

    विकल्प (type) {
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); भङ्ग;
        स्वभाव: // nothing
            भङ्ग;
    }

    ggml_critical_section_end();
}

शून्य ggml_quantize_free(शून्य) {
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

bool ggml_quantize_requires_imatrix(गणना ggml_type type) {
    प्रत्यागम
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        गणना ggml_type   type,
           ध्रुव भिन्न * src,
                  शून्य * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           ध्रुव भिन्न * imatrix) {
    ध्रुव int64_t n = nrows * n_per_row;

    यदि (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != रिक्त);
    }

    GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ध्रुव size_t start_row = start / n_per_row;
    ध्रुव size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q2_0:    result = quantize_q2_0   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (वर्ण *) dst + start_row * row_size, nrows, n_per_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                size_t elemsize = परिमाण(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                size_t elemsize = परिमाण(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                size_t elemsize = परिमाण(भिन्न);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } भङ्ग;
        स्वभाव:
            प्रतिज्ञा(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    प्रत्यागम result;
}

////////////////////////////////////////////////////////////////////////////////

शून्य ggml_log_get(ggml_log_callback * log_callback, शून्य ** user_data) {
    *log_callback = g_logger_state.log_callback;
    *user_data    = g_logger_state.log_callback_user_data;
}

शून्य ggml_log_set(ggml_log_callback log_callback, शून्य * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

शून्य ggml_threadpool_params_init(संरचना ggml_threadpool_params * p, पूर्णाङ्क n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

संरचना ggml_threadpool_params ggml_threadpool_params_default(पूर्णाङ्क n_threads) {
    संरचना ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    प्रत्यागम p;
}

bool ggml_threadpool_params_match(ध्रुव संरचना ggml_threadpool_params * p0, ध्रुव संरचना ggml_threadpool_params * p1) {
    यदि (p0->n_threads  != p1->n_threads  ) प्रत्यागम false;
    यदि (p0->prio       != p1->prio       ) प्रत्यागम false;
    यदि (p0->poll       != p1->poll       ) प्रत्यागम false;
    यदि (p0->strict_cpu != p1->strict_cpu ) प्रत्यागम false;
    प्रत्यागम memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
