#परिभाषय _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#परिभाषय USE_MATH_परिभाषते // For M_PI on MSVC

#अन्तर्भाव "ggml-version.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-threading.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml.h"

// FIXME: required here for quantization functions
#अन्तर्भाव "ggml-quants.h"

#यदि_परिभाषितम् GGML_USE_CPU_HBM
#अन्तर्भाव <hbwmalloc.h>
#यदि_अन्तः

#यदि defined(_MSC_VER) || defined(__MINGW32__)
#अन्तर्भाव <malloc.h> // using malloc.h with MSC/MINGW
#नोचेद्यदि !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#अन्तर्भाव <alloca.h>
#यदि_अन्तः

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
#यदि_अन्तः

#यदि defined(__APPLE__)
#अन्तर्भाव <unistd.h>
#अन्तर्भाव <mach/mach.h>
#अन्तर्भाव <TargetConditionals.h>
#यदि_अन्तः

#यदि defined(_WIN32)
#परिभाषय WIN32_LEAN_AND_माध्यम्
#यदि_अपरिभाषितम् NOMINMAX
    #परिभाषय NOMINMAX
#यदि_अन्तः
#अन्तर्भाव <windows.h>
#यदि_अन्तः

#परिभाषय अप्रयुक्तम् GGML_अप्रयुक्तम्

uint64_t ggml_graph_अग्र_uid(शून्य) {
#यदि_परिभाषितम् _MSC_VER
#यदि defined(_WIN32)
    स्थिर चञ्चल LONG गणक = 1;
    प्रत्यागम (uint64_t) InterlockedIncrement(&गणक) - 1;
#अन्यथा
    स्थिर चञ्चल दीर्घ दीर्घ गणक = 1;
    प्रत्यागम (uint64_t) _InterlockedIncrement64(&गणक) - 1;
#यदि_अन्तः
#अन्यथा
    स्थिर uint64_t गणक = 1;
    प्रत्यागम __atomic_fetch_add(&गणक, 1, __ATOMIC_RELAXED);
#यदि_अन्तः
}

// Needed for ggml_fp32_to_bf16_row()
#यदि defined(__AVX512BF16__)
#यदि defined(_MSC_VER)
#परिभाषय m512i(p) p
#अन्यथा
#अन्तर्भाव <immintrin.h>
#परिभाषय m512i(p) (__m512i)(p)
#यदि_अन्तः // defined(_MSC_VER)
#यदि_अन्तः // defined(__AVX512BF16__)

#यदि defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#अन्तर्भाव <unistd.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/wait.h>
#यदि defined(__linux__)
#अन्तर्भाव <sys/prctl.h>
#यदि_अन्तः

#यदि defined(__ANDROID__)
#अन्तर्भाव <unwind.h>
#अन्तर्भाव <dlfcn.h>
#अन्तर्भाव <stdio.h>

संरचना backtrace_state {
    शून्य ** current;
    शून्य ** अन्तः_2;
};

स्थिर Unwind_Reason_संहिता unwind_callback(संरचना _Unwind_Context* context, शून्य* arg) {
    संरचना backtrace_state * state = (संरचना backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    यदि (pc) {
        यदि (state->current == state->अन्तः_2) {
            प्रत्यागम URC_अन्तः_सम्बन्धि_राशिः;
        } अन्यथा {
            *state->current++ = (शून्य*)pc;
        }
    }
    प्रत्यागम _URC_NO_REASON;
}

स्थिर शून्य ggml_print_backtrace_symbols(शून्य) {
    ध्रुव पूर्णाङ्क max = 100;
    शून्य* बफर[max];

    संरचना backtrace_state state = {बफर, बफर + max};
    _Unwind_Backtrace(unwind_callback, &state);

    पूर्णाङ्क गणनम् = state.current - बफर;

    प्रति (पूर्णाङ्क idx = 0; idx < गणनम्; ++idx) {
        ध्रुव शून्य * addr = बफर[idx];
        ध्रुव वर्ण * चिह्नम् = "";

        Dl_info info;
        यदि (dladdr(addr, &info) && info.dli_sname) {
            चिह्नम् = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, चिह्नम्);
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
#यदि_अन्तः

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
#यदि_अन्तः
#यदि defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t आकार = 0;
    वर्ण * पङ्क्तिः = रिक्त;
    ssize_t दीर्घता = 0;
    यावत् ((दीर्घता = getline(&पङ्क्तिः, &आकार, f)) > 0) {
        यदि (!वाक्ये_तुलय(पङ्क्तिः, "TracerPid:", परिमाण("TracerPid:") - 1) &&
            (दीर्घता != परिमाण("TracerPid:\t0\n") - 1 || पङ्क्तिः[दीर्घता - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            मुक्त(पङ्क्तिः);
            fclose(f);
            प्रत्यागम;
        }
    }
    मुक्त(पङ्क्तिः);
    fclose(f);
    पूर्णाङ्क तालकम्[2] = { -1, -1 };
    (शून्य) !pipe(तालकम्); // Don't start gdb until after PR_SET_PTRACER
#यदि_अन्तः
    ध्रुव पूर्णाङ्क parent_pid = getpid();
    ध्रुव पूर्णाङ्क child_pid = fork();
    यदि (child_pid < 0) { // error
#यदि defined(__linux__)
        close(तालकम्[1]);
        close(तालकम्[0]);
#यदि_अन्तः
        प्रत्यागम;
    } अन्यथा यदि (child_pid == 0) { // child
        वर्ण attach[32];
        snprintf(attach, परिमाण(attach), "attach %d", parent_pid);
#यदि defined(__linux__)
        close(तालकम्[1]);
        (शून्य) !पठ(तालकम्[0], तालकम्, 1);
        close(तालकम्[0]);
#यदि_अन्तः
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
        prctl(PR_समुच्चयः_PTRACER, child_pid);
        close(तालकम्[1]);
        close(तालकम्[0]);
#यदि_अन्तः
        waitpid(child_pid, रिक्त, 0);
    }
}
#अन्यथा
शून्य ggml_print_backtrace(शून्य) {
    // platform not supported
}
#यदि_अन्तः

स्थिर ggml_abort_callback_t g_abort_callback = रिक्त;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_समुच्चयः_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    प्रत्यागम ret_val;
}

शून्य ggml_abort(ध्रुव वर्ण * file, पूर्णाङ्क पङ्क्तिः, ध्रुव वर्ण * fmt, ...) {
    fflush(stdout);

    वर्ण सन्देशः[2048];
    पूर्णाङ्क विस्थापनम् = snprintf(सन्देशः, परिमाण(सन्देशः), "%s:%d: ", file, पङ्क्तिः);

    va_list तर्काः_2;
    va_start(तर्काः_2, fmt);
    vsnprintf(सन्देशः + विस्थापनम्, परिमाण(सन्देशः) - विस्थापनम्, fmt, तर्काः_2);
    va_end(तर्काः_2);

    यदि (g_abort_callback) {
        g_abort_callback(सन्देशः);
    } अन्यथा {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", सन्देशः);
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
    शून्य * log_callback_user_दत्त;
};
स्थिर संरचना ggml_logger_state g_logger_state = {ggml_log_callback_default, रिक्त};

स्थिर शून्य ggml_log_आन्तरिकम्_v(गणना ggml_log_level level, ध्रुव वर्ण * format, va_list तर्काः_2) {
    यदि (format == रिक्त) {
        प्रत्यागम;
    }
    va_list तर्काः_copy;
    va_copy(तर्काः_copy, तर्काः_2);
    वर्ण बफर[128];
    पूर्णाङ्क len = vsnprintf(बफर, 128, format, तर्काः_2);
    यदि (len < 128) {
        g_logger_state.log_callback(level, बफर, g_logger_state.log_callback_user_दत्त);
    } अन्यथा {
        वर्ण * buffer2 = (वर्ण *) शून्यस्मृतिं_देहि(len + 1, परिमाण(वर्ण));
        vsnprintf(buffer2, len + 1, format, तर्काः_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_दत्त);
        मुक्त(buffer2);
    }
    va_end(तर्काः_copy);
}

शून्य ggml_log_आन्तरिकम्(गणना ggml_log_level level, ध्रुव वर्ण * format, ...) {
    va_list तर्काः_2;
    va_start(तर्काः_2, format);
    ggml_log_आन्तरिकम्_v(level, format, तर्काः_2);
    va_end(तर्काः_2);
}

शून्य ggml_log_callback_default(गणना ggml_log_level level, ध्रुव वर्ण * text, शून्य * user_दत्त) {
    (शून्य) level;
    (शून्य) user_दत्त;
    fputs(text, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#यदि_परिभाषितम् GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#यदि_अन्तः


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#यदि defined(__has_feature)
#यदि __has_feature(thread_sanitizer)
#परिभाषय GGML_TSAN_ENABLED 1
#यदि_अन्तः
#अन्यथा
#यदि defined(__SANITIZE_THREAD__)
#परिभाषय GGML_TSAN_ENABLED 1
#यदि_अन्तः
#यदि_अन्तः

शून्य * ggml_aligned_malloc(size_t आकार) {
#यदि defined(__s390x__)
    ध्रुव पूर्णाङ्क alignment = 256;
#अन्यथा
    ध्रुव पूर्णाङ्क alignment = 64;
#यदि_अन्तः

#यदि defined(_MSC_VER) || defined(__MINGW32__)
    प्रत्यागम _aligned_malloc(आकार, alignment);
#अन्यथा
    यदि (आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * aligned_स्मृतिः = रिक्त;
  #यदि_परिभाषितम् GGML_USE_CPU_HBM
    पूर्णाङ्क परिणाम = hbw_posix_memalign(&aligned_स्मृतिः, alignment, आकार);
  #नोचेद्यदि TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_अप्रयुक्तम्(alignment);
    kern_return_t alloc_स्थितिः = vm_allocate((vm_map_t) mach_task_self(), (vm_स्थानाङ्कः_t *) &aligned_स्मृतिः, आकार, VM_FLAGS_ANYWHERE);
    पूर्णाङ्क परिणाम = EFAULT;
    विकल्प (alloc_स्थितिः) {
        पक्ष KERN_SUCCESS:
            परिणाम = 0;
            भङ्ग;
        पक्ष KERN_INVALID_स्थानाङ्कः:
            परिणाम = EINVAL;
            भङ्ग;
        पक्ष KERN_NO_SPACE:
            परिणाम = ENOMEM;
            भङ्ग;
        स्वभाव:
            परिणाम = EFAULT;
            भङ्ग;
    }
  #अन्यथा
    पूर्णाङ्क परिणाम = posix_memalign(&aligned_स्मृतिः, alignment, आकार);
  #यदि_अन्तः
    यदि (परिणाम != 0) {
        // Handle allocation failure
        ध्रुव वर्ण *error_desc = "unknown allocation error";
        विकल्प (परिणाम) {
            पक्ष EINVAL:
                error_desc = "invalid alignment value";
                भङ्ग;
            पक्ष ENOMEM:
                error_desc = "insufficient memory";
                भङ्ग;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, आकार/(1024.0*1024.0));
        प्रत्यागम रिक्त;
    }
    प्रत्यागम aligned_स्मृतिः;
#यदि_अन्तः
}

शून्य ggml_aligned_free(शून्य * ptr, size_t आकार) {
    GGML_अप्रयुक्तम्(आकार);
#यदि defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#नोचेद्यदि GGML_USE_CPU_HBM
    यदि (ptr != रिक्त) {
        hbw_free(ptr);
    }
#नोचेद्यदि TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    यदि (ptr != रिक्त) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_स्थानाङ्कः_t)ptr, आकार);
    }
#अन्यथा
    मुक्त(ptr);
#यदि_अन्तः
}


अन्तःस्थ स्थिर शून्य * ggml_malloc(size_t आकार) {
    यदि (आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * परिणाम = स्मृतिं_देहि(आकार);
    यदि (परिणाम == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, आकार/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    प्रत्यागम परिणाम;
}

// calloc
अन्तःस्थ स्थिर शून्य * ggml_calloc(size_t अङ्कः, size_t आकार) {
    यदि (अङ्कः == 0 || आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        प्रत्यागम रिक्त;
    }
    शून्य * परिणाम = शून्यस्मृतिं_देहि(अङ्कः, आकार);
    यदि (परिणाम == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, आकार/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    प्रत्यागम परिणाम;
}

#परिभाषय GGML_MALLOC(आकार)      ggml_malloc(आकार)
#परिभाषय GGML_CALLOC(अङ्कः, आकार) ggml_calloc(अङ्कः, आकार)

#परिभाषय GGML_FREE(ptr) मुक्त(ptr)

ध्रुव वर्ण * ggml_स्थितिः_गन्तव्यम्_string(गणना ggml_स्थितिः स्थितिः) {
    विकल्प (स्थितिः) {
        पक्ष GGML_स्थितिः_ALLOC_FAILED: प्रत्यागम "GGML status: error (failed to allocate memory)";
        पक्ष GGML_स्थितिः_FAILED:       प्रत्यागम "GGML status: error (operation failed)";
        पक्ष GGML_स्थितिः_SUCCESS:      प्रत्यागम "GGML status: success";
        पक्ष GGML_स्थितिः_ABORTED:      प्रत्यागम "GGML status: warning (operation aborted)";
    }

    प्रत्यागम "GGML status: unknown";
}

भिन्न ggml_fp16_गन्तव्यम्_fp32(ggml_fp16_t x) {
#परिभाषय ggml_fp16_गन्तव्यम्_fp32 do_not_use_ggml_fp16_गन्तव्यम्_fp32_अन्तर्_ggml
    प्रत्यागम GGML_FP16_गन्तव्यम्_FP32(x);
}

ggml_fp16_t ggml_fp32_गन्तव्यम्_fp16(भिन्न x) {
#परिभाषय ggml_fp32_गन्तव्यम्_fp16 do_not_use_ggml_fp32_गन्तव्यम्_fp16_अन्तर्_ggml
    प्रत्यागम GGML_FP32_गन्तव्यम्_FP16(x);
}

भिन्न ggml_bf16_गन्तव्यम्_fp32(ggml_bf16_t x) {
#परिभाषय ggml_bf16_गन्तव्यम्_fp32 do_not_use_ggml_bf16_गन्तव्यम्_fp32_अन्तर्_ggml
    प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_गन्तव्यम्_bf16(भिन्न x) {
#परिभाषय ggml_fp32_गन्तव्यम्_bf16 do_not_use_ggml_fp32_गन्तव्यम्_bf16_अन्तर्_ggml
    प्रत्यागम GGML_FP32_गन्तव्यम्_BF16(x);
}

शून्य ggml_fp16_गन्तव्यम्_fp32_row(ध्रुव ggml_fp16_t * x, भिन्न * y, int64_t n) {
    प्रति (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_गन्तव्यम्_FP32(x[i]);
    }
}

शून्य ggml_fp32_गन्तव्यम्_fp16_row(ध्रुव भिन्न * x, ggml_fp16_t * y, int64_t n) {
    पूर्णाङ्क i = 0;
    प्रति (; i < n; ++i) {
        y[i] = GGML_FP32_गन्तव्यम्_FP16(x[i]);
    }
}

शून्य ggml_bf16_गन्तव्यम्_fp32_row(ध्रुव ggml_bf16_t * x, भिन्न * y, int64_t n) {
    पूर्णाङ्क i = 0;
    प्रति (; i < n; ++i) {
        y[i] = GGML_BF16_गन्तव्यम्_FP32(x[i]);
    }
}

शून्य ggml_fp32_गन्तव्यम्_bf16_row_ref(ध्रुव भिन्न * x, ggml_bf16_t * y, int64_t n) {
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_गन्तव्यम्_bf16(x[i]);
    }
}

शून्य ggml_fp32_गन्तव्यम्_bf16_row(ध्रुव भिन्न * x, ggml_bf16_t * y, int64_t n) {
  पूर्णाङ्क i = 0;
#यदि defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  प्रति (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#यदि_अन्तः
    प्रति (; i < n; i++) {
        y[i] = GGML_FP32_गन्तव्यम्_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    प्रत्यागम memcmp(guid_a, guid_b, परिमाण(ggml_guid)) == 0;
}

ध्रुव वर्ण * ggml_संस्करणम्(शून्य) {
    प्रत्यागम GGML_संस्करणम्;
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
    अप्रयुक्तम्(once);
    अप्रयुक्तम्(param);
    अप्रयुक्तम्(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    Query_Performance_गणक(&t);
    timer_start = t.QuadPart;

    प्रत्यागम TRUE;
}
शून्य ggml_time_init(शून्य) {
    स्थिर INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, रिक्त, रिक्त);
}
int64_t ggml_time_ms(शून्य) {
    LARGE_INTEGER t;
    Query_Performance_गणक(&t);
    प्रत्यागम ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(शून्य) {
    LARGE_INTEGER t;
    Query_Performance_गणक(&t);
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
#यदि_अन्तः

int64_t ggml_आवृत्तयः(शून्य) {
    प्रत्यागम clock();
}

int64_t ggml_आवृत्तयः_प्रत्येकम्_ms(शून्य) {
    प्रत्यागम CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#यदि_परिभाषितम् _WIN32
स्थिर wchar_t * ggml_mbstowcs(ध्रुव वर्ण * mbs) {
    पूर्णाङ्क wlen = Multi_Byte_गन्तव्यम्_Wide_Char(CP_UTF8, 0, mbs, -1, रिक्त, 0);
    यदि (!wlen) {
        errno = EINVAL;
        प्रत्यागम रिक्त;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * परिमाण(wchar_t));
    wlen = Multi_Byte_गन्तव्यम्_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    यदि (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        प्रत्यागम रिक्त;
    }

    प्रत्यागम wbuf;
}
#यदि_अन्तः

FILE * ggml_fopen(ध्रुव वर्ण * fname, ध्रुव वर्ण * बहुलकम्) {
#यदि_परिभाषितम् _WIN32
    FILE * file = रिक्त;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    यदि (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(बहुलकम्);
        यदि (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    प्रत्यागम file;
#अन्यथा
    प्रत्यागम fopen(fname, बहुलकम्);
#यदि_अन्तः

}

स्थिर ध्रुव संरचना ggml_type_traits type_traits[GGML_TYPE_गणनम्] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(द्विगुण),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(भिन्न),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(ggml_fp16_t),
        .is_quantized             = false,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) ggml_fp16_गन्तव्यम्_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_गन्तव्यम्_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_आकार                = QK1_0,
        .type_आकार                = परिमाण(खण्डम्_q1_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_आकार                = QK2_0,
        .type_आकार                = परिमाण(खण्डम्_q2_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_आकार                = QK4_0,
        .type_आकार                = परिमाण(खण्डम्_q4_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_आकार                = QK4_1,
        .type_आकार                = परिमाण(खण्डम्_q4_1),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_आकार                = QK5_0,
        .type_आकार                = परिमाण(खण्डम्_q5_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_आकार                = QK5_1,
        .type_आकार                = परिमाण(खण्डम्_q5_1),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_आकार                = QK8_0,
        .type_आकार                = परिमाण(खण्डम्_q8_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_आकार                = QK8_1,
        .type_आकार                = परिमाण(खण्डम्_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_आकार                = QK_MXFP4,
        .type_आकार                = परिमाण(खण्डम्_mxfp4),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_आकार                = QK_NVFP4,
        .type_आकार                = परिमाण(खण्डम्_nvfp4),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q2_K),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q3_K),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q4_K),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q5_K),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q6_K),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq2_xxs),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq2_xs),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq3_xxs),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq3_s),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq2_s),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq1_s),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq1_s,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq1_m),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq1_m,
        .from_float_ref           = रिक्त,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_आकार                = QK4_NL,
        .type_आकार                = परिमाण(खण्डम्_iq4_nl),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_iq4_xs),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_आकार                = 1,
        .type_आकार                = परिमाण(ggml_bf16_t),
        .is_quantized             = false,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) ggml_bf16_गन्तव्यम्_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_गन्तव्यम्_bf16_row_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_tq1_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_आकार                = QK_K,
        .type_आकार                = परिमाण(खण्डम्_tq2_0),
        .is_quantized             = true,
        .गन्तव्यम्_float                 = (ggml_गन्तव्यम्_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_आकार                = 0,
        .type_आकार                = 0,
        .is_quantized             = false,
    },
};

ध्रुव संरचना ggml_type_traits * ggml_get_type_traits(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम &type_traits[type];
}

//
// ggml object
//

संरचना ggml_object {
    size_t offs;
    size_t आकार;

    संरचना ggml_object * अग्र;

    गणना ggml_object_type type;

    वर्ण padding[4];
};

स्थिर ध्रुव size_t GGML_OBJECT_आकार = परिमाण(संरचना ggml_object);

//
// ggml context
//

संरचना ggml_context {
    size_t mem_आकार;
    शून्य * mem_बफर;
    bool   mem_बफर_owned;
    bool   no_alloc;

    पूर्णाङ्क    n_objects;

    संरचना ggml_object * objects_begin;
    संरचना ggml_object * objects_अन्तः;
};

//
// data types
//

स्थिर ध्रुव वर्ण * GGML_OP_NAME[GGML_OP_गणनम्] = {
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

static_assert(GGML_OP_गणनम् == 102, "GGML_OP_COUNT != 102");

स्थिर ध्रुव वर्ण * GGML_OP_चिह्नम्[GGML_OP_गणनम्] = {
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

static_assert(GGML_OP_गणनम् == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_गणनम् == 2, "GGML_OP_POOL_COUNT != 2");

स्थिर ध्रुव वर्ण * GGML_UNARY_OP_NAME[GGML_UNARY_OP_गणनम्] = {
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

static_assert(GGML_UNARY_OP_गणनम् == 22, "GGML_UNARY_OP_COUNT != 22");

स्थिर ध्रुव वर्ण * GGML_GLU_OP_NAME[GGML_GLU_OP_गणनम्] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_गणनम् == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(परिमाण(संरचना ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(परिमाण(संरचना ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

शून्य ggml_print_object(ध्रुव संरचना ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->आकार, (ध्रुव शून्य *) obj->अग्र);
}

शून्य ggml_print_objects(ध्रुव संरचना ggml_context * ctx) {
    संरचना ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (ध्रुव शून्य *) ctx);

    यावत् (obj != रिक्त) {
        ggml_print_object(obj);
        obj = obj->अग्र;
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
    ध्रुव size_t blck_आकार = ggml_blck_आकार(tensor->type);
    यदि (blck_आकार == 1) {
        nbytes = ggml_type_आकार(tensor->type);
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    अन्यथा {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_आकार;
        प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    प्रत्यागम nbytes;
}

size_t ggml_nbytes_pad(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_आकार(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम type_traits[type].blck_आकार;
}

size_t ggml_type_आकार(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम type_traits[type].type_आकार;
}

size_t ggml_row_आकार(गणना ggml_type type, int64_t ne) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रतिज्ञा(ne % ggml_blck_आकार(type) == 0);
    प्रत्यागम ggml_type_आकार(type)*ne/ggml_blck_आकार(type);
}

द्विगुण ggml_type_sizef(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम ((द्विगुण)(type_traits[type].type_आकार))/type_traits[type].blck_आकार;
}

ध्रुव वर्ण * ggml_type_name(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम type_traits[type].type_name;
}

bool ggml_is_quantized(गणना ggml_type type) {
    प्रतिज्ञा(type >= 0);
    प्रतिज्ञा(type < GGML_TYPE_गणनम्);
    प्रत्यागम type_traits[type].is_quantized;
}

ध्रुव वर्ण * ggml_op_name(गणना ggml_op op) {
    प्रत्यागम GGML_OP_NAME[op];
}

ध्रुव वर्ण * ggml_op_चिह्नम्(गणना ggml_op op) {
    प्रत्यागम GGML_OP_चिह्नम्[op];
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

size_t ggml_element_आकार(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_type_आकार(tensor->type);
}

bool ggml_is_scalar(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_सदिशम्(ध्रुव संरचना ggml_tensor * tensor) {
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

गणना ggml_type ggml_ftype_गन्तव्यम्_ggml_type(गणना ggml_ftype ftype) {
    गणना ggml_type wtype = GGML_TYPE_गणनम्;

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
        पक्ष GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_गणनम्; भङ्ग;
        पक्ष GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_गणनम्; भङ्ग;
    }

    GGML_ASSERT(wtype != GGML_TYPE_गणनम्);

    प्रत्यागम wtype;
}

size_t ggml_tensor_overhead(शून्य) {
    प्रत्यागम GGML_OBJECT_आकार + GGML_TENSOR_आकार;
}

bool ggml_is_transposed(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->nb[0] > tensor->nb[1];
}

स्थिर bool ggml_is_contiguous_m_n(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क m, पूर्णाङ्क n) {
    size_t अग्र_nb = ggml_type_आकार(tensor->type);
    यदि (tensor->ne[0] != ggml_blck_आकार(tensor->type) && tensor->nb[0] != अग्र_nb) {
        प्रत्यागम false;
    }
    अग्र_nb *= tensor->ne[0]/ggml_blck_आकार(tensor->type);
    प्रति (पूर्णाङ्क i = 1; i < n; i++) {
        यदि (i > m) {
            यदि (tensor->ne[i] != 1 && tensor->nb[i] != अग्र_nb) {
                प्रत्यागम false;
            }
            अग्र_nb *= tensor->ne[i];
        } अन्यथा {
            // this dimension does not need to be contiguous
            अग्र_nb = tensor->ne[i]*tensor->nb[i];
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

bool ggml_is_contiguous_गन्तव्यम्_1(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_गन्तव्यम्_2(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_गन्तव्यम्_3(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_आकार(tensor->type)/ggml_blck_आकार(tensor->type);
}

bool ggml_is_permuted(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_आकार(tensor->type);
}

bool ggml_is_contiguous_पङ्क्तिकाः(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम
        tensor->ne[0] == ggml_blck_आकार(tensor->type) ||
        tensor->nb[0] == ggml_type_आकार(tensor->type);
}

स्थिर अन्तःस्थ bool ggml_is_padded_1d(ध्रुव संरचना ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम
        tensor->nb[0] == ggml_type_आकार(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_रिक्तानि(ध्रुव संरचना ggml_tensor * tensor) {
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

    प्रत्यागम ggml_is_रिक्तानि(t0) ? ggml_is_रिक्तानि(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

स्थिर अन्तःस्थ bool ggml_can_repeat_पङ्क्तिकाः(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#परिभाषय GGML_ASSERT_ALIGNED(ptr) \
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

    ggml_critical_section_अन्तः();

    संरचना ggml_context * ctx = GGML_MALLOC(परिमाण(संरचना ggml_context));

    // allow to call ggml_init with 0 size
    यदि (params.mem_आकार == 0) {
        params.mem_आकार = GGML_MEM_ALIGN;
    }

    ध्रुव size_t mem_आकार = params.mem_बफर ? params.mem_आकार : GGML_PAD(params.mem_आकार, GGML_MEM_ALIGN);

    *ctx = (संरचना ggml_context) {
        /*.mem_size           =*/ mem_आकार,
        /*.mem_buffer         =*/ params.mem_बफर ? params.mem_बफर : ggml_aligned_malloc(mem_आकार),
        /*.mem_buffer_owned   =*/ params.mem_बफर ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ रिक्त,
        /*.objects_end        =*/ रिक्त,
    };

    GGML_ASSERT(ctx->mem_बफर != रिक्त);

    GGML_ASSERT_ALIGNED(ctx->mem_बफर);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    प्रत्यागम ctx;
}

शून्य ggml_reset(संरचना ggml_context * ctx) {
    यदि (ctx == रिक्त) {
        प्रत्यागम;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = रिक्त;
    ctx->objects_अन्तः   = रिक्त;
}

शून्य ggml_free(संरचना ggml_context * ctx) {
    यदि (ctx == रिक्त) {
        प्रत्यागम;
    }

    यदि (ctx->mem_बफर_owned) {
        ggml_aligned_free(ctx->mem_बफर, ctx->mem_आकार);
    }

    GGML_FREE(ctx);
}

size_t ggml_प्रयुक्तम्_mem(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->objects_अन्तः == रिक्त ? 0 : ctx->objects_अन्तः->offs + ctx->objects_अन्तः->आकार;
}

bool ggml_get_no_alloc(संरचना ggml_context * ctx) {
    प्रत्यागम ctx->no_alloc;
}

शून्य ggml_समुच्चयः_no_alloc(संरचना ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

शून्य * ggml_get_mem_बफर(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->mem_बफर;
}

size_t ggml_get_mem_आकार(ध्रुव संरचना ggml_context * ctx) {
    प्रत्यागम ctx->mem_आकार;
}

size_t ggml_get_max_tensor_आकार(ध्रुव संरचना ggml_context * ctx) {
    size_t max_आकार = 0;

    प्रति (संरचना ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != रिक्त; tensor = ggml_get_अग्र_tensor(ctx, tensor)) {
        size_t अष्टकानि = ggml_nbytes(tensor);
        max_आकार = MAX(max_आकार, अष्टकानि);
    }

    प्रत्यागम max_आकार;
}

////////////////////////////////////////////////////////////////////////////////

स्थिर संरचना ggml_object * ggml_new_object(संरचना ggml_context * ctx, गणना ggml_object_type type, size_t आकार) {
    // always insert objects at the end of the context's memory pool
    संरचना ggml_object * obj_cur = ctx->objects_अन्तः;

    ध्रुव size_t cur_offs = obj_cur == रिक्त ? 0 : obj_cur->offs;
    ध्रुव size_t cur_आकार = obj_cur == रिक्त ? 0 : obj_cur->आकार;
    ध्रुव size_t cur_अन्तः  = cur_offs + cur_आकार;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(आकार <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t आकार_needed = GGML_PAD(आकार, GGML_MEM_ALIGN);

    वर्ण * ध्रुव mem_बफर = ctx->mem_बफर;
    संरचना ggml_object * ध्रुव obj_new = (संरचना ggml_object *)(mem_बफर + cur_अन्तः);

    // integer overflow checks
    यदि (cur_अन्तः > SIZE_MAX - आकार_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_अन्तः, आकार_needed);
        प्रत्यागम रिक्त;
    }
    यदि (cur_अन्तः + आकार_needed > SIZE_MAX - GGML_OBJECT_आकार) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_अन्तः, आकार_needed, (size_t) GGML_OBJECT_आकार);
        प्रत्यागम रिक्त;
    }

    यदि (cur_अन्तः + आकार_needed + GGML_OBJECT_आकार > ctx->mem_आकार) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_अन्तः + आकार_needed + GGML_OBJECT_आकार, ctx->mem_आकार);
#यदि_अपरिभाषितम् NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#यदि_अन्तः
        प्रत्यागम रिक्त;
    }

    *obj_new = (संरचना ggml_object) {
        .offs = cur_अन्तः + GGML_OBJECT_आकार,
        .आकार = आकार_needed,
        .अग्र = रिक्त,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_बफर + obj_new->offs);

    यदि (obj_cur != रिक्त) {
        obj_cur->अग्र = obj_new;
    } अन्यथा {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_अन्तः = obj_new;

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

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_गणनम्);
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

    size_t दत्त_आकार = ggml_row_आकार(type, ne[0]);
    प्रति (पूर्णाङ्क i = 1; i < n_dims; i++) {
        दत्त_आकार *= ne[i];
    }

    GGML_ASSERT(view_src == रिक्त || दत्त_आकार == 0 || दत्त_आकार + view_offs <= ggml_nbytes(view_src));

    शून्य * दत्त = view_src != रिक्त ? view_src->दत्त : रिक्त;
    यदि (दत्त != रिक्त) {
        दत्त = (वर्ण *) दत्त + view_offs;
    }

    size_t obj_alloc_आकार = 0;

    यदि (view_src == रिक्त && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_आकार = दत्त_आकार;
    }

    GGML_ASSERT(GGML_TENSOR_आकार <= SIZE_MAX - obj_alloc_आकार);

    संरचना ggml_object * ध्रुव obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_आकार + obj_alloc_आकार);
    GGML_ASSERT(obj_new);

    संरचना ggml_tensor * ध्रुव परिणाम = (संरचना ggml_tensor *)((वर्ण *)ctx->mem_बफर + obj_new->offs);

    *परिणाम = (संरचना ggml_tensor) {
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
        /*.data         =*/ obj_alloc_आकार > 0 ? (शून्य *)(परिणाम + 1) : दत्त,
        /*.name         =*/ { 0 },
        /*.extra        =*/ रिक्त,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    प्रति (पूर्णाङ्क i = 0; i < n_dims; i++) {
        परिणाम->ne[i] = ne[i];
    }

    परिणाम->nb[0] = ggml_type_आकार(type);
    परिणाम->nb[1] = परिणाम->nb[0]*(परिणाम->ne[0]/ggml_blck_आकार(type));
    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = परिणाम->nb[i - 1]*परिणाम->ne[i - 1];
    }

    ctx->n_objects++;

    प्रत्यागम परिणाम;
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

शून्य * ggml_new_बफर(संरचना ggml_context * ctx, size_t nbytes) {
    संरचना ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_बफर, nbytes);

    प्रत्यागम (uint8_t *)ctx->mem_बफर + obj->offs;
}

संरचना ggml_tensor * ggml_dup_tensor(संरचना ggml_context * ctx, ध्रुव संरचना ggml_tensor * src) {
    प्रत्यागम ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

शून्य ggml_unravel_सूचकांक(ध्रुव संरचना ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
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

शून्य * ggml_get_दत्त(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->दत्त;
}

भिन्न * ggml_get_दत्त_f32(ध्रुव संरचना ggml_tensor * tensor) {
    प्रतिज्ञा(tensor->type == GGML_TYPE_F32);
    प्रत्यागम (भिन्न *)(tensor->दत्त);
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

संरचना ggml_tensor * ggml_समुच्चयः_name(संरचना ggml_tensor * tensor, ध्रुव वर्ण * name) {
    size_t i;
    प्रति (i = 0; i < परिमाण(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    प्रत्यागम tensor;
}

संरचना ggml_tensor * ggml_format_name(संरचना ggml_tensor * tensor, ध्रुव वर्ण * fmt, ...) {
    va_list तर्काः_2;
    va_start(तर्काः_2, fmt);
    vsnprintf(tensor->name, परिमाण(tensor->name), fmt, तर्काः_2);
    va_end(तर्काः_2);
    प्रत्यागम tensor;
}

संरचना ggml_tensor * ggml_view_tensor(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * src) {
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(परिणाम, "%s (view)", src->name);

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = src->nb[i];
    }

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_get_first_tensor(ध्रुव संरचना ggml_context * ctx) {
    संरचना ggml_object * obj = ctx->objects_begin;

    वर्ण * ध्रुव mem_बफर = ctx->mem_बफर;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            प्रत्यागम (संरचना ggml_tensor *)(mem_बफर + obj->offs);
        }

        obj = obj->अग्र;
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_get_अग्र_tensor(ध्रुव संरचना ggml_context * ctx, संरचना ggml_tensor * tensor) {
    संरचना ggml_object * obj = (संरचना ggml_object *) ((वर्ण *)tensor - GGML_OBJECT_आकार);
    obj = obj->अग्र;

    वर्ण * ध्रुव mem_बफर = ctx->mem_बफर;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            प्रत्यागम (संरचना ggml_tensor *)(mem_बफर + obj->offs);
        }

        obj = obj->अग्र;
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_get_tensor(संरचना ggml_context * ctx, ध्रुव वर्ण * name) {
    संरचना ggml_object * obj = ctx->objects_begin;

    वर्ण * ध्रुव mem_बफर = ctx->mem_बफर;

    यावत् (obj != रिक्त) {
        यदि (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            संरचना ggml_tensor * cur = (संरचना ggml_tensor *)(mem_बफर + obj->offs);
            यदि (वाक्यं_तुलय(cur->name, name) == 0) {
                प्रत्यागम cur;
            }
        }

        obj = obj->अग्र;
    }

    प्रत्यागम रिक्त;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

स्थिर संरचना ggml_tensor * ggml_dup_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_DUP;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

स्थिर संरचना ggml_tensor * ggml_सङ्कलय_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_सङ्कलय;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_सङ्कलय(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_सङ्कलय_impl(ctx, a, b, false);
}

संरचना ggml_tensor * ggml_सङ्कलय_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_सङ्कलय_impl(ctx, a, b, true);
}

// ggml_add_cast

स्थिर संरचना ggml_tensor * ggml_सङ्कलय_cast_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_पङ्क्तिकाः(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    परिणाम->op     = GGML_OP_सङ्कलय;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_सङ्कलय_cast(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        गणना   ggml_type      type) {
    प्रत्यागम ggml_सङ्कलय_cast_impl(ctx, a, b, type);
}

संरचना ggml_tensor * ggml_सङ्कलय_id(
            संरचना ggml_context * ctx,
            संरचना ggml_tensor  * a,
            संरचना ggml_tensor  * b,
            संरचना ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_सङ्कलय_ID;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = ids;

    प्रत्यागम परिणाम;
}

// ggml_add1

स्थिर संरचना ggml_tensor * ggml_add1_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_ADD1;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
        size_t                विस्थापनम्,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(विस्थापनम् < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, विस्थापनम्, inplace ? 1 : 0 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_ACC;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_acc(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, विस्थापनम्, false);
}

संरचना ggml_tensor * ggml_acc_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, विस्थापनम्, true);
}

// ggml_sub

स्थिर संरचना ggml_tensor * ggml_sub_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SUB;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_MUL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_DIV;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SQR;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SQRT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_LOG;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SIN;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_COS;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

संरचना ggml_tensor * ggml_योग(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, a->type, 1);

    परिणाम->op     = GGML_OP_योग;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_sum_rows

संरचना ggml_tensor * ggml_योग_पङ्क्तिकाः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    परिणाम->op     = GGML_OP_योग_पङ्क्तिकाः;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_cumsum

संरचना ggml_tensor * ggml_cumsum(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_CUMSUM;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_mean

संरचना ggml_tensor * ggml_माध्यम्(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_माध्यम्;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_argmax

संरचना ggml_tensor * ggml_argmax(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    परिणाम->op     = GGML_OP_ARGMAX;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_count_equal

संरचना ggml_tensor * ggml_गणनम्_equal(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    परिणाम->op     = GGML_OP_गणनम्_EQUAL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_repeat

संरचना ggml_tensor * ggml_repeat(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    परिणाम->op     = GGML_OP_REPEAT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_repeat_4d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ध्रुव bool can_repeat = ggml_is_रिक्तानि(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    परिणाम->op     = GGML_OP_REPEAT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_repeat_back

संरचना ggml_tensor * ggml_repeat_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    परिणाम->op     = GGML_OP_REPEAT_पृष्ठभागः;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, dim);

    परिणाम->op     = GGML_OP_CONCAT;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

संरचना ggml_tensor * ggml_चरणः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_चरणः);
}

संरचना ggml_tensor * ggml_चरणः_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_चरणः);
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
        भिन्न                 negative_नतिः,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params(परिणाम, &negative_नतिः, परिमाण(negative_नतिः));

    परिणाम->op     = GGML_OP_LEAKY_RELU;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

संरचना ggml_tensor * ggml_gelu_शीघ्र(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_GELU_शीघ्र);
}

संरचना ggml_tensor * ggml_gelu_शीघ्र_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_शीघ्र);
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
    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_समुच्चयः_op_params_f32(परिणाम, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_समुच्चयः_op_params_f32(परिणाम, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_समुच्चयः_op_params_f32(परिणाम, 3, beta);
    ggml_समुच्चयः_op_params_f32(परिणाम, 4, eps);

    परिणाम->op     = GGML_OP_UNARY;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_silu_back

संरचना ggml_tensor * ggml_silu_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SILU_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, रिक्त, 0);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) op);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, (int32_t) swapped);

    परिणाम->op     = GGML_OP_GLU;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

संरचना ggml_tensor * ggml_आवर्तनम्(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary(ctx, a, GGML_UNARY_OP_आवर्तनम्);
}

संरचना ggml_tensor * ggml_आवर्तनम्_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_unary_inplace(ctx, a, GGML_UNARY_OP_आवर्तनम्);
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

संरचना ggml_tensor * ggml_glu_विभज(
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

संरचना ggml_tensor * ggml_reglu_विभज(
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

संरचना ggml_tensor * ggml_geglu_विभज(
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

संरचना ggml_tensor * ggml_swiglu_विभज(
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

संरचना ggml_tensor * ggml_geglu_erf_विभज(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

संरचना ggml_tensor * ggml_geglu_शीघ्र(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_शीघ्र, false);
}

संरचना ggml_tensor * ggml_geglu_शीघ्र_swapped(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    प्रत्यागम ggml_glu_impl(ctx, a, रिक्त, GGML_GLU_OP_GEGLU_शीघ्र, true);
}

संरचना ggml_tensor * ggml_geglu_शीघ्र_विभज(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    प्रत्यागम ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_शीघ्र, false);
}

संरचना ggml_tensor * ggml_swiglu_oai(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 alpha,
        भिन्न                 सीमा) {
    संरचना ggml_tensor * परिणाम = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_समुच्चयः_op_params_f32(परिणाम, 2, alpha);
    ggml_समुच्चयः_op_params_f32(परिणाम, 3, सीमा);

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_swiglu_clamp(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 सीमा) {
    संरचना ggml_tensor * परिणाम = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_समुच्चयः_op_params_f32(परिणाम, 3, सीमा);

    प्रत्यागम परिणाम;
}

// ggml_norm

स्थिर संरचना ggml_tensor * ggml_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params(परिणाम, &eps, परिमाण(eps));

    परिणाम->op     = GGML_OP_NORM;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params(परिणाम, &eps, परिमाण(eps));

    परिणाम->op     = GGML_OP_RMS_NORM;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

संरचना ggml_tensor * ggml_rms_norm_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 eps) {
    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params(परिणाम, &eps, परिमाण(eps));

    परिणाम->op     = GGML_OP_RMS_NORM_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_group_norm

स्थिर संरचना ggml_tensor * ggml_गणः_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, n_groups);
    ggml_समुच्चयः_op_params_f32(परिणाम, 1, eps);

    परिणाम->op     = GGML_OP_गणः_NORM;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_गणः_norm(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps) {
    प्रत्यागम ggml_गणः_norm_impl(ctx, a, n_groups, eps, false);
}

संरचना ggml_tensor * ggml_गणः_norm_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_groups,
        भिन्न                 eps) {
    प्रत्यागम ggml_गणः_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

स्थिर संरचना ggml_tensor * ggml_l2_norm_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 eps,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, eps);

    परिणाम->op     = GGML_OP_L2_NORM;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

bool ggml_prec_समुच्चयः_acc(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    विकल्प (a->op) {
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ध्रुव int32_t prec_i32 = (int32_t) prec;
                ggml_समुच्चयः_op_params_i32(a, 0, prec_i32);
            }
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ध्रुव int32_t prec_i32 = (int32_t) prec;
                ggml_समुच्चयः_op_params_i32(a, 3, prec_i32);
            }
            भङ्ग;
        स्वभाव:
            प्रत्यागम false;
    };

    प्रत्यागम true;
}

bool ggml_prec_समुच्चयः_src(
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
                ggml_समुच्चयः_op_params_i32(a, 2 + idx, prec_i32);
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_MUL_MAT;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

शून्य ggml_mul_mat_समुच्चयः_prec(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ध्रुव int32_t prec_i32 = (int32_t) prec;

    ggml_समुच्चयः_op_params_i32(a, 0, prec_i32);
}

शून्य ggml_mul_mat_समुच्चयः_hint(
        संरचना ggml_tensor * a,
        गणना ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    ध्रुव int32_t hint_i32 = (int32_t) hint;

    ggml_समुच्चयः_op_params_i32(a, 1, hint_i32);
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_MUL_MAT_ID;
    परिणाम->src[0] = as;
    परिणाम->src[1] = b;
    परिणाम->src[2] = ids;

    प्रत्यागम परिणाम;
}

// ggml_out_prod

स्थिर अन्तःस्थ bool ggml_can_बहिर्गमनम्_prod(ध्रुव संरचना ggml_tensor * t0, ध्रुव संरचना ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

संरचना ggml_tensor * ggml_बहिर्गमनम्_prod(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_बहिर्गमनम्_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    ध्रुव int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_बहिर्गमनम्_PROD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_scale

स्थिर संरचना ggml_tensor * ggml_scale_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 s,
        भिन्न                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[2] = { s, b };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op     = GGML_OP_SCALE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

स्थिर संरचना ggml_tensor * ggml_समुच्चयः_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                विस्थापनम्,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(विस्थापनम् < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, विस्थापनम्, inplace ? 1 : 0 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_समुच्चयः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_समुच्चयः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, nb1, nb2, nb3, विस्थापनम्, false);
}

संरचना ggml_tensor * ggml_समुच्चयः_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, nb1, nb2, nb3, विस्थापनम्, true);
}

संरचना ggml_tensor * ggml_समुच्चयः_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], विस्थापनम्, false);
}

संरचना ggml_tensor * ggml_समुच्चयः_1d_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], विस्थापनम्, true);
}

संरचना ggml_tensor * ggml_समुच्चयः_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], विस्थापनम्, false);
}

संरचना ggml_tensor * ggml_समुच्चयः_2d_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        size_t                nb1,
        size_t                विस्थापनम्) {
    प्रत्यागम ggml_समुच्चयः_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], विस्थापनम्, true);
}

// ggml_cpy

स्थिर संरचना ggml_tensor * ggml_cpy_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, b);
    यदि (वाक्यदैर्घ्यम्(b->name) > 0) {
        ggml_format_name(परिणाम, "%s (copy of %s)", b->name, a->name);
    } अन्यथा {
        ggml_format_name(परिणाम, "%s (copy)", a->name);
    }

    परिणाम->op     = GGML_OP_CPY;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(परिणाम, "%s (copy)", a->name);

    परिणाम->op     = GGML_OP_CPY;
    परिणाम->src[0] = a;
    परिणाम->src[1] = परिणाम; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    प्रत्यागम परिणाम;
}

// ggml_cont

स्थिर संरचना ggml_tensor * ggml_cont_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (cont)", a->name);

    परिणाम->op     = GGML_OP_CONT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(परिणाम, "%s (cont)", a->name);

    परिणाम->op     = GGML_OP_CONT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_reshape

संरचना ggml_tensor * ggml_reshape(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor * a,
        संरचना ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_reshape_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    ध्रुव int64_t ne[1] = { ne0 };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_reshape_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    ध्रुव int64_t ne[2] = { ne0, ne1 };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

स्थिर संरचना ggml_tensor * ggml_view_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_dims,
        ध्रुव int64_t       * ne,
        size_t                विस्थापनम्) {
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, विस्थापनम्);
    ggml_format_name(परिणाम, "%s (view)", a->name);

    ggml_समुच्चयः_op_params(परिणाम, &विस्थापनम्, परिमाण(विस्थापनम्));

    परिणाम->op     = GGML_OP_VIEW;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_view_1d

संरचना ggml_tensor * ggml_view_1d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        size_t                विस्थापनम्) {
    संरचना ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 1, &ne0, विस्थापनम्);

    प्रत्यागम परिणाम;
}

// ggml_view_2d

संरचना ggml_tensor * ggml_view_2d(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                विस्थापनम्) {
    ध्रुव int64_t ne[2] = { ne0, ne1 };

    संरचना ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 2, ne, विस्थापनम्);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = परिणाम->nb[1]*ne1;
    परिणाम->nb[3] = परिणाम->nb[2];

    प्रत्यागम परिणाम;
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
        size_t                विस्थापनम्) {
    ध्रुव int64_t ne[3] = { ne0, ne1, ne2 };

    संरचना ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 3, ne, विस्थापनम्);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = nb2;
    परिणाम->nb[3] = परिणाम->nb[2]*ne2;

    प्रत्यागम परिणाम;
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
        size_t                विस्थापनम्) {
    ध्रुव int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    संरचना ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 4, ne, विस्थापनम्);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = nb2;
    परिणाम->nb[3] = nb3;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (permuted)", a->name);

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

    परिणाम->ne[0] = ne[0];
    परिणाम->ne[1] = ne[1];
    परिणाम->ne[2] = ne[2];
    परिणाम->ne[3] = ne[3];

    परिणाम->nb[0] = nb[0];
    परिणाम->nb[1] = nb[1];
    परिणाम->nb[2] = nb[2];
    परिणाम->nb[3] = nb[3];

    परिणाम->op     = GGML_OP_PERMUTE;
    परिणाम->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    प्रत्यागम परिणाम;
}

// ggml_transpose

संरचना ggml_tensor * ggml_transpose(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (transposed)", a->name);

    परिणाम->ne[0] = a->ne[1];
    परिणाम->ne[1] = a->ne[0];

    परिणाम->nb[0] = a->nb[1];
    परिणाम->nb[1] = a->nb[0];

    परिणाम->op     = GGML_OP_TRANSPOSE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_get_rows

संरचना ggml_tensor * ggml_get_पङ्क्तिकाः(
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    परिणाम->op     = GGML_OP_GET_पङ्क्तिकाः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_get_rows_back

संरचना ggml_tensor * ggml_get_पङ्क्तिकाः_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_सदिशम्(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    परिणाम->op     = GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_set_rows

संरचना ggml_tensor * ggml_समुच्चयः_पङ्क्तिकाः(
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

    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(a));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(b));

    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_समुच्चयः_पङ्क्तिकाः;
    परिणाम->src[0] = b;
    परिणाम->src[1] = c;
    परिणाम->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    प्रत्यागम परिणाम;
}

// ggml_diag

संरचना ggml_tensor * ggml_diag(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    ध्रुव int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, 4, ne);

    परिणाम->op     = GGML_OP_DIAG;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_diag_mask_inf

स्थिर संरचना ggml_tensor * ggml_diag_आवरणम्_inf_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_DIAG_आवरणम्_INF;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_diag_आवरणम्_inf(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_आवरणम्_inf_impl(ctx, a, n_past, false);
}

संरचना ggml_tensor * ggml_diag_आवरणम्_inf_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_आवरणम्_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

स्थिर संरचना ggml_tensor * ggml_diag_आवरणम्_zero_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_DIAG_आवरणम्_ZERO;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_diag_आवरणम्_zero(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_आवरणम्_zero_impl(ctx, a, n_past, false);
}

संरचना ggml_tensor * ggml_diag_आवरणम्_zero_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_past) {
    प्रत्यागम ggml_diag_आवरणम्_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

स्थिर संरचना ggml_tensor * ggml_clamp_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        भिन्न                 min,
        भिन्न                 max,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[] = { min, max };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_CLAMP;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
        संरचना ggml_tensor  * आवरणम्,
        भिन्न                 scale,
        भिन्न                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    यदि (आवरणम्) {
        GGML_ASSERT(आवरणम्->type == GGML_TYPE_F16 || आवरणम्->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(आवरणम्));
        GGML_ASSERT(आवरणम्->ne[0] == a->ne[0]);
        GGML_ASSERT(आवरणम्->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%आवरणम्->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%आवरणम्->ne[3] == 0);
    }

    यदि (max_bias > 0.0f) {
        GGML_ASSERT(आवरणम्);
    }

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भिन्न params[] = { scale, max_bias };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_SOFT_MAX;
    परिणाम->src[0] = a;
    परिणाम->src[1] = आवरणम्;

    प्रत्यागम परिणाम;
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
        संरचना ggml_tensor  * आवरणम्,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, आवरणम्, scale, max_bias, false);
}

संरचना ggml_tensor * ggml_soft_max_ext_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * आवरणम्,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_impl(ctx, a, आवरणम्, scale, max_bias, true);
}

शून्य ggml_soft_max_सङ्कलय_sinks(
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

स्थिर संरचना ggml_tensor * ggml_soft_max_ext_पृष्ठभागः_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias,
        bool                  inplace) {
    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SOFT_MAX_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    स्मृतिं_प्रतिलिख((भिन्न *) परिणाम->op_params + 0, &scale,    परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख((भिन्न *) परिणाम->op_params + 1, &max_bias, परिमाण(भिन्न));

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_soft_max_ext_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_ext_पृष्ठभागः_impl(ctx, a, b, scale, max_bias, false);
}

संरचना ggml_tensor * ggml_soft_max_ext_पृष्ठभागः_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        भिन्न                 scale,
        भिन्न                 max_bias) {
    प्रत्यागम ggml_soft_max_ext_पृष्ठभागः_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

स्थिर संरचना ggml_tensor * ggml_rope_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((बहुलकम् & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_सदिशम्(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_प्रयुक्तम् = बहुलकम् & GGML_ROPE_TYPE_MROPE;
    यदि (mrope_प्रयुक्तम्) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } अन्यथा {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    यदि (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, बहुलकम्, /*n_ctx*/ 0, n_ctx_orig };
    स्मृतिं_प्रतिलिख(params +  5, &freq_आधारः,    परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(params +  6, &freq_scale,   परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(params +  7, &ext_गुणकः,   परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(params +  8, &attn_गुणकः,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(params +  9, &beta_fast,    परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(params + 10, &beta_slow,    परिमाण(भिन्न));
    यदि (mrope_प्रयुक्तम् && sections) {
        स्मृतिं_प्रतिलिख(params + 11, sections,  परिमाण(int32_t) * GGML_MROPE_SECTIONS);
    } अन्यथा {
        स्मृतिं_पूरय(params + 11, 0,         परिमाण(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_ROPE;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_rope(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, बहुलकम्, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

संरचना ggml_tensor * ggml_rope_multi(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_multi_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[GGML_MROPE_SECTIONS],
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, true
    );
}

संरचना ggml_tensor * ggml_rope_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, बहुलकम्, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

संरचना ggml_tensor * ggml_rope_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, रिक्त, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_ext_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, c, n_dims, रिक्त, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, true
    );
}

संरचना ggml_tensor * ggml_rope_custom(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, false
    );
}

संरचना ggml_tensor * ggml_rope_custom_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    प्रत्यागम ggml_rope_impl(
        ctx, a, b, रिक्त, n_dims, रिक्त, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale,
        ext_गुणकः, attn_गुणकः, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
स्थिर भिन्न ggml_rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न n_rot, भिन्न आधारः) {
    प्रत्यागम n_dims * logf(n_ctx_orig / (n_rot * 2 * (भिन्न)M_PI)) / (2 * logf(आधारः));
}

शून्य ggml_rope_yarn_corr_dims(
    पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न freq_आधारः, भिन्न beta_fast, भिन्न beta_slow, भिन्न dims[2]
) {
    // start and end correction dims
    भिन्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_आधारः));
    भिन्न अन्तः_2   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_आधारः));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, अन्तः_2);
}

// ggml_rope_back

संरचना ggml_tensor * ggml_rope_ext_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    संरचना ggml_tensor * परिणाम = ggml_rope_ext(
        ctx, a, b, c, n_dims, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow);
    परिणाम->op = GGML_OP_ROPE_पृष्ठभागः;
    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_rope_multi_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c,
        पूर्णाङ्क                   n_dims,
        पूर्णाङ्क                   sections[4],
        पूर्णाङ्क                   बहुलकम्,
        पूर्णाङ्क                   n_ctx_orig,
        भिन्न                 freq_आधारः,
        भिन्न                 freq_scale,
        भिन्न                 ext_गुणकः,
        भिन्न                 attn_गुणकः,
        भिन्न                 beta_fast,
        भिन्न                 beta_slow) {
    संरचना ggml_tensor * परिणाम = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow);
    परिणाम->op = GGML_OP_ROPE_पृष्ठभागः;
    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_rope_समुच्चयः_विस्थापनम्(
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_पृष्ठभागः);
    GGML_ASSERT(n_offs >= 0);

    ध्रुव int32_t बहुलकम् = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(बहुलकम् != GGML_ROPE_TYPE_VISION);

    ggml_समुच्चयः_op_params_i32(a, 15, n_offs);
    प्रत्यागम a;
}

स्थिर int64_t ggml_calc_conv_output_आकार(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p, पूर्णाङ्क d) {
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

    ध्रुव int64_t OH = is_2D ? ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    ध्रुव int64_t OW =         ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    ध्रुव int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_IM2COL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_im2col_पृष्ठभागः(
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_IM2_COL_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    परिणाम = ggml_reshape_3d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 2, 1, 3));                // [N, OC, OL]

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_mul_mat(ctx, im2col, a);

    परिणाम = ggml_reshape_3d(ctx, परिणाम, परिणाम->ne[0], परिणाम->ne[2], 1);

    प्रत्यागम परिणाम;
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
    ध्रुव int64_t T_अन्तर् = a->ne[1];
    ध्रुव int64_t K = K_OC / oc;
    ध्रुव int64_t T_बहिर्गमनम् = (T_अन्तर् - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_बहिर्गमनम् > 0);

    ध्रुव int64_t ne[4] = { T_बहिर्गमनम्, oc, 1, 1 };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_COL2IM_1D;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_conv_transpose_1d

स्थिर int64_t ggml_calc_conv_transpose_1d_output_आकार(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p, पूर्णाङ्क d) {
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
        ggml_calc_conv_transpose_1d_output_आकार(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_CONV_TRANSPOSE_1D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 1, 3, 2)); // [N, OC, OH, OW]


    प्रत्यागम परिणाम;
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
    अप्रयुक्तम्(OC);
    ध्रुव int64_t KD = a->ne[2];
    ध्रुव int64_t KH = a->ne[1];
    ध्रुव int64_t KW = a->ne[0];
    ध्रुव int64_t OD = ggml_calc_conv_output_आकार(ID, KD, s2, p2, d2);
    ध्रुव int64_t OH = ggml_calc_conv_output_आकार(IH, KH, s1, p1, d1);
    ध्रुव int64_t OW = ggml_calc_conv_output_आकार(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    ध्रुव int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_IM2COL_3D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_mul_mat(ctx, new_a, new_b);
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    प्रत्यागम परिणाम;
}

// ggml_conv_2d_dw_direct

संरचना ggml_tensor * ggml_conv_2d_dw_प्रत्यक्षम्(
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
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, b->type, 4, ne);

    यदि (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        ध्रुव int64_t type_आकार = ggml_type_आकार(परिणाम->type);
        GGML_ASSERT(ggml_blck_आकार(परिणाम->type) == 1);
        परिणाम->nb[0] = परिणाम->ne[2] * type_आकार;
        परिणाम->nb[1] = परिणाम->ne[0] * परिणाम->nb[0];
        परिणाम->nb[2] = type_आकार;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_CONV_2D_DW;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    प्रत्यागम परिणाम;
}

// ggml_conv_2d_direct

संरचना ggml_tensor * ggml_conv_2d_प्रत्यक्षम्(
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
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, s0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, s1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 2, p0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 3, p1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 4, d0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 5, d1);

    परिणाम->op = GGML_OP_CONV_2D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_conv_3d_direct

संरचना ggml_tensor * ggml_conv_3d_प्रत्यक्षम्(
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
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_आकार(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0,  s0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1,  s1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 2,  s2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 3,  p0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 4,  p1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 5,  p2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 6,  d0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 7,  d1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 8,  d2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 9,  c);
    ggml_समुच्चयः_op_params_i32(परिणाम, 10, n);
    ggml_समुच्चयः_op_params_i32(परिणाम, 11, oc);

    परिणाम->op = GGML_OP_CONV_3D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_conv_transpose_2d_p0

स्थिर int64_t ggml_calc_conv_transpose_output_आकार(int64_t ins, int64_t ks, पूर्णाङ्क s, पूर्णाङ्क p) {
    प्रत्यागम (ins - 1) * s - 2 * p + ks;
}

संरचना ggml_tensor * ggml_conv_transpose_2d_p0(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        पूर्णाङ्क                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    ध्रुव int64_t ne[4] = {
        ggml_calc_conv_transpose_output_आकार(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_आकार(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    संरचना ggml_tensor* परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, stride);

    परिणाम->op     = GGML_OP_CONV_TRANSPOSE_2D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_pool_*

स्थिर int64_t ggml_calc_pool_output_आकार(int64_t ins, पूर्णाङ्क ks, पूर्णाङ्क s, भिन्न p) {
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
        ggml_calc_pool_output_आकार(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_POOL_1D;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम;
    ध्रुव int64_t ne[4] = {
        ggml_calc_pool_output_आकार(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_आकार(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_POOL_2D;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_pool_2d_पृष्ठभागः(
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
    संरचना ggml_tensor * परिणाम;
    परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_POOL_2_D_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = af;

    प्रत्यागम परिणाम;
}

// ggml_upscale / ggml_interpolate

स्थिर संरचना ggml_tensor * ggml_interpolate_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              बहुलकम्) {
    GGML_ASSERT((बहुलकम् & 0xFF) < GGML_SCALE_बहुलकम्_गणनम्);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(बहुलकम् & GGML_SCALE_FLAG_ANTIALIAS) || (बहुलकम् & 0xFF) == GGML_SCALE_बहुलकम्_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t)बहुलकम्);

    परिणाम->op     = GGML_OP_UPSCALE;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_upscale(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   scale_गुणकः,
        गणना ggml_scale_बहुलकम्  बहुलकम्) {
    GGML_ASSERT(scale_गुणकः > 1);
    प्रत्यागम ggml_interpolate_impl(ctx, a, a->ne[0] * scale_गुणकः, a->ne[1] * scale_गुणकः, a->ne[2], a->ne[3], बहुलकम्);
}

संरचना ggml_tensor * ggml_upscale_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   ne0,
        पूर्णाङ्क                   ne1,
        पूर्णाङ्क                   ne2,
        पूर्णाङ्क                   ne3,
        गणना ggml_scale_बहुलकम्  बहुलकम्) {
    प्रत्यागम ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, बहुलकम्);
}

संरचना ggml_tensor * ggml_interpolate(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              बहुलकम्) {
    प्रत्यागम ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, बहुलकम्);
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, lp0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, rp0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 2, lp1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 3, rp1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 4, lp2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 5, rp2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 6, lp3);
    ggml_समुच्चयः_op_params_i32(परिणाम, 7, rp3);
    ggml_समुच्चयः_op_params_i32(परिणाम, 8, 0); // not circular by default


    परिणाम->op     = GGML_OP_PAD;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_समुच्चयः_op_params_i32(परिणाम, 8, 1); // circular
    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_PAD_REFLECT_1D;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_roll

संरचना ggml_tensor * ggml_roll(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   shift0,
        पूर्णाङ्क                   shift1,
        पूर्णाङ्क                   shift2,
        पूर्णाङ्क                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_आकार(a->type));
    GGML_ASSERT(निरपेक्षम्(shift0) < a->ne[0]);
    GGML_ASSERT(निरपेक्षम्(shift1) < a->ne[1]);
    GGML_ASSERT(निरपेक्षम्(shift2) < a->ne[2]);
    GGML_ASSERT(निरपेक्षम्(shift3) < a->ne[3]);

    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, shift0);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, shift1);
    ggml_समुच्चयः_op_params_i32(परिणाम, 2, shift2);
    ggml_समुच्चयः_op_params_i32(परिणाम, 3, shift3);

    परिणाम->op     = GGML_OP_ROLL;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_timestep_embedding

संरचना ggml_tensor * ggml_timestep_embedding(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * timesteps,
        पूर्णाङ्क                   dim,
        पूर्णाङ्क                   max_period) {

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, dim);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, max_period);

    परिणाम->op     = GGML_OP_TIMESTEP_EMBEDDING;
    परिणाम->src[0] = timesteps;

    प्रत्यागम परिणाम;
}

// ggml_tri

संरचना ggml_tensor * ggml_tri(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    गणना ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, type);

    परिणाम->op = GGML_OP_TRI;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_fill

स्थिर संरचना ggml_tensor * ggml_fill_impl(
    संरचना ggml_context * ctx,
    संरचना ggml_tensor  * a,
    भिन्न                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, c);

    परिणाम->op = GGML_OP_FILL;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
        गणना ggml_क्रमय_क्रमः   क्रमः) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) क्रमः);

    परिणाम->op     = GGML_OP_ARGSORT;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_argsort_top_k

संरचना ggml_tensor * ggml_argsort_शिखरम्_k(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    संरचना ggml_tensor * परिणाम = ggml_argsort(ctx, a, GGML_क्रमय_क्रमः_DESC);

    परिणाम = ggml_view_4d(ctx, परिणाम,
                k, परिणाम->ne[1], परिणाम->ne[2], परिणाम->ne[3],
                   परिणाम->nb[1], परिणाम->nb[2], परिणाम->nb[3],
                0);

    प्रत्यागम परिणाम;
}

// ggml_top_k

संरचना ggml_tensor * ggml_शिखरम्_k(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        पूर्णाङ्क                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    परिणाम->op     = GGML_OP_शिखरम्_K;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_arange

संरचना ggml_tensor * ggml_arange(
        संरचना ggml_context * ctx,
        भिन्न                 start,
        भिन्न                 stop,
        भिन्न                 चरणः) {
    GGML_ASSERT(stop > start);

    ध्रुव int64_t चरणाः = (int64_t) ceilf((stop - start) / चरणः);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, चरणाः);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, start);
    ggml_समुच्चयः_op_params_f32(परिणाम, 1, stop);
    ggml_समुच्चयः_op_params_f32(परिणाम, 2, चरणः);

    परिणाम->op = GGML_OP_ARANGE;

    प्रत्यागम परिणाम;
}

// ggml_flash_attn_ext

संरचना ggml_tensor * ggml_flash_attn_ext(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * v,
        संरचना ggml_tensor  * आवरणम्,
        भिन्न                 scale,
        भिन्न                 max_bias,
        भिन्न                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    यदि (आवरणम्) {
        GGML_ASSERT(आवरणम्->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(आवरणम्));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % आवरणम्->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % आवरणम्->ne[3] == 0);
    }

    यदि (max_bias > 0.0f) {
        GGML_ASSERT(आवरणम्);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    भिन्न params[] = { scale, max_bias, logit_softcap };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_FLASH_ATTN_EXT;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = आवरणम्;

    प्रत्यागम परिणाम;
}


शून्य ggml_flash_attn_ext_समुच्चयः_prec(
        संरचना ggml_tensor * a,
        गणना ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ध्रुव int32_t prec_i32 = (int32_t) prec;

    ggml_समुच्चयः_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

गणना ggml_prec ggml_flash_attn_ext_get_prec(
        ध्रुव संरचना ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ध्रुव int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    प्रत्यागम (गणना ggml_prec) prec_i32;
}

शून्य ggml_flash_attn_ext_समुच्चयः_n_kv_max(
        संरचना ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_समुच्चयः_op_params_i32(a, 4, n_kv_max);
}

शून्य ggml_flash_attn_ext_सङ्कलय_sinks(
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

संरचना ggml_tensor * ggml_flash_attn_पृष्ठभागः(
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

    गणना ggml_type परिणाम_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_आकार(परिणाम_type) == 1);
    ध्रुव size_t tsize = ggml_type_आकार(परिणाम_type);

    ध्रुव size_t offs_q = 0;
    ध्रुव size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t अन्तः_2    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    ध्रुव size_t nelements = (अन्तः_2 + tsize - 1)/tsize;

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_समुच्चयः_op_params(परिणाम, &masked_i, परिमाण(masked_i));

    परिणाम->op     = GGML_OP_FLASH_ATTN_पृष्ठभागः;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = d;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    परिणाम->op     = GGML_OP_SSM_CONV;
    परिणाम->src[0] = sx;
    परिणाम->src[1] = c;

    प्रत्यागम परिणाम;
}

// ggml_ssm_scan

संरचना ggml_tensor * ggml_ssm_अवलोकनम्(
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
    GGML_ASSERT(x->nb[0] == ggml_type_आकार(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_आकार(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_आकार(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        ध्रुव int64_t d_state      = s->ne[0];
        ध्रुव int64_t शीर्ष_dim     = x->ne[0];
        ध्रुव int64_t n_शीर्ष       = x->ne[1];
        ध्रुव int64_t n_seq_tokens = x->ne[2];
        ध्रुव int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_शीर्ष);
        GGML_ASSERT(dt->ne[1] == n_seq_tokens);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == शीर्ष_dim);
        GGML_ASSERT(s->ne[2] == n_शीर्ष);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_tokens);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_सदिशम्(ids));
        GGML_ASSERT(A->ne[1] == n_शीर्ष);
        GGML_ASSERT(ggml_is_matrix(A));

        यदि (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    परिणाम->op   = GGML_OP_SSM_अवलोकनम्;
    परिणाम->src[0] = s;
    परिणाम->src[1] = x;
    परिणाम->src[2] = dt;
    परिणाम->src[3] = A;
    परिणाम->src[4] = B;
    परिणाम->src[5] = C;
    परिणाम->src[6] = ids;

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) K);

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_WIN_PART;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_समुच्चयः_op_params(परिणाम, params, परिमाण(params));

    परिणाम->op     = GGML_OP_WIN_UNPART;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    परिणाम->op     = GGML_OP_GET_REL_POS;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
}

// ggml_add_rel_pos

स्थिर संरचना ggml_tensor * ggml_सङ्कलय_rel_pos_impl(
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_समुच्चयः_op_params_i32(परिणाम, 0, inplace ? 1 : 0);

    परिणाम->op     = GGML_OP_सङ्कलय_REL_POS;
    परिणाम->src[0] = a;
    परिणाम->src[1] = pw;
    परिणाम->src[2] = ph;

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_सङ्कलय_rel_pos(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * pw,
        संरचना ggml_tensor  * ph) {
    प्रत्यागम ggml_सङ्कलय_rel_pos_impl(ctx, a, pw, ph, false);
}

संरचना ggml_tensor * ggml_सङ्कलय_rel_pos_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * pw,
        संरचना ggml_tensor  * ph) {
    प्रत्यागम ggml_सङ्कलय_rel_pos_impl(ctx, a, pw, ph, true);
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_RWKV_WKV6;
    परिणाम->src[0] = k;
    परिणाम->src[1] = v;
    परिणाम->src[2] = r;
    परिणाम->src[3] = tf;
    परिणाम->src[4] = td;
    परिणाम->src[5] = state;

    प्रत्यागम परिणाम;
}

// ggml_gated_linear_attn

संरचना ggml_tensor * ggml_gated_रैखिकम्_attn(
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, scale);

    परिणाम->op     = GGML_OP_GATED_रैखिकम्_ATTN;
    परिणाम->src[0] = k;
    परिणाम->src[1] = v;
    परिणाम->src[2] = q;
    परिणाम->src[3] = g;
    परिणाम->src[4] = state;

    प्रत्यागम परिणाम;
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
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_RWKV_WKV7;
    परिणाम->src[0] = r;
    परिणाम->src[1] = w;
    परिणाम->src[2] = k;
    परिणाम->src[3] = v;
    परिणाम->src[4] = a;
    परिणाम->src[5] = b;
    परिणाम->src[6] = state;

    प्रत्यागम परिणाम;
}

// ggml_unary

स्थिर संरचना ggml_tensor * ggml_unary_impl(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        गणना ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(a));

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) op);

    परिणाम->op     = GGML_OP_UNARY;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM1;
    परिणाम->src[0] = a;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM2;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    संरचना ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM3;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    प्रत्यागम परिणाम;
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
        संरचना ggml_tensor ** तर्काः_2,
        पूर्णाङ्क                   n_तर्काः,
        ggml_custom_op_t      fun,
        पूर्णाङ्क                   n_tasks,
        शून्य                * userdata) {

    GGML_ASSERT(n_तर्काः < GGML_MAX_SRC);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    संरचना ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op = GGML_OP_CUSTOM;
    प्रति (पूर्णाङ्क i = 0; i < n_तर्काः; i++) {
        परिणाम->src[i] = तर्काः_2[i];
    }

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_custom_inplace(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor ** तर्काः_2,
        पूर्णाङ्क                   n_तर्काः,
        ggml_custom_op_t      fun,
        पूर्णाङ्क                   n_tasks,
        शून्य                * userdata) {

    GGML_ASSERT(n_तर्काः < GGML_MAX_SRC - 1);

    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    संरचना ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_समुच्चयः_op_params(परिणाम, &params, परिमाण(params));

    परिणाम->op = GGML_OP_CUSTOM;
    परिणाम->src[0] = a;
    प्रति (पूर्णाङ्क i = 0; i < n_तर्काः; i++) {
        परिणाम->src[i + 1] = तर्काः_2[i];
    }

    प्रत्यागम परिणाम;
}
// ggml_cross_entropy_loss

संरचना ggml_tensor * ggml_cross_entropy_loss(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, a->type, 1);

    परिणाम->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
}

// ggml_cross_entropy_loss_back

संरचना ggml_tensor * ggml_cross_entropy_loss_पृष्ठभागः(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        संरचना ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    संरचना ggml_tensor * परिणाम = ggml_dup_tensor(ctx, b);

    परिणाम->op     = GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    प्रत्यागम परिणाम;
}

// opt_step_adamw

संरचना ggml_tensor * ggml_opt_चरणः_adamw(
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

    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_OPT_चरणः_ADAMW;
    परिणाम->src[0] = a;
    परिणाम->src[1] = grad;
    परिणाम->src[2] = m;
    परिणाम->src[3] = v;
    परिणाम->src[4] = adamw_params;

    प्रत्यागम परिणाम;
}

// opt_step_sgd

संरचना ggml_tensor * ggml_opt_चरणः_sgd(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * grad,
        संरचना ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    संरचना ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_OPT_चरणः_SGD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = grad;
    परिणाम->src[2] = params;

    प्रत्यागम परिणाम;
}

// solve_tri

संरचना ggml_tensor * ggml_solve_tri(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * a,
        संरचना ggml_tensor  * b,
        bool                  वामः,
        bool                  अधस्तनम्,
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

    GGML_ASSERT(अधस्तनम् && वामः && !uni); // TODO: support other variants

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    परिणाम->op     = GGML_OP_SOLVE_TRI;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    प्रत्यागम परिणाम;
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
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(q));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(k));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(v));
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
    ध्रुव int64_t state_पङ्क्तिकाः = K * S_v * n_seqs;
    ध्रुव int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_पङ्क्तिकाः, 1, 1 };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_समुच्चयः_op_params_i32(परिणाम, 0, (int32_t) K);

    परिणाम->op     = GGML_OP_GATED_DELTA_NET;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = g;
    परिणाम->src[4] = beta;
    परिणाम->src[5] = state;

    प्रत्यागम परिणाम;
}

// ggml_lightning_indexer

संरचना ggml_tensor * ggml_lightning_indexer(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * q,
        संरचना ggml_tensor  * k,
        संरचना ggml_tensor  * weights,
        संरचना ggml_tensor  * आवरणम्) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    आवरणम्->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   आवरणम्->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   आवरणम्->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   आवरणम्->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % आवरणम्->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    संरचना ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op   = GGML_OP_LIGHTNING_INDEXER;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = weights;
    परिणाम->src[3] = आवरणम्;

    प्रत्यागम परिणाम;
}

// ggml_dsv4_hc_comb

संरचना ggml_tensor * ggml_dsv4_hc_comb(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * mixes,
        संरचना ggml_tensor  * scale,
        संरचना ggml_tensor  * आधारः,
        भिन्न                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(आधारः->type == GGML_TYPE_F32);
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
    GGML_ASSERT(आधारः->ne[0] == hc_mix_dim);
    GGML_ASSERT(आधारः->ne[1] == 1);
    GGML_ASSERT(आधारः->ne[2] == 1);
    GGML_ASSERT(आधारः->ne[3] == 1);

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, eps);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, n_iter);

    परिणाम->op     = GGML_OP_DSV4_HC_COMB;
    परिणाम->src[0] = mixes;
    परिणाम->src[1] = scale;
    परिणाम->src[2] = आधारः;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_समुच्चयः_op_params_f32(परिणाम, 0, scale);
    ggml_समुच्चयः_op_params_i32(परिणाम, 1, gated ? 1 : 0);

    परिणाम->op     = GGML_OP_DSV4_HC_PRE;
    परिणाम->src[0] = x;
    परिणाम->src[1] = weights;

    प्रत्यागम परिणाम;
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

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    परिणाम->op     = GGML_OP_DSV4_HC_POST;
    परिणाम->src[0] = x;
    परिणाम->src[1] = residual;
    परिणाम->src[2] = post;
    परिणाम->src[3] = comb;

    प्रत्यागम परिणाम;
}

// ggml_top_k_mask

संरचना ggml_tensor * ggml_शिखरम्_k_आवरणम्(
        संरचना ggml_context * ctx,
        संरचना ggml_tensor  * आवरणम्,
        संरचना ggml_tensor  * शिखरम्_k,
        संरचना ggml_tensor  * score,
        संरचना ggml_tensor  * खण्डानि,
        संरचना ggml_tensor  * extra) {
    GGML_ASSERT(आवरणम्->type == GGML_TYPE_F16 || आवरणम्->type == GGML_TYPE_F32);
    GGML_ASSERT(आवरणम्->nb[0] == ggml_type_आकार(आवरणम्->type));
    GGML_ASSERT(शिखरम्_k->type == GGML_TYPE_I32);
    GGML_ASSERT(शिखरम्_k->ne[1] % आवरणम्->ne[1] == 0 && शिखरम्_k->ne[2] % आवरणम्->ne[2] == 0 && शिखरम्_k->ne[3] % आवरणम्->ne[3] == 0);

    यदि (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == शिखरम्_k->ne[1] && score->ne[2] == शिखरम्_k->ne[2] && score->ne[3] == शिखरम्_k->ne[3]);
    }

    यदि (खण्डानि) {
        GGML_ASSERT(खण्डानि->type == GGML_TYPE_I32);
        GGML_ASSERT(खण्डानि->ne[2] == 1 && खण्डानि->ne[3] == 1);
        GGML_ASSERT(!score || खण्डानि->ne[1] == score->ne[0]);
    }

    यदि (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == शिखरम्_k->ne[1] && extra->ne[2] == शिखरम्_k->ne[2] && extra->ne[3] == शिखरम्_k->ne[3]);
    }

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, आवरणम्->type, आवरणम्->ne[0], शिखरम्_k->ne[1], शिखरम्_k->ne[2], शिखरम्_k->ne[3]);

    परिणाम->op     = GGML_OP_शिखरम्_K_आवरणम्;
    परिणाम->src[0] = आवरणम्;
    परिणाम->src[1] = शिखरम्_k;
    परिणाम->src[2] = score;
    परिणाम->src[3] = खण्डानि;
    परिणाम->src[4] = extra;

    प्रत्यागम परिणाम;
}

////////////////////////////////////////////////////////////////////////////////

संरचना ggml_hash_समुच्चयः ggml_hash_समुच्चयः_new(size_t आकार) {
    आकार = ggml_hash_आकार(आकार);
    संरचना ggml_hash_समुच्चयः परिणाम;
    परिणाम.आकार = आकार;
    परिणाम.कुञ्चिकाः = GGML_MALLOC(परिमाण(संरचना ggml_tensor *) * आकार);
    परिणाम.प्रयुक्तम् = GGML_CALLOC(ggml_bitset_आकार(आकार), परिमाण(ggml_bitset_t));
    प्रत्यागम परिणाम;
}

शून्य ggml_hash_समुच्चयः_reset(संरचना ggml_hash_समुच्चयः * hash_समुच्चयः) {
    स्मृतिं_पूरय(hash_समुच्चयः->प्रयुक्तम्, 0, परिमाण(ggml_bitset_t) * ggml_bitset_आकार(hash_समुच्चयः->आकार));
}

शून्य ggml_hash_समुच्चयः_free(संरचना ggml_hash_समुच्चयः * hash_समुच्चयः) {
    GGML_FREE(hash_समुच्चयः->प्रयुक्तम्);
    GGML_FREE(hash_समुच्चयः->कुञ्चिकाः);
}

size_t ggml_hash_आकार(size_t min_sz) {
    // next primes after powers of two
    स्थिर ध्रुव size_t अभाज्याः[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    स्थिर ध्रुव size_t n_अभाज्याः = परिमाण(अभाज्याः)/परिमाण(अभाज्याः[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_अभाज्याः;
    यावत् (l < r) {
        size_t m = (l + r)/2;
        यदि (अभाज्याः[m] < min_sz) {
            l = m + 1;
        } अन्यथा {
            r = m;
        }
    }
    size_t sz = l < n_अभाज्याः ? अभाज्याः[l] : min_sz | 1;
    प्रत्यागम sz;
}

संरचना hash_map {
    संरचना ggml_hash_समुच्चयः समुच्चयः;
    संरचना ggml_tensor ** vals;
};

स्थिर संरचना hash_map * ggml_new_hash_map(size_t आकार) {
    संरचना hash_map * परिणाम = GGML_MALLOC(परिमाण(संरचना hash_map));
    परिणाम->समुच्चयः = ggml_hash_समुच्चयः_new(आकार);
    परिणाम->vals = GGML_CALLOC(परिणाम->समुच्चयः.आकार, परिमाण(संरचना ggml_tensor *));
    प्रत्यागम परिणाम;
}

स्थिर शून्य ggml_hash_map_free(संरचना hash_map * map) {
    ggml_hash_समुच्चयः_free(&map->समुच्चयः);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

स्थिर शून्य ggml_सङ्कलय_or_समुच्चयः(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_सङ्कलय_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_acc_or_समुच्चयः(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor,
        ध्रुव  size_t         nb1,
        ध्रुव  size_t         nb2,
        ध्रुव  size_t         nb3,
        ध्रुव  size_t         विस्थापनम्) {
    संरचना ggml_tensor * src = cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, विस्थापनम्, cgraph->grad_accs[isrc]);
    } अन्यथा {
        संरचना ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, विस्थापनम्, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_add1_or_समुच्चयः(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[isrc];
    GGML_ASSERT(src);
    यदि (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

स्थिर शून्य ggml_sub_or_समुच्चयः(
        संरचना ggml_context * ctx,
        संरचना ggml_cgraph  * cgraph,
        size_t                isrc,
        संरचना ggml_tensor  * tensor) {
    संरचना ggml_tensor * src = cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[isrc];
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
    संरचना ggml_hash_समुच्चयः * hash_समुच्चयः = &cgraph->visited_hash_समुच्चयः;
    ध्रुव size_t isrc0 = src0 ? ggml_hash_find(hash_समुच्चयः, src0) : (size_t) -1;
    ध्रुव size_t isrc1 = src1 ? ggml_hash_find(hash_समुच्चयः, src1) : (size_t) -1;
    ध्रुव size_t isrc2 = src2 ? ggml_hash_find(hash_समुच्चयः, src2) : (size_t) -1;
    ध्रुव bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_पूर्णम् && ggml_bitset_get(hash_समुच्चयः->प्रयुक्तम्, isrc0) && grads_needed[isrc0];
    ध्रुव bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_पूर्णम् && ggml_bitset_get(hash_समुच्चयः->प्रयुक्तम्, isrc1) && grads_needed[isrc1];
    ध्रुव bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_पूर्णम् && ggml_bitset_get(hash_समुच्चयः->प्रयुक्तम्, isrc2) && grads_needed[isrc2];

    विकल्प (tensor->op) {
        पक्ष GGML_OP_DUP: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_सङ्कलय: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                संरचना ggml_tensor * tmp = grad;
                यदि (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_पृष्ठभागः(ctx, tmp, src1);
                }
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, tmp);
            }
        } भङ्ग;
        पक्ष GGML_OP_ADD1: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, ggml_माध्यम्(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } भङ्ग;
        पक्ष GGML_OP_ACC: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ध्रुव size_t nb1    = ((int32_t *) tensor->op_params)[0];
                ध्रुव size_t nb2    = ((int32_t *) tensor->op_params)[1];
                ध्रुव size_t nb3    = ((int32_t *) tensor->op_params)[2];
                ध्रुव size_t विस्थापनम् = ((int32_t *) tensor->op_params)[3];

                संरचना ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, विस्थापनम्);

                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } भङ्ग;
        पक्ष GGML_OP_SUB: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
            यदि (src1_needs_grads) {
                ggml_sub_or_समुच्चयः(ctx, cgraph, isrc1, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_MUL: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            यदि (src1_needs_grads) {
                संरचना ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                यदि (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_पृष्ठभागः(ctx, tmp, src1);
                }
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, tmp);
            }
        } भङ्ग;
        पक्ष GGML_OP_DIV: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            यदि (src1_needs_grads) {
                ggml_sub_or_समुच्चयः(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } भङ्ग;
        पक्ष GGML_OP_SQR: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } भङ्ग;
        पक्ष GGML_OP_SQRT: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } भङ्ग;
        पक्ष GGML_OP_LOG: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_SIN: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } भङ्ग;
        पक्ष GGML_OP_COS: {
            यदि (src0_needs_grads) {
                ggml_sub_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } भङ्ग;
        पक्ष GGML_OP_योग: {
            यदि (src0_needs_grads) {
                ggml_add1_or_समुच्चयः(ctx, cgraph, isrc0, grad);
            }
        } भङ्ग;
        पक्ष GGML_OP_योग_पङ्क्तिकाः: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_माध्यम्: {
            यदि (src0_needs_grads) {
                ggml_add1_or_समुच्चयः(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_REPEAT: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_repeat_पृष्ठभागः(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_REPEAT_पृष्ठभागः: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_RMS_NORM: {
            यदि (src0_needs_grads) {
                भिन्न eps;
                स्मृतिं_प्रतिलिख(&eps, tensor->op_params, परिमाण(भिन्न));
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_rms_norm_पृष्ठभागः(ctx, grad, src0, eps));
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
                    ggml_बहिर्गमनम्_prod(ctx, // [n,m,qq,rr]
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
                    tmp = ggml_repeat_पृष्ठभागः(ctx, tmp, src0);
                }
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, tmp);
            }
            यदि (src1_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_बहिर्गमनम्_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } भङ्ग;
        पक्ष GGML_OP_SCALE: {
            यदि (src0_needs_grads) {
                भिन्न s;
                स्मृतिं_प्रतिलिख(&s, tensor->op_params, परिमाण(भिन्न));
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_समुच्चयः: {
            ध्रुव size_t nb1    = ((ध्रुव int32_t *) tensor->op_params)[0];
            ध्रुव size_t nb2    = ((ध्रुव int32_t *) tensor->op_params)[1];
            ध्रुव size_t nb3    = ((ध्रुव int32_t *) tensor->op_params)[2];
            ध्रुव size_t विस्थापनम् = ((ध्रुव int32_t *) tensor->op_params)[3];

            संरचना ggml_tensor * tensor_grad_view = रिक्त;

            यदि (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, विस्थापनम्);
            }

            यदि (src0_needs_grads) {
                संरचना ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, विस्थापनम्, false));
            }

            यदि (src1_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } भङ्ग;
        पक्ष GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            यदि (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
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
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_RESHAPE: {
            यदि (src0_needs_grads) {
                संरचना ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } भङ्ग;
        पक्ष GGML_OP_VIEW: {
            यदि (src0_needs_grads) {
                size_t विस्थापनम्;

                स्मृतिं_प्रतिलिख(&विस्थापनम्, tensor->op_params, परिमाण(विस्थापनम्));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                यदि (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_आकार(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_आकार(src0);
                    GGML_ASSERT(विस्थापनम् % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    विस्थापनम् = (विस्थापनम् / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_समुच्चयः(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, विस्थापनम्);
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
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } भङ्ग;
        पक्ष GGML_OP_TRANSPOSE: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः: {
            यदि (src0_needs_grads) {
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_get_पङ्क्तिकाः_पृष्ठभागः(ctx, grad, src1, src0));
            }
            यदि (src1_needs_grads) {
                // noop
            }
        } भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_INF: {
            यदि (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                ध्रुव पूर्णाङ्क n_past = ((ध्रुव int32_t *) tensor->op_params)[0];
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_diag_आवरणम्_zero_impl(ctx, grad, n_past, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_ZERO: {
            यदि (src0_needs_grads) {
                ध्रुव पूर्णाङ्क n_past = ((ध्रुव int32_t *) tensor->op_params)[0];
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_diag_आवरणम्_zero_impl(ctx, grad, n_past, false));
            }
        } भङ्ग;
        पक्ष GGML_OP_SOFT_MAX: {
            यदि (src0_needs_grads) {
                भिन्न scale    = 1.0f;
                भिन्न max_bias = 0.0f;

                स्मृतिं_प्रतिलिख(&scale,    (ध्रुव भिन्न *) tensor->op_params + 0, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) tensor->op_params + 1, परिमाण(भिन्न));

                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_soft_max_ext_पृष्ठभागः(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } भङ्ग;
        पक्ष GGML_OP_ROPE: {
            यदि (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                ध्रुव पूर्णाङ्क n_dims     = ((ध्रुव int32_t *) tensor->op_params)[1];
                ध्रुव पूर्णाङ्क बहुलकम्       = ((ध्रुव int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                ध्रुव पूर्णाङ्क n_ctx_orig = ((ध्रुव int32_t *) tensor->op_params)[4];
                भिन्न freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow;
                पूर्णाङ्क sections[4] = {0, 0, 0, 0};

                स्मृतिं_प्रतिलिख(&freq_आधारः,   (ध्रुव भिन्न *) tensor->op_params +  5, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&freq_scale,  (ध्रुव भिन्न *) tensor->op_params +  6, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&ext_गुणकः,  (ध्रुव भिन्न *) tensor->op_params +  7, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&attn_गुणकः, (ध्रुव भिन्न *) tensor->op_params +  8, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&beta_fast,   (ध्रुव भिन्न *) tensor->op_params +  9, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&beta_slow,   (ध्रुव भिन्न *) tensor->op_params + 10, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&sections,                    tensor->op_params + 11, परिमाण(sections));

                संरचना ggml_tensor * rope_पृष्ठभागः = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_पृष्ठभागः(ctx, grad, src1, src2, n_dims,
                        बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow) :
                    ggml_rope_multi_पृष्ठभागः(ctx, grad, src1, src2, n_dims, sections,
                        बहुलकम्, n_ctx_orig, freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow);
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, rope_पृष्ठभागः);
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

                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, ggml_im2col_पृष्ठभागः(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
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

                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_pool_2d_पृष्ठभागः(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } भङ्ग;
        पक्ष GGML_OP_WIN_PART:
        पक्ष GGML_OP_WIN_UNPART:
        पक्ष GGML_OP_UNARY: {
            विकल्प (ggml_get_unary_op(tensor)) {
                पक्ष GGML_UNARY_OP_ABS: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SGN: {
                    // noop
                } भङ्ग;
                पक्ष GGML_UNARY_OP_NEG: {
                    यदि (src0_needs_grads) {
                        ggml_sub_or_समुच्चयः(ctx, cgraph, isrc0, grad);
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_चरणः: {
                    // noop
                } भङ्ग;
                पक्ष GGML_UNARY_OP_RELU: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_चरणः(ctx, src0), grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SILU: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_silu_पृष्ठभागः(ctx, grad, src0));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_EXP: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_EXPM1: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } भङ्ग;
                पक्ष GGML_UNARY_OP_SOFTPLUS: {
                    यदि (src0_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
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
                ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_cross_entropy_loss_पृष्ठभागः(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } भङ्ग;
        पक्ष GGML_OP_GLU: {
            विकल्प (ggml_get_glu_op(tensor)) {
                पक्ष GGML_GLU_OP_SWIGLU: {
                    यदि (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc0, ggml_silu_पृष्ठभागः(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    यदि (src1_needs_grads) {
                        ggml_सङ्कलय_or_समुच्चयः(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
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
        पक्ष GGML_OP_गणनम्:
        स्वभाव: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

स्थिर size_t ggml_दर्शय_parents_graph(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * ग्रन्थिः, bool compute) {
    यदि (ग्रन्थिः->op != GGML_OP_NONE && compute) {
        ग्रन्थिः->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ध्रुव size_t ग्रन्थिः_hash_pos = ggml_hash_find(&cgraph->visited_hash_समुच्चयः, ग्रन्थिः);
    GGML_ASSERT(ग्रन्थिः_hash_pos != GGML_HASHSET_पूर्णम्);

    यदि (ggml_bitset_get(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, ग्रन्थिः_hash_pos)) {
        // already visited

        यदि (compute) {
            // update the compute flag regardless
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; ++i) {
                संरचना ggml_tensor * src = ग्रन्थिः->src[i];
                यदि (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_दर्शय_parents_graph(cgraph, src, true);
                }
            }
        }

        प्रत्यागम ग्रन्थिः_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_समुच्चयः.कुञ्चिकाः[ग्रन्थिः_hash_pos] = ग्रन्थिः;
    ggml_bitset_समुच्चयः(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, ग्रन्थिः_hash_pos);
    cgraph->use_counts[ग्रन्थिः_hash_pos] = 0;

    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; ++i) {
        ध्रुव पूर्णाङ्क k =
            (cgraph->क्रमः == GGML_CGRAPH_EVAL_क्रमः_वामः_गन्तव्यम्_दक्षिणः) ? i :
            (cgraph->क्रमः == GGML_CGRAPH_EVAL_क्रमः_दक्षिणः_गन्तव्यम्_वामः) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        संरचना ggml_tensor * src = ग्रन्थिः->src[k];
        यदि (src) {
            ध्रुव size_t src_hash_pos = ggml_दर्शय_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    यदि (ग्रन्थिः->op == GGML_OP_NONE && !(ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->आकार);

        यदि (वाक्यदैर्घ्यम्(ग्रन्थिः->name) == 0) {
            ggml_format_name(ग्रन्थिः, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = ग्रन्थिः;
        cgraph->n_leafs++;
    } अन्यथा {
        GGML_ASSERT(cgraph->n_nodes < cgraph->आकार);

        यदि (वाक्यदैर्घ्यम्(ग्रन्थिः->name) == 0) {
            ggml_format_name(ग्रन्थिः, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = ग्रन्थिः;
        cgraph->n_nodes++;
    }

    प्रत्यागम ग्रन्थिः_hash_pos;
}

स्थिर शून्य ggml_build_forward_impl(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor, bool expand, bool compute) {
    यदि (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    ध्रुव पूर्णाङ्क n_old = cgraph->n_nodes;

    ggml_दर्शय_parents_graph(cgraph, tensor, compute);

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

शून्य ggml_build_forward_क्रमः(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor) {
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

    स्मृतिं_पूरय(cgraph->grads,     0, cgraph->visited_hash_समुच्चयः.आकार*परिमाण(संरचना ggml_tensor *));
    स्मृतिं_पूरय(cgraph->grad_accs, 0, cgraph->visited_hash_समुच्चयः.आकार*परिमाण(संरचना ggml_tensor *));
    bool * grads_needed = शून्यस्मृतिं_देहि(cgraph->visited_hash_समुच्चयः.आकार, परिमाण(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        प्रति (पूर्णाङ्क i = 0; i < n_nodes_f; ++i) {
            संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
            any_params = any_params || (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    प्रति (पूर्णाङ्क i = 0; i < n_nodes_f; ++i) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        यदि (ग्रन्थिः->type == GGML_TYPE_I32) {
            अनुवर्त;
        }

        bool ग्रन्थिः_needs_grad = (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM) || (ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        विकल्प (ग्रन्थिः->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            पक्ष GGML_OP_IM2COL:      // only used for its shape
            पक्ष GGML_OP_IM2_COL_पृष्ठभागः: // same as IM2COL
                ignore_src[0] = true;
                भङ्ग;
            पक्ष GGML_OP_UNARY: {
                ध्रुव गणना ggml_unary_op uop = ggml_get_unary_op(ग्रन्थिः);
                // SGN and STEP unary ops are piecewise constant
                यदि (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_चरणः) {
                    ignore_src[0] = true;
                }
            } भङ्ग;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            पक्ष GGML_OP_CPY:           // gradients in CPY target are irrelevant
            पक्ष GGML_OP_GET_पङ्क्तिकाः:      // row indices not differentiable
            पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः: // same as for GET_ROWS
            पक्ष GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                भङ्ग;

            स्वभाव:
                भङ्ग;
        }
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; ++j) {
            यदि (!ग्रन्थिः->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_समुच्चयः, ग्रन्थिः->src[j])]) {
                अनुवर्त;
            }
            GGML_ASSERT(ग्रन्थिः->src[j]->type == GGML_TYPE_F32 || ग्रन्थिः->src[j]->type == GGML_TYPE_F16);
            ग्रन्थिः_needs_grad = true;
            भङ्ग;
        }
        यदि (!ग्रन्थिः_needs_grad) {
            अनुवर्त;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!ग्रन्थिः->view_src || ग्रन्थिः->op == GGML_OP_CPY || ग्रन्थिः->op == GGML_OP_समुच्चयः_पङ्क्तिकाः || ग्रन्थिः->op == GGML_OP_VIEW ||
            ग्रन्थिः->op == GGML_OP_RESHAPE || ग्रन्थिः->op == GGML_OP_PERMUTE || ग्रन्थिः->op == GGML_OP_TRANSPOSE);

        ध्रुव size_t ihash = ggml_hash_find(&cgraph->visited_hash_समुच्चयः, ग्रन्थिः);
        GGML_ASSERT(ihash != GGML_HASHSET_पूर्णम्);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, ihash));
        यदि (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } अन्यथा यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ग्रन्थिः->ne);
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

स्थिर शून्य * incr_ptr_aligned(शून्य ** p, size_t आकार, size_t align) {
    शून्य * ptr = *p;
    ptr = (शून्य *) GGML_PAD((uintptr_t) ptr, align);
    *p = (शून्य *) ((वर्ण *) ptr + आकार);
    प्रत्यागम ptr;
}

स्थिर size_t ggml_graph_nbytes(size_t आकार, bool grads) {
    size_t hash_आकार = ggml_hash_आकार(आकार * 2);
    शून्य * p = (वर्ण *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, परिमाण(संरचना ggml_cgraph), 1);
    incr_ptr_aligned(&p, आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_आकार * परिमाण(int32_t), परिमाण(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // hash keys
    यदि (grads) {
        incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_आकार(hash_आकार) * परिमाण(ggml_bitset_t), परिमाण(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    प्रत्यागम nbytes - 1024;
}

size_t ggml_graph_overhead_custom(size_t आकार, bool grads) {
    प्रत्यागम GGML_OBJECT_आकार + GGML_PAD(ggml_graph_nbytes(आकार, grads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(शून्य) {
    प्रत्यागम ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_आकार, false);
}

संरचना ggml_cgraph * ggml_new_graph_custom(संरचना ggml_context * ctx, size_t आकार, bool grads) {
    ध्रुव size_t obj_आकार = ggml_graph_nbytes(आकार, grads);
    संरचना ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_आकार);
    संरचना ggml_cgraph * cgraph = (संरचना ggml_cgraph *) ((वर्ण *) ctx->mem_बफर + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_आकार = ggml_hash_आकार(आकार * 2);

    शून्य * p = cgraph + 1;

    संरचना ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, आकार      * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    संरचना ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, आकार      * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_आकार * परिमाण(int32_t), परिमाण(int32_t));
    संरचना ggml_tensor ** hash_कुञ्चिकाः_ptr  =         incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *));
    संरचना ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)) : रिक्त;
    संरचना ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_आकार * परिमाण(संरचना ggml_tensor *), परिमाण(संरचना ggml_tensor *)) : रिक्त;

    ggml_bitset_t * hash_प्रयुक्तम् = incr_ptr_aligned(&p, ggml_bitset_आकार(hash_आकार) * परिमाण(ggml_bitset_t), परिमाण(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    प्रतिज्ञा(obj_आकार == (size_t)((वर्ण *)p - (वर्ण *)cgraph));

    *cgraph = (संरचना ggml_cgraph) {
        /*.size         =*/ आकार,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_आकार, hash_प्रयुक्तम्, hash_कुञ्चिकाः_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_क्रमः_वामः_गन्तव्यम्_दक्षिणः,
        /*.uid          =*/ 0,
    };

    ggml_hash_समुच्चयः_reset(&cgraph->visited_hash_समुच्चयः);
    यदि (grads) {
        स्मृतिं_पूरय(cgraph->grads,     0, hash_आकार*परिमाण(संरचना ggml_tensor *));
        स्मृतिं_पूरय(cgraph->grad_accs, 0, hash_आकार*परिमाण(संरचना ggml_tensor *));
    }

    प्रत्यागम cgraph;
}

संरचना ggml_cgraph * ggml_new_graph(संरचना ggml_context * ctx) {
    प्रत्यागम ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_आकार, false);
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
        /*.visited_hash_set =*/ cgraph0->visited_hash_समुच्चयः,
        /*.order            =*/ cgraph0->क्रमः,
        /*.uid              =*/ 0
    };

    प्रत्यागम cgraph;
}

शून्य ggml_graph_cpy(संरचना ggml_cgraph * src, संरचना ggml_cgraph * dst) {
    GGML_ASSERT(dst->आकार >= src->n_leafs);
    GGML_ASSERT(dst->आकार >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_समुच्चयः.आकार >= src->visited_hash_समुच्चयः.आकार);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->क्रमः   = src->क्रमः;

    प्रति (पूर्णाङ्क i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    प्रति (size_t i = 0; i < src->visited_hash_समुच्चयः.आकार; ++i) {
        // copy all hashset keys (tensors) that are in use
        यदि (ggml_bitset_get(src->visited_hash_समुच्चयः.प्रयुक्तम्, i)) {
            size_t new_hash_pos = ggml_hash_निवेशय(&dst->visited_hash_समुच्चयः, src->visited_hash_समुच्चयः.कुञ्चिकाः[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    यदि (dst->grads) {
        स्मृतिं_पूरय(dst->grads,     0, dst->visited_hash_समुच्चयः.आकार*परिमाण(संरचना ggml_tensor *));
        स्मृतिं_पूरय(dst->grad_accs, 0, dst->visited_hash_समुच्चयः.आकार*परिमाण(संरचना ggml_tensor *));
    }
    यदि (src->grads) {
        GGML_ASSERT(dst->grads     != रिक्त);
        GGML_ASSERT(dst->grad_accs != रिक्त);
        प्रति (पूर्णाङ्क i = 0; i < src->n_nodes; ++i) {
            ध्रुव size_t igrad_src = ggml_hash_find(&src->visited_hash_समुच्चयः, src->nodes[i]);
            ध्रुव size_t igrad_dst = ggml_hash_find(&dst->visited_hash_समुच्चयः, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_पूर्णम्);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_समुच्चयः.प्रयुक्तम्, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_पूर्णम्);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_समुच्चयः.प्रयुक्तम्, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

संरचना ggml_cgraph * ggml_graph_dup(संरचना ggml_context * ctx, संरचना ggml_cgraph * cgraph, bool force_grads) {
    संरचना ggml_cgraph * परिणाम = ggml_new_graph_custom(ctx, cgraph->आकार, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, परिणाम);
    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_समुच्चयः_zero(संरचना ggml_tensor * tensor) {
    यदि (ggml_is_रिक्तानि(tensor)) {
        प्रत्यागम tensor;
    }
    यदि (tensor->बफर) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } अन्यथा {
        GGML_ASSERT(tensor->दत्त);
        स्मृतिं_पूरय(tensor->दत्त, 0, ggml_nbytes(tensor));
    }
    प्रत्यागम tensor;
}

शून्य ggml_graph_reset(संरचना ggml_cgraph * cgraph) {
    यदि (!cgraph) {
        प्रत्यागम;
    }
    GGML_ASSERT(cgraph->grads != रिक्त);

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः     = cgraph->nodes[i];
        संरचना ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, ग्रन्थिः);

        यदि (ग्रन्थिः->op == GGML_OP_OPT_चरणः_ADAMW) {
            // clear momenta
            ggml_समुच्चयः_zero(ग्रन्थिः->src[2]);
            ggml_समुच्चयः_zero(ग्रन्थिः->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        यदि (grad_acc) {
            यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                ध्रुव भिन्न onef = 1.0f;
                यदि (grad_acc->बफर) {
                    ggml_backend_tensor_समुच्चयः(grad_acc, &onef, 0, परिमाण(भिन्न));
                } अन्यथा {
                    GGML_ASSERT(grad_acc->दत्त);
                    *((भिन्न *) grad_acc->दत्त) = onef;
                }
            } अन्यथा {
                ggml_समुच्चयः_zero(grad_acc);
            }
        }
    }
}

शून्य ggml_graph_clear(संरचना ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_समुच्चयः_reset(&cgraph->visited_hash_समुच्चयः);
}

पूर्णाङ्क ggml_graph_आकार(संरचना ggml_cgraph * cgraph) {
    प्रत्यागम cgraph->आकार;
}

संरचना ggml_tensor * ggml_graph_ग्रन्थिः(संरचना ggml_cgraph * cgraph, पूर्णाङ्क i) {
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

शून्य ggml_graph_सङ्कलय_ग्रन्थिः(संरचना ggml_cgraph * cgraph, संरचना ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->आकार > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

संरचना ggml_tensor * ggml_graph_get_tensor(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव वर्ण * name) {
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
        संरचना ggml_tensor * पर्णम् = cgraph->leafs[i];

        यदि (वाक्यं_तुलय(पर्णम्->name, name) == 0) {
            प्रत्यागम पर्णम्;
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        यदि (वाक्यं_तुलय(ग्रन्थिः->name, name) == 0) {
            प्रत्यागम ग्रन्थिः;
        }
    }

    प्रत्यागम रिक्त;
}

संरचना ggml_tensor * ggml_graph_get_grad(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * ग्रन्थिः) {
    ध्रुव size_t igrad = ggml_hash_find(&cgraph->visited_hash_समुच्चयः, ग्रन्थिः);
    प्रत्यागम igrad != GGML_HASHSET_पूर्णम् && ggml_bitset_get(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, igrad) && cgraph->grads ? cgraph->grads[igrad] : रिक्त;
}

संरचना ggml_tensor * ggml_graph_get_grad_acc(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * ग्रन्थिः) {
    ध्रुव size_t igrad = ggml_hash_find(&cgraph->visited_hash_समुच्चयः, ग्रन्थिः);
    प्रत्यागम igrad != GGML_HASHSET_पूर्णम् && ggml_bitset_get(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : रिक्त;
}

शून्य ggml_graph_print(ध्रुव संरचना ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                ग्रन्थिः->ne[0], ग्रन्थिः->ne[1], ग्रन्थिः->ne[2],
                ggml_op_name(ग्रन्थिः->op), (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, ग्रन्थिः) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                ग्रन्थिः->ne[0], ग्रन्थिः->ne[1],
                ggml_op_name(ग्रन्थिः->op),
                ggml_get_name(ग्रन्थिः));
    }

    GGML_LOG_INFO("========================================\n");
}

स्थिर पूर्णाङ्क ggml_ग्रन्थिः_list_find_tensor(ध्रुव संरचना ggml_cgraph * cgraph,
                                      ध्रुव पूर्णाङ्क *                idxs,
                                      पूर्णाङ्क                        गणनम्,
                                      ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    प्रति (पूर्णाङ्क i = 0; i < गणनम्; ++i) {
        ध्रुव पूर्णाङ्क ग्रन्थिः_idx = idxs[i];

        यदि (ग्रन्थिः_idx >= cgraph->n_nodes) {
            प्रत्यागम -1;
        }
        यदि (cgraph->nodes[ग्रन्थिः_idx] == tensor) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम -1;
}

स्थिर bool ggml_is_constant(ध्रुव संरचना ggml_tensor * tensor) {
    प्रत्यागम tensor->बफर != रिक्त && ggml_backend_बफर_get_usage(tensor->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(ध्रुव संरचना ggml_cgraph * cgraph,
                                ध्रुव पूर्णाङ्क *                ग्रन्थिः_idxs,
                                पूर्णाङ्क                        गणनम्,
                                ध्रुव गणना ggml_op *       ops,
                                ध्रुव पूर्णाङ्क *                outputs,
                                पूर्णाङ्क                        अङ्कः_outputs) {
    GGML_ASSERT(outputs && अङ्कः_outputs > 0);

    प्रति (पूर्णाङ्क i = 0; i < गणनम्; ++i) {
        यदि (ग्रन्थिः_idxs[i] >= cgraph->n_nodes) {
            प्रत्यागम false;
        }

        ध्रुव संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idxs[i]];

        यदि (ग्रन्थिः->op != ops[i]) {
            प्रत्यागम false;
        }

        यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            प्रत्यागम false;
        }

        यदि (ggml_ग्रन्थिः_list_find_tensor(cgraph, outputs, अङ्कः_outputs, ग्रन्थिः) != -1) {
            अनुवर्त;
        }

        यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_OUTPUT) {
            प्रत्यागम false;
        }

        पूर्णाङ्क subgraph_उपयुङ्क्ते = 0;
        प्रति (पूर्णाङ्क j = i + 1; j < गणनम्; ++j) {
            ध्रुव संरचना ggml_tensor * other_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idxs[j]];
            प्रति (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                यदि (other_ग्रन्थिः->src[src_idx] == ग्रन्थिः) {
                    subgraph_उपयुङ्क्ते++;
                }
            }
        }

        यदि (subgraph_उपयुङ्क्ते != ggml_ग्रन्थिः_get_use_गणनम्(cgraph, ग्रन्थिः_idxs[i])) {
            प्रत्यागम false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        संरचना ggml_tensor * view_src = ग्रन्थिः->view_src;
        यावत् (view_src) {
            यदि (ggml_ग्रन्थिः_list_find_tensor(cgraph, ग्रन्थिः_idxs, गणनम्, view_src) == -1 && !ggml_is_constant(view_src)) {
                प्रत्यागम false;
            }
            view_src = view_src->view_src;
        }
    }

    प्रत्यागम true;
}

// check if node is part of the graph
स्थिर bool ggml_graph_find(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * ग्रन्थिः) {
    यदि (cgraph == रिक्त) {
        प्रत्यागम true;
    }

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        यदि (cgraph->nodes[i] == ग्रन्थिः) {
            प्रत्यागम true;
        }
    }

    प्रत्यागम false;
}

स्थिर संरचना ggml_tensor * ggml_graph_get_parent(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव संरचना ggml_tensor * ग्रन्थिः) {
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * parent = cgraph->nodes[i];
        संरचना ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        यदि (grad == ग्रन्थिः) {
            प्रत्यागम parent;
        }
    }

    प्रत्यागम रिक्त;
}

स्थिर शून्य ggml_graph_dump_dot_ग्रन्थिः_edge(FILE * fp, ध्रुव संरचना ggml_cgraph * gb, संरचना ggml_tensor * ग्रन्थिः, संरचना ggml_tensor * parent, ध्रुव वर्ण * label)  {
    संरचना ggml_tensor * gparent = ggml_graph_get_parent(gb, ग्रन्थिः);
    संरचना ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (शून्य *) gparent0 : (शून्य *) parent,
            gparent ? (शून्य *) gparent : (शून्य *) ग्रन्थिः,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

स्थिर शून्य ggml_graph_dump_dot_पर्णम्_edge(FILE * fp, संरचना ggml_tensor * ग्रन्थिः, संरचना ggml_tensor * parent, ध्रुव वर्ण * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (शून्य *) parent,
            (शून्य *) ग्रन्थिः,
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
        संरचना ggml_tensor * ग्रन्थिः = gb->nodes[i];
        संरचना ggml_tensor * grad = ggml_graph_get_grad(gb, ग्रन्थिः);

        यदि (ggml_graph_get_parent(gb, ग्रन्थिः) != रिक्त) {
            अनुवर्त;
        }

        यदि (ग्रन्थिः->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, परिमाण(color), "yellow");
        } अन्यथा यदि (grad) {
            यदि (ggml_graph_find(cgraph, ग्रन्थिः)) {
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
                (शून्य *) ग्रन्थिः, color);

        यदि (वाक्यदैर्घ्यम्(ग्रन्थिः->name) > 0) {
            fprintf(fp, "%s (%s)|", ग्रन्थिः->name, ggml_type_name(ग्रन्थिः->type));
        } अन्यथा {
            fprintf(fp, "(%s)|", ggml_type_name(ग्रन्थिः->type));
        }

        यदि (ggml_is_matrix(ग्रन्थिः)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, ग्रन्थिः->ne[0], ग्रन्थिः->ne[1], ggml_op_चिह्नम्(ग्रन्थिः->op));
        } अन्यथा {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, ग्रन्थिः->ne[0], ग्रन्थिः->ne[1], ग्रन्थिः->ne[2], ggml_op_चिह्नम्(ग्रन्थिः->op));
        }

        यदि (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_चिह्नम्(grad->op));
        } अन्यथा {
            fprintf(fp, "\"; ]\n");
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_leafs; i++) {
        संरचना ggml_tensor * ग्रन्थिः = gb->leafs[i];

        snprintf(color, परिमाण(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (शून्य *) ग्रन्थिः, color);

        यदि (वाक्यदैर्घ्यम्(ग्रन्थिः->name) > 0) {
            fprintf(fp, "%s (%s)|", ग्रन्थिः->name, ggml_type_name(ग्रन्थिः->type));
        } अन्यथा {
            fprintf(fp, "(%s)|", ggml_type_name(ग्रन्थिः->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, ग्रन्थिः->ne[0], ग्रन्थिः->ne[1]);
        यदि (ggml_nelements(ग्रन्थिः) < 5 && ग्रन्थिः->दत्त != रिक्त) {
            fprintf(fp, " | (");
            प्रति (पूर्णाङ्क j = 0; j < ggml_nelements(ग्रन्थिः); j++) {
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
                यदि (j < ggml_nelements(ग्रन्थिः) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = gb->nodes[i];

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (ग्रन्थिः->src[j]) {
                वर्ण label[16];
                snprintf(label, परिमाण(label), "src %d", j);
                ggml_graph_dump_dot_ग्रन्थिः_edge(fp, gb, ग्रन्थिः, ग्रन्थिः->src[j], label);
            }
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < gb->n_leafs; i++) {
        संरचना ggml_tensor * ग्रन्थिः = gb->leafs[i];

        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (ग्रन्थिः->src[j]) {
                वर्ण label[16];
                snprintf(label, परिमाण(label), "src %d", j);
                ggml_graph_dump_dot_पर्णम्_edge(fp, ग्रन्थिः, ग्रन्थिः->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

शून्य ggml_समुच्चयः_निवेशः(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_निवेशः;
}

शून्य ggml_समुच्चयः_output(संरचना ggml_tensor * tensor) {
    प्रति (संरचना ggml_tensor * cur = tensor; cur != रिक्त; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

शून्य ggml_समुच्चयः_param(संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

शून्य ggml_समुच्चयः_loss(संरचना ggml_tensor * tensor) {
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

    ggml_critical_section_अन्तः();
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

    ggml_critical_section_अन्तः();
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
               int64_t   n_प्रत्येकम्_row,
           ध्रुव भिन्न * imatrix) {
    ध्रुव int64_t n = nrows * n_प्रत्येकम्_row;

    यदि (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != रिक्त);
    }

    GGML_ASSERT(start % type_traits[type].blck_आकार == 0);
    GGML_ASSERT(start % n_प्रत्येकम्_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    ध्रुव size_t start_row = start / n_प्रत्येकम्_row;
    ध्रुव size_t row_आकार  = ggml_row_आकार(type, n_प्रत्येकम्_row);

    size_t परिणाम = 0;

    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:    परिणाम = quantize_q1_0   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q2_0:    परिणाम = quantize_q2_0   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_0:    परिणाम = quantize_q4_0   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_1:    परिणाम = quantize_q4_1   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_0:    परिणाम = quantize_q5_0   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_1:    परिणाम = quantize_q5_1   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q8_0:    परिणाम = quantize_q8_0   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_MXFP4:   परिणाम = quantize_mxfp4  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_NVFP4:   परिणाम = quantize_nvfp4  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q2_K:    परिणाम = quantize_q2_K   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q3_K:    परिणाम = quantize_q3_K   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q4_K:    परिणाम = quantize_q4_K   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q5_K:    परिणाम = quantize_q5_K   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_Q6_K:    परिणाम = quantize_q6_K   (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_TQ1_0:   परिणाम = quantize_tq1_0  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_TQ2_0:   परिणाम = quantize_tq2_0  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_XXS: परिणाम = quantize_iq2_xxs(src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_XS:  परिणाम = quantize_iq2_xs (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS: परिणाम = quantize_iq3_xxs(src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:   परिणाम = quantize_iq3_s  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ2_S:   परिणाम = quantize_iq2_s  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ1_S:   परिणाम = quantize_iq1_s  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ1_M:   परिणाम = quantize_iq1_m  (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL:  परिणाम = quantize_iq4_nl (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_IQ4_XS:  परिणाम = quantize_iq4_xs (src + start, (वर्ण *) dst + start_row * row_आकार, nrows, n_प्रत्येकम्_row, imatrix); भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                size_t elemsize = परिमाण(ggml_fp16_t);
                ggml_fp32_गन्तव्यम्_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                परिणाम = n * elemsize;
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                size_t elemsize = परिमाण(ggml_bf16_t);
                ggml_fp32_गन्तव्यम्_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                परिणाम = n * elemsize;
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                size_t elemsize = परिमाण(भिन्न);
                परिणाम = n * elemsize;
                स्मृतिं_प्रतिलिख((uint8_t *)dst + start * elemsize, src + start, परिणाम);
            } भङ्ग;
        स्वभाव:
            प्रतिज्ञा(false);
    }

    GGML_ASSERT(परिणाम == nrows * row_आकार);

    प्रत्यागम परिणाम;
}

////////////////////////////////////////////////////////////////////////////////

शून्य ggml_log_get(ggml_log_callback * log_callback, शून्य ** user_दत्त) {
    *log_callback = g_logger_state.log_callback;
    *user_दत्त    = g_logger_state.log_callback_user_दत्त;
}

शून्य ggml_log_समुच्चयः(ggml_log_callback log_callback, शून्य * user_दत्त) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_दत्त = user_दत्त;
}

शून्य ggml_threadpool_params_init(संरचना ggml_threadpool_params * p, पूर्णाङ्क n_तन्तवः) {
    p->n_तन्तवः  = n_तन्तवः;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    स्मृतिं_पूरय(p->cpumask, 0, GGML_MAX_N_तन्तवः); // all-zero means use the default affinity (usually inherited)
}

संरचना ggml_threadpool_params ggml_threadpool_params_default(पूर्णाङ्क n_तन्तवः) {
    संरचना ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_तन्तवः);
    प्रत्यागम p;
}

bool ggml_threadpool_params_मेलनम्(ध्रुव संरचना ggml_threadpool_params * p0, ध्रुव संरचना ggml_threadpool_params * p1) {
    यदि (p0->n_तन्तवः  != p1->n_तन्तवः  ) प्रत्यागम false;
    यदि (p0->prio       != p1->prio       ) प्रत्यागम false;
    यदि (p0->poll       != p1->poll       ) प्रत्यागम false;
    यदि (p0->strict_cpu != p1->strict_cpu ) प्रत्यागम false;
    प्रत्यागम memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_तन्तवः) == 0;
}
